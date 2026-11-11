use crate::chip::Chip;
use crate::expr::{Constraint, Variable};
use crate::starkom::proto::plonk::v10 as proto;
use crate::utils::{self, encode_cell, encode_usize, padded_circuit_size};
use crate::witness::{Cell, CellOffset, Partitioner, Witness, WitnessView};
use anyhow::{Result, anyhow};
use primitive_types::H256;
use prost::Name;
use starkom_ff::{Field, Field256};
use starkom_pcs::{self as pcs, hash::Hasher};
use starkom_poly::Polynomial;
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Default blowup factor (16) in logarithmic form.
///
/// Used with the underlying PCS to compute low-degree extensions.
pub const OPTIONS_DEFAULT_BLOWUP_LOG2: usize = 4;

const COMMIT_INDEX_CIRCUIT: usize = 0;
const COMMIT_INDEX_WITNESS: usize = 1;
const COMMIT_INDEX_PERMUTATION_ARGUMENT: usize = 2;
const COMMIT_INDEX_QUOTIENT: usize = 3;
const NUM_COMMIT_INDICES: usize = 4;

// Domain separator tags used for various Fiat-Shamir challenges.

static DST_ALPHA: LazyLock<H256> =
    LazyLock::new(|| utils::make_dst(b"starkom/plonk/challenge/alpha"));

static DST_BETA: LazyLock<H256> =
    LazyLock::new(|| utils::make_dst(b"starkom/plonk/challenge/beta"));

static DST_GAMMA: LazyLock<H256> =
    LazyLock::new(|| utils::make_dst(b"starkom/plonk/challenge/gamma"));

static DST_DELTA: LazyLock<H256> =
    LazyLock::new(|| utils::make_dst(b"starkom/plonk/challenge/delta"));

static DST_PSI: LazyLock<H256> = LazyLock::new(|| utils::make_dst(b"starkom/plonk/challenge/psi"));

static DST_XI: LazyLock<H256> = LazyLock::new(|| utils::make_dst(b"starkom/plonk/challenge/xi"));

/// Builds the set of all rotations used in a circuit.
///
/// NOTE: we're always including 0 and +1 because we need to open xi and xi*omega regardless;
/// they're needed for the final algebraic check and for the shifted permutation argument,
/// respectively.
fn get_rotation_set<'a, F: Field>(
    gates: impl IntoIterator<Item = &'a Constraint<F>>,
) -> BTreeSet<isize> {
    gates
        .into_iter()
        .flat_map(|constraint| {
            constraint
                .get_free_variables()
                .iter()
                .map(Variable::rotation)
                .collect::<BTreeSet<isize>>()
                .into_iter()
        })
        .chain([0, 1])
        .collect::<BTreeSet<isize>>()
}

/// Calculates the degree bound of the PLONK quotient, typically much higher than the circuit's
/// general [degree bound](`Circuit::degree_bound`) `N` because the constraint equations involve
/// several polynomial multiplications, such as the gate selectors multiplied by the gate
/// constraints combined with the witness columns.
///
/// The algorithm uses the formula `(N - 1) * E`, where `E = max(max_gate_degree, 3)`. The rationale
/// behind it is:
///
/// * each column has degree less than or equal to `N - 1`;
/// * the grand gate constraint has degree less than or equal to
///   `(N - 1) * (1 + max_gate_degree)` (the selector contributes one factor, degree composition
///   with the constraint columns contributes `max_gate_degree` more);
/// * each helper authentication constraint of the permutation argument is the product of three
///   polynomials of degree less than `N` (the helper column, the numerator factor, and the
///   denominator factor) times the [circuit area selector](`Circuit::circuit_area_selector`) of
///   degree `N - num_rows`, for a total of `3 * (N - 1) + N - num_rows`; every circuit has at least
///   one row, so that never exceeds `4 * (N - 1)`, and covering it with the `(N - 1) * (1 + E)`
///   bound below requires `1 + E >= 4`, hence `E >= 3`;
/// * the anchor and public cell constraints have degree less than or equal to `2 * (N - 1)`, and
///   the recurrence constraint is linear in the committed columns even after being gated, so none
///   of them exceeds the above;
/// * the grand PLONK constraint therefore has degree less than or equal to `(N - 1) * (1 + E)`;
/// * dividing that by the zero polynomial (`x^N - 1`, degree-N) yields a quotient with degree
///   `(N - 1) * (1 + E) - N`;
/// * so the degree bound of the quotient is `(N - 1) * (1 + E) - N + 1`
/// * ... which simplifies to `(N - 1) * E`.
fn quotient_degree_bound<'a, F: Field>(
    degree_bound: usize,
    gate_constraints: impl IntoIterator<Item = &'a Constraint<F>>,
) -> usize {
    let max_gate_degree = gate_constraints
        .into_iter()
        .map(|constraint| constraint.get_degree())
        .max()
        .unwrap_or(0);
    (degree_bound - 1) * std::cmp::max(max_gate_degree, 3)
}

/// Calculates the degree bound of the polynomials used to mask the
/// [quotient split](`Circuit::split_quotient`).
fn mask_degree_bound<'a, F: Field>(
    security_bits: usize,
    blowup_log2: usize,
    grinding_bits: usize,
    gate_constraints: impl IntoIterator<Item = &'a Constraint<F>>,
) -> usize {
    get_rotation_set(gate_constraints).len()
        + pcs::num_queries(security_bits, blowup_log2, grinding_bits)
        + 1
}

/// Returns the number of witness randomizers of a circuit, i.e. the uniformly random polynomials
/// committed after the witness columns in the witness batch.
///
/// The witness batch is committed over the base field `F`, so every leaf of its Merkle tree holds
/// one element of `F` per polynomial, and the Merkle paths reveal the hashes of unopened leaves. A
/// blinded circuit needs ~256 bits of entropy in every leaf to make brute-forcing them infeasible,
/// so if it has fewer witness columns than the extension degree `T` of `G` over `F` (e.g. 4 for
/// Goldilocks^4, 1 when F and G coincide) it pads the batch with random polynomials up to `T`.
/// They take part in no constraint and the verifier ignores their openings.
fn num_witness_randomizers<G: Field256>(blinded: bool, num_columns: usize) -> usize {
    if blinded {
        G::EXTENSION_DEGREE.saturating_sub(num_columns)
    } else {
        0
    }
}

/// Builds a polynomial vanishes on the padding area. Used to switch the permutation argument off in
/// the padding area while keeping it active in the circuit area, hence the name.
///
/// REQUIRES: `0 < num_rows <= degree_bound`. Note that the padding area may be empty, e.g. in an
/// unblinded circuit with a power-of-2 number of rows.
///
/// Note that this function computes the vanishing polynomial of the padding area rather than the
/// Lagrange indicator of the witness rows: the two gate constraints identically, but this one has
/// degree `N - num_rows` rather than `N - 1`.
fn make_circuit_area_selector<F: Field>(num_rows: usize, degree_bound: usize) -> Polynomial<F> {
    let padding_size = degree_bound - num_rows;
    let omega: F = Polynomial::<F>::domain_element2(1, degree_bound).into();
    let mut coefficients: Vec<F> = Vec::with_capacity(padding_size);
    coefficients.push(F::ONE);
    let mut power = omega.pow_small(num_rows);
    for i in 0..padding_size {
        coefficients.push(F::ZERO);
        for j in (0..=i).rev() {
            let previous = coefficients[j];
            coefficients[j + 1] -= power * previous;
        }
        power *= omega;
    }
    coefficients.reverse();
    Polynomial::with_coefficients(coefficients)
}

/// Evaluates the [Lagrange basis `L_0`](`Polynomial::lagrange0_2`) at a point, without
/// materializing the polynomial.
fn lagrange0<F: Field256>(x: F, n: usize) -> F {
    (x.pow_small(n) - F::ONE) * (F::from(n as u64) * (x - F::ONE)).invert_unwrap()
}

/// Derives the Fiat-Shamir challenge used to batch the public cell constraints of all columns in a
/// single constraint. See [`Circuit::build_public_cell_constraint`] for details.
///
/// NOTE: the claimed public values MUST be part of the transcript. The witness polynomials are
/// domain-shifted by the DEEP layer in the PCS before getting committed to any Merkle tree, so none
/// of the Merkle roots provided by the PCS are bound to the public inputs.
fn public_cell_challenge<F: Field, G: Field256, H: Hasher<G>>(
    transcript_hash: H256,
    public_inputs: &BTreeMap<Cell, F>,
) -> G {
    let mut transcript = Vec::with_capacity(public_inputs.len() * 2 + 2);
    transcript.push(transcript_hash);
    transcript.push(encode_usize(public_inputs.len()));
    for (cell, value) in public_inputs {
        transcript.push(encode_cell(*cell));
        transcript.push(H256::from_slice(&value.to_u256().to_big_endian()));
    }
    H::challenge(*DST_PSI, transcript.as_slice())
}

fn pcs_config_to_proto(config: &pcs::Options) -> proto::PcsConfig {
    proto::PcsConfig {
        degree_bound: config.degree_bound as u64,
        security_bits: config.security_bits as u32,
        blowup_log2: config.blowup_log2 as u32,
        fold_rate: config.fold_rate as u32,
        max_rounds: config.max_rounds as u64,
        grinding_bits: config.grinding_bits as u32,
    }
}

fn pcs_config_from_proto<F: Field>(proto: &proto::PcsConfig) -> Result<pcs::Options> {
    let config = pcs::Options {
        degree_bound: proto.degree_bound as usize,
        security_bits: proto.security_bits as usize,
        blowup_log2: proto.blowup_log2 as usize,
        fold_rate: proto.fold_rate as usize,
        max_rounds: proto.max_rounds as usize,
        grinding_bits: proto.grinding_bits as usize,
    };
    config.validate::<F>()?;
    Ok(config)
}

/// Circuit compilation & proving options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// Converts all constraints to canonical form using [`Constraint::canonicalize`].
    ///
    /// When disabled, proving errors out rather than attempting canonicalization if there are
    /// negative exponents.
    ///
    /// Canonicalization is carried out inside [`CircuitBuilder::build`].
    ///
    /// WARNING: canonicalized constraints may be more permissive than their original form because a
    /// negative exponent requires the variable to be different from zero. Starkom does not allow
    /// proving with negative exponents, so enable this flag only if your circuit is correctly
    /// constrained even when those variables are zero.
    pub canonicalize_constraints: bool,

    /// The target security level in bits for the underlying PCS (see
    /// [`pcs::Options::security_bits`]). Together with [`blowup_log2`](`Self::blowup_log2`) and
    /// [`grinding_bits`](`Self::grinding_bits`) it determines the number of FRI queries (see
    /// [`pcs::num_queries`]).
    ///
    /// NOTE: when the [`blind`](`Self::blind`) flag is enabled every FRI query requires extra
    /// blinding rows, so a higher security level may increase the padded circuit size.
    pub security_bits: usize,

    /// Log2 of the blowup factor used to compute the low-degree extensions for the underlying PCS.
    pub blowup_log2: usize,

    /// Number of FRI folds per round, i.e. the base-2 logarithm of the folding arity (see
    /// [`pcs::Options::fold_rate`]). Must be greater than 0.
    pub fri_fold_rate: usize,

    /// Maximum number of FRI folding rounds (see [`pcs::Options::max_rounds`]). The polynomial left
    /// after the last round is sent in the clear, so 0 sends the whole DEEP quotient.
    ///
    /// NOTE: specifying 0 does not impact zero-knowledge. If the [`blind`](`Self::blind`) flag is
    /// enabled it will be infeasible for an external observer to derive the original witness
    /// regardless of the number of FRI rounds. `max_fri_rounds` and
    /// [`fri_fold_rate`](`Self::fri_fold_rate`) are only for proof size optimization.
    pub max_fri_rounds: usize,

    /// Number of proof-of-work bits required by the underlying PCS (see
    /// [`pcs::Options::grinding_bits`]). Every grinding bit lowers the number of FRI queries, and
    /// therefore the number of blinding rows of blinded circuits. Must not exceed 256.
    pub grinding_bits: usize,

    /// Indicates whether or not blinding must be added to the circuit. In other words, this flag
    /// toggles the zero-knowledge feature of Starkom.
    ///
    /// Blinding is disabled by default because it adds extra costs due to increasing the minimum
    /// padding area and adding several columns for randomization.
    pub blind: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            canonicalize_constraints: false,
            security_bits: 128,
            blowup_log2: OPTIONS_DEFAULT_BLOWUP_LOG2,
            fri_fold_rate: 1,
            max_fri_rounds: usize::MAX,
            grinding_bits: 0,
            blind: false,
        }
    }
}

pub trait CircuitView<F: Field, G: Field256<BaseField = F>> {
    /// Returns a reference to the [`CircuitBuilder`].
    fn builder(&self) -> &CircuitBuilder<F, G>;

    /// Returns a mutable reference to the [`CircuitBuilder`].
    fn builder_mut(&mut self) -> &mut CircuitBuilder<F, G>;

    /// Returns the number of columns included in the view.
    ///
    /// If this is the root view, that is the raw [`CircuitBuilder`] instance, the width is
    /// unbounded and `None` is returned.
    fn width(&self) -> Option<usize>;

    /// Returns the number of rows included in the view.
    ///
    /// If this is the root view, that is the raw [`CircuitBuilder`] instance, the height is
    /// unbounded and `None` is returned.
    fn height(&self) -> Option<usize>;

    /// Returns the row offset of the view (0 for the [`CircuitBuilder`] itself).
    ///
    /// This is always an absolute value even for transitive sub-views. It is not relative to
    /// the parent view.
    fn row_offset(&self) -> usize;

    /// Returns the column offset of the view (0 for the [`CircuitBuilder`] itself).
    ///
    /// This is always an absolute value even for transitive sub-views. It is not relative to
    /// the parent view.
    fn column_offset(&self) -> usize;

    /// Returns [`Self::row_offset()`] and [`Self::column_offset()`] as a [`Cell`].
    fn root_cell(&self) -> Cell {
        Cell::new(self.row_offset(), self.column_offset())
    }

    /// Creates a [`Cell`] relative to this view.
    ///
    /// For example, if this view is at row offset 3 and column offset 5, then `cell(6, 2)` will
    /// return the cell at row 9 and column 7.
    fn cell(&self, row_offset: impl CellOffset, column_offset: impl CellOffset) -> Cell {
        let row = self.row_offset() as isize + row_offset.into_offset();
        let column = self.column_offset() as isize + column_offset.into_offset();
        assert!(row >= 0);
        assert!(column >= 0);
        Cell::new(row as usize, column as usize)
    }

    /// Indicates whether the specified `cell` is contained in this view.
    ///
    /// Always true for the root [`CircuitBuilder`].
    fn contains_cell(&self, cell: Cell) -> bool {
        let row_offset = self.row_offset();
        let column_offset = self.column_offset();
        if cell.row() < row_offset || cell.column() < column_offset {
            return false;
        }
        if let Some(width) = self.width() {
            assert!(cell.column() < column_offset + width);
        }
        if let Some(height) = self.height() {
            assert!(cell.row() < row_offset + height);
        }
        true
    }

    /// Adds a gate to the circuit.
    ///
    /// NOTE: the function panics if the root cell of the gate lies outside the view. It is however
    /// not forbidden to refer to external cells from within the gate constraint by using rotations
    /// or overflowing column numbers. For example, it's okay to place a gate at row 0 and refer to
    /// `rvar(0, -1)`, or to place it at the rightmost column and refer to `var(1)`.
    fn add_gate(&mut self, row: usize, constraint: Constraint<F>) {
        let root_cell = self.cell(row, 0);
        assert!(self.contains_cell(root_cell));
        self.builder_mut().add_gate_internal(root_cell, constraint);
    }

    /// "Connects" two circuit [`Cell`]s, meaning they will be constrained to have the same value.
    fn connect(&mut self, cell1: Option<Cell>, cell2: Option<Cell>) {
        self.builder_mut().connect_internal(cell1, cell2);
    }

    /// Spawns a child `CircuitView` at the given coordinates.
    ///
    /// If `width` and/or `height` are specified they must not overflow the parent view. If they are
    /// not specified the child view will automatically extend to the full width and/or height of
    /// the parent view.
    ///
    /// The function will panic if the child view overflows the parent view.
    ///
    /// NOTE: the width and height of the root view are undefined because the [`CircuitBuilder`] is
    /// unbounded, so the `width` and `height` parameters MUST be specified when creating direct
    /// children of the [`CircuitBuilder`].
    fn sub(
        &mut self,
        row_offset: usize,
        column_offset: usize,
        width: Option<usize>,
        height: Option<usize>,
    ) -> impl CircuitView<F, G> {
        let width = width.unwrap_or(self.width().unwrap() - column_offset);
        let height = height.unwrap_or(self.height().unwrap() - row_offset);
        if let Some(parent_width) = self.width() {
            assert!(column_offset + width <= parent_width);
        }
        if let Some(parent_height) = self.height() {
            assert!(row_offset + height <= parent_height);
        }
        let row_offset = self.row_offset() + row_offset;
        let column_offset = self.column_offset() + column_offset;
        CircuitSectionBuilder::new(self.builder_mut(), row_offset, column_offset, width, height)
    }

    /// Spawns a child `CircuitView` at the given coordinates and runs the provided `callback` on
    /// it.
    ///
    /// `sub_fn` returns `self`, not the child view. The child view is only valid for the duration
    /// of the callback, while `self` is returned to make `sub_fn` chainable.
    ///
    /// See [`Self::sub`] for details about how the `width` and `height` parameters are treated.
    fn sub_fn(
        &mut self,
        row_offset: usize,
        column_offset: usize,
        width: Option<usize>,
        height: Option<usize>,
        callback: impl FnOnce(&mut CircuitSectionBuilder<F, G>),
    ) -> &mut Self {
        self.sub_fn_or(row_offset, column_offset, width, height, |view| {
            callback(view);
            Ok(())
        })
        .unwrap()
    }

    /// Like [`Self::sub_fn`] but the callback may return an error.
    fn sub_fn_or(
        &mut self,
        row_offset: usize,
        column_offset: usize,
        width: Option<usize>,
        height: Option<usize>,
        callback: impl FnOnce(&mut CircuitSectionBuilder<F, G>) -> Result<()>,
    ) -> Result<&mut Self> {
        let width = width.unwrap_or(self.width().unwrap() - column_offset);
        let height = height.unwrap_or(self.height().unwrap() - row_offset);
        if let Some(parent_width) = self.width() {
            assert!(column_offset + width <= parent_width);
        }
        if let Some(parent_height) = self.height() {
            assert!(row_offset + height <= parent_height);
        }
        let row_offset = self.row_offset() + row_offset;
        let column_offset = self.column_offset() + column_offset;
        callback(&mut CircuitSectionBuilder::new(
            self.builder_mut(),
            row_offset,
            column_offset,
            width,
            height,
        ))?;
        Ok(self)
    }

    /// Spawns a child `WitnessView` at the given coordinates and runs the provided `chip` on it.
    fn sub_chip<const I: usize, const O: usize>(
        &mut self,
        row_offset: usize,
        column_offset: usize,
        chip: &impl Chip<F, I, O>,
        inputs: [Option<Cell>; I],
    ) -> Result<[Option<Cell>; O]> {
        let chip_width = chip.width();
        let chip_height = chip.height();
        if let Some(parent_width) = self.width() {
            assert!(column_offset + chip_width <= parent_width);
        }
        if let Some(parent_height) = self.height() {
            assert!(row_offset + chip_height <= parent_height);
        }
        let row_offset = self.row_offset() + row_offset;
        let column_offset = self.column_offset() + column_offset;
        let mut child = CircuitSectionBuilder::new(
            self.builder_mut(),
            row_offset,
            column_offset,
            chip_width,
            chip_height,
        );
        chip.build(&mut child, inputs)
    }
}

#[derive(Debug)]
pub struct CircuitViewGenerator<'a, F: Field, G: Field256<BaseField = F>> {
    /// Reference to the [`CircuitBuilder`].
    builder: &'a mut CircuitBuilder<F, G>,

    /// Row offset where all sub-sections are rooted.
    row_offset: usize,

    /// Width of each sub-section.
    width: usize,

    /// Height of each sub-section.
    height: usize,

    /// Number of sub-sections to generate.
    count: usize,
}

impl<'a, F: Field, G: Field256<BaseField = F>> CircuitViewGenerator<'a, F, G> {
    pub fn get(&'a mut self, index: usize) -> CircuitSectionBuilder<'a, F, G> {
        assert!(index < self.count);
        CircuitSectionBuilder::new(
            self.builder,
            self.row_offset,
            self.width * index,
            self.width,
            self.height,
        )
    }
}

/// Describes an instance of a gate.
///
/// NOTE: this struct doesn't specify the row of the root cell where the gate was placed because
/// activating a gate at the correct rows is the gate selector's job. The column of the root cell,
/// on the other hand, is specified by [`Self::column_index`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GateInstance {
    /// Column of the root cell where the gate was placed.
    column_index: usize,

    /// Index of the gate selector polynomial within the [selector pool](`Circuit::selectors`).
    selector_index: usize,
}

/// Allows building PLONK [`Circuit`]s.
#[derive(Debug, Default, Clone)]
pub struct CircuitBuilder<F: Field, G: Field256<BaseField = F>> {
    /// Current number of rows in the circuit.
    num_rows: usize,

    /// Current number of columns in the circuit.
    num_columns: usize,

    /// The gates of the circuit, indexed by constraint.
    ///
    /// For every gate type (that is, for every unique gate constraint) this map associates the list
    /// of places where the gate has been instantiated. Such list is represented as an array of
    /// "root cells", each root cell being the reference cell for the gate instance: the row of the
    /// root cell corresponds to rotation 0 of all variables referenced by the gate, and the column
    /// corresponds to the column offset of the [`CircuitView`] used to instantiate the gate.
    ///
    /// NOTE: in order to minimize the number of different gate types stored in a circuit, the
    /// constraints stored in this map are _not_ [remapped](`Constraint::remap_variables`). This map
    /// basically keeps "raw gate types".
    gates: BTreeMap<Constraint<F>, Vec<Cell>>,

    /// Cell partitioning inferred from the connections made with [`Self::connect`].
    partitioner: Partitioner,

    /// List of cells that are revealed in the proofs.
    public_cells: BTreeSet<Cell>,

    _data: PhantomData<G>,
}

impl<F: Field, G: Field256<BaseField = F>> CircuitBuilder<F, G> {
    fn add_gate_internal(&mut self, mut root_cell: Cell, mut constraint: Constraint<F>) {
        {
            let min_column_index = constraint.get_min_column_index();
            root_cell = root_cell.remap(Cell::new(0, min_column_index));
            constraint = constraint.remap_variables(-(min_column_index as isize));
        }
        let row = root_cell.row();
        let column = root_cell.column();
        {
            self.num_rows = std::cmp::max(self.num_rows, row + 1);
            self.num_columns = std::cmp::max(self.num_columns, column + 1);
            for variable in constraint.get_free_variables() {
                let rotation = variable.rotation();
                self.num_rows = std::cmp::max(
                    self.num_rows,
                    if rotation < 0 {
                        assert!(rotation.unsigned_abs() <= row);
                        row - rotation.unsigned_abs()
                    } else {
                        row + rotation.unsigned_abs()
                    } + 1,
                );
                self.num_columns =
                    std::cmp::max(self.num_columns, column + variable.column_index() + 1);
            }
        }
        self.gates.entry(constraint).or_default().push(root_cell);
    }

    fn connect_internal(&mut self, cell1: Option<Cell>, cell2: Option<Cell>) {
        match (cell1, cell2) {
            (Some(cell1), Some(cell2)) => {
                self.num_rows = std::cmp::max(self.num_rows, cell1.row() + 1);
                self.num_rows = std::cmp::max(self.num_rows, cell2.row() + 1);
                self.num_columns = std::cmp::max(self.num_columns, cell1.column() + 1);
                self.num_columns = std::cmp::max(self.num_columns, cell2.column() + 1);
                self.partitioner.connect(cell1, cell2);
            }
            _ => {}
        }
    }

    /// Updates the list of witness rows that are revealed.
    ///
    /// This method drops any previously provided lists, so if it's called multiple times only the
    /// list provided in the last call is used.
    ///
    /// Ideally you should call this method only once after adding all gates, right before
    /// [`Self::build`].
    pub fn declare_public_cells<I: IntoIterator<Item = Cell>>(&mut self, cells: I) {
        self.public_cells = BTreeSet::from_iter(cells);
    }

    fn make_selector(degree_bound: usize, activation_row_set: BTreeSet<usize>) -> Polynomial<F> {
        let mut selector_values = vec![F::ZERO; degree_bound];
        for row in activation_row_set {
            selector_values[row] = F::ONE;
        }
        Polynomial::encode2(selector_values)
    }

    fn build_gates_and_selectors(
        &self,
        degree_bound: usize,
    ) -> (
        BTreeMap<Constraint<F>, BTreeSet<GateInstance>>,
        Vec<Polynomial<F>>,
    ) {
        // Keys are (constraint, column_index) pairs; values are activation row sets.
        let mut row_set_map: BTreeMap<(Constraint<F>, usize), BTreeSet<usize>> =
            BTreeMap::default();
        for (constraint, root_cells) in &self.gates {
            for root_cell in root_cells.as_slice() {
                let key = (constraint.clone(), root_cell.column());
                let row = root_cell.row();
                row_set_map.entry(key).or_default().insert(row);
            }
        }

        // Roughly the inverse of `row_set_map`: keys are activation row sets, values are the list
        // of (constraint, column_index) instances that activate at those rows.
        let mut gates_by_row_set: BTreeMap<BTreeSet<usize>, Vec<(Constraint<F>, usize)>> =
            BTreeMap::default();
        for ((constraint, column_index), row_set) in row_set_map {
            gates_by_row_set
                .entry(row_set)
                .or_default()
                .push((constraint, column_index));
        }

        let mut gates: BTreeMap<Constraint<F>, BTreeSet<GateInstance>> = BTreeMap::default();
        let mut selectors: Vec<Polynomial<F>> = vec![];

        for (selector_index, (activation_row_set, gate_instances)) in
            gates_by_row_set.into_iter().enumerate()
        {
            selectors.push(Self::make_selector(degree_bound, activation_row_set));
            for (constraint, column_index) in gate_instances {
                gates.entry(constraint).or_default().insert(GateInstance {
                    column_index,
                    selector_index,
                });
            }
        }

        (gates, selectors)
    }

    /// Compiles the circuit built so far into a [`Circuit`] object.
    pub fn build(mut self, options: Options) -> Result<Circuit<F, G>> {
        if options.canonicalize_constraints {
            let mut old_gates: BTreeMap<Constraint<F>, Vec<Cell>> = BTreeMap::default();
            std::mem::swap(&mut self.gates, &mut old_gates);
            for (constraint, mut root_cells) in old_gates {
                self.gates
                    .entry(constraint.canonicalize())
                    .or_default()
                    .append(&mut root_cells);
            }
        } else {
            for (constraint, _) in &self.gates {
                if !constraint.is_canonical() {
                    return Err(anyhow!("constraint `{}` is not canonical", constraint));
                }
            }
        }

        let degree_bound = padded_circuit_size::<F, G>(
            self.num_rows,
            options.security_bits,
            options.blowup_log2,
            options.grinding_bits,
            options.blind,
            self.gates.iter().flat_map(|(constraint, _)| {
                constraint
                    .get_free_variables()
                    .iter()
                    .map(Variable::rotation)
                    .collect::<BTreeSet<isize>>()
            }),
        );

        let (gates, selectors) = Self::build_gates_and_selectors(&self, degree_bound);

        let sigma_values: Vec<Vec<F>> = {
            let mut sigma = vec![F::ZERO; degree_bound * self.num_columns];
            let omega = Polynomial::<G>::domain_element2(1, degree_bound);
            let mut k = F::ONE;
            for i in 0..self.num_columns {
                let offset = i * degree_bound;
                sigma[offset] = k;
                for j in 1..degree_bound {
                    sigma[offset + j] = sigma[offset + j - 1] * omega;
                }
                k *= F::MULTIPLICATIVE_GENERATOR;
            }
            for node in self.partitioner.iter_nodes() {
                let indices: Vec<usize> = node
                    .iter()
                    .map(|cell| cell.column() * degree_bound + cell.row())
                    .collect();
                let mut permuted: Vec<F> = indices.iter().map(|&i| sigma[i]).collect();
                permuted.rotate_left(1);
                for i in 0..indices.len() {
                    sigma[indices[i]] = permuted[i];
                }
            }
            sigma
                .chunks_exact(degree_bound)
                .map(|chunk| chunk.to_vec())
                .collect()
        };

        let sigma = sigma_values
            .iter()
            .map(|chunk| Polynomial::encode2(chunk.to_vec()))
            .collect();

        Ok(Circuit {
            num_rows: self.num_rows,
            num_columns: self.num_columns,
            pcs_config: pcs::Options {
                degree_bound,
                security_bits: options.security_bits,
                blowup_log2: options.blowup_log2,
                fold_rate: options.fri_fold_rate,
                max_rounds: options.max_fri_rounds,
                grinding_bits: options.grinding_bits,
            },
            blinded: options.blind,
            selectors,
            gates,
            sigma,
            sigma_values,
            circuit_area_selector: make_circuit_area_selector(self.num_rows, degree_bound),
            public_cells: self.public_cells,
            _data: PhantomData,
        })
    }
}

impl<F: Field, G: Field256<BaseField = F>> CircuitView<F, G> for CircuitBuilder<F, G> {
    fn builder(&self) -> &CircuitBuilder<F, G> {
        self
    }

    fn builder_mut(&mut self) -> &mut CircuitBuilder<F, G> {
        self
    }

    fn width(&self) -> Option<usize> {
        None
    }

    fn height(&self) -> Option<usize> {
        None
    }

    fn row_offset(&self) -> usize {
        0
    }

    fn column_offset(&self) -> usize {
        0
    }
}

/// Implements [`CircuitView`] for a sub-section of the circuit.
#[derive(Debug)]
pub struct CircuitSectionBuilder<'a, F: Field, G: Field256<BaseField = F>> {
    /// Reference to the parent [`CircuitBuilder`].
    builder: &'a mut CircuitBuilder<F, G>,

    /// Row offset of the sub-section.
    row_offset: usize,

    /// Column offset of the sub-section.
    column_offset: usize,

    /// Width (number of columns) of the sub-section.
    width: usize,

    /// Height (number of rows) of the sub-section.
    height: usize,
}

impl<'a, F: Field, G: Field256<BaseField = F>> CircuitSectionBuilder<'a, F, G> {
    fn new(
        builder: &'a mut CircuitBuilder<F, G>,
        row_offset: usize,
        column_offset: usize,
        width: usize,
        height: usize,
    ) -> Self {
        Self {
            builder,
            row_offset,
            column_offset,
            width,
            height,
        }
    }
}

impl<'a, F: Field, G: Field256<BaseField = F>> CircuitView<F, G>
    for CircuitSectionBuilder<'a, F, G>
{
    fn builder(&self) -> &CircuitBuilder<F, G> {
        self.builder
    }

    fn builder_mut(&mut self) -> &mut CircuitBuilder<F, G> {
        self.builder
    }

    fn width(&self) -> Option<usize> {
        Some(self.width)
    }

    fn height(&self) -> Option<usize> {
        Some(self.height)
    }

    fn row_offset(&self) -> usize {
        self.row_offset
    }

    fn column_offset(&self) -> usize {
        self.column_offset
    }
}

/// A PLONK proof.
///
/// The API in the implementation mostly mirrors that of the underlying PCS proof.
#[derive(Debug, Clone)]
pub struct Proof<F: Field, G: Field256<BaseField = F>, H: Hasher<G>> {
    public_inputs: BTreeMap<Cell, F>,
    commitment: pcs::Commitment<G, H>,
    inner_proof: pcs::Proof<G, H>,
    _data: PhantomData<F>,
}

impl<F: Field, G: Field256<BaseField = F>, H: Hasher<G>> Proof<F, G, H> {
    /// Returns the options used to configure the PCS.
    pub fn pcs_config(&self) -> &pcs::Options {
        self.inner_proof.options()
    }

    /// Returns the number of committed polynomials.
    ///
    /// These include the circuit selectors, the sigma polynomials, the circuit area selector, the
    /// witness columns, the witness randomizers, the permutation argument accumulator, the LogUp
    /// helper columns, the chunks of the grand quotient, and the FRI randomizer.
    pub fn num_polys(&self) -> usize {
        self.commitment.num_polys()
    }

    /// Returns a reference to the map of the revealed witness locations and corresponding values.
    pub fn public_inputs(&self) -> &BTreeMap<Cell, F> {
        &self.public_inputs
    }

    /// Serializes this proof to a [`Proof`](`proto::Proof`) protobuf.
    pub fn to_proto(&self) -> proto::Proof {
        proto::Proof {
            public_inputs: self
                .public_inputs
                .iter()
                .map(|(cell, &value)| proto::proof::PublicInput {
                    row: cell.row() as u64,
                    column: cell.column() as u64,
                    value: utils::store_scalar(value),
                })
                .collect(),
            commitment: Some(self.commitment.to_proto()),
            inner_proof: Some(self.inner_proof.to_proto()),
        }
    }

    /// Deserializes a proof from a [`Proof`](`proto::Proof`) protobuf.
    ///
    /// NOTE: this method does not validate the proof, it only deserializes it. The caller must
    /// invoke [`Circuit::verify`] or [`CompressedCircuit::verify`] separately.
    pub fn from_proto(proto: &proto::Proof) -> Result<Self> {
        let public_inputs: BTreeMap<Cell, F> = proto
            .public_inputs
            .iter()
            .map(|input| {
                Ok((
                    Cell::new(input.row as usize, input.column as usize),
                    utils::load_scalar(input.value.as_slice())?,
                ))
            })
            .collect::<Result<_>>()?;
        let commitment = match &proto.commitment {
            Some(commitment) => pcs::Commitment::from_proto(commitment),
            None => Err(anyhow!("missing PCS commitment")),
        }?;
        let inner_proof = match &proto.inner_proof {
            Some(inner_proof) => pcs::Proof::from_proto(inner_proof),
            None => Err(anyhow!("missing PCS proof")),
        }?;
        Ok(Self {
            public_inputs,
            commitment,
            inner_proof,
            _data: PhantomData,
        })
    }
}

/// A PLONK circuit.
#[derive(Debug, Clone)]
pub struct Circuit<F: Field, G: Field256<BaseField = F>> {
    /// The raw number of rows of the circuit.
    ///
    /// Unlike [`Self::degree_bound`], this count doesn't include the blinding rows and is not
    /// padded to the next power of 2.
    num_rows: usize,

    /// Number of witness columns.
    num_columns: usize,

    /// Whether or not the circuit is blinded (see [`Options::blind`]).
    blinded: bool,

    /// Configuration parameters for the underlying PCS.
    pcs_config: pcs::Options,

    /// Gate selectors.
    ///
    /// This is a pool of Lagrange bases that the grand gate constraint uses to selectively activate
    /// gates at the rows where they're used.
    ///
    /// The size of this pool is not (necessarily) the number of rows: [`CircuitBuilder`]'s
    /// algorithm tries to reuse selectors as much as possible.
    selectors: Vec<Polynomial<F>>,

    /// Gates used in the circuit: the first component of each pair is the gate constraint and the
    /// second component is the set of instances of that gate across the circuit.
    gates: BTreeMap<Constraint<F>, BTreeSet<GateInstance>>,

    /// Sigma polynomials of the permutation argument, one for every witness column.
    sigma: Vec<Polynomial<F>>,

    /// The [sigma polynomials](`Self::sigma`) expressed on the value domain.
    ///
    /// The layout is analogous to [`Self::sigma`] itself: the values are indexed column-first.
    sigma_values: Vec<Vec<F>>,

    /// Selector that switches the permutation argument off in the padding area.
    ///
    /// See [`make_circuit_area_selector`]. It's committed in the circuit batch, right after the
    /// sigma polynomials, so the verifier reads it from the openings rather than evaluating it.
    circuit_area_selector: Polynomial<F>,

    /// List of witness cells that are revealed in the proofs.
    public_cells: BTreeSet<Cell>,

    _data: PhantomData<G>,
}

impl<F: Field, G: Field256<BaseField = F>> Circuit<F, G> {
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    pub fn is_blinded(&self) -> bool {
        self.blinded
    }

    pub fn pcs_config(&self) -> &pcs::Options {
        &self.pcs_config
    }

    pub fn num_gates(&self) -> usize {
        self.gates
            .iter()
            .map(|(_, instances)| instances.len())
            .sum()
    }

    pub fn public_cells(&self) -> &BTreeSet<Cell> {
        &self.public_cells
    }

    pub fn get_max_gate_degree(&self) -> usize {
        self.gates
            .iter()
            .map(|(constraint, _)| constraint.get_degree())
            .max()
            .unwrap_or(0)
    }

    /// Makes an empty [`Witness`] objects suitable for use with this circuit.
    pub fn make_witness(&self) -> Witness<F> {
        let rotations = self.gates.iter().flat_map(|(constraint, _)| {
            constraint
                .get_free_variables()
                .iter()
                .map(Variable::rotation)
                .collect::<BTreeSet<isize>>()
        });
        let degree_bound = padded_circuit_size::<F, G>(
            self.num_rows,
            self.pcs_config.security_bits,
            self.pcs_config.blowup_log2,
            self.pcs_config.grinding_bits,
            self.blinded,
            rotations,
        );
        Witness::new(self.num_rows, self.num_columns, degree_bound)
    }

    /// Performs various checks to verify that the provided `witness` is compatible with this
    /// circuit and doesn't violate any constraints.
    pub fn check_witness(&self, witness: &Witness<F>) -> Result<()> {
        let degree_bound = self.pcs_config.degree_bound;

        if witness.num_rows() != self.num_rows {
            return Err(anyhow!(
                "wrong number of rows: got {}, want {}",
                witness.num_rows(),
                self.num_rows
            ));
        }
        if witness.num_columns() != self.num_columns {
            return Err(anyhow!(
                "wrong number of columns: got {}, want {}",
                witness.num_columns(),
                self.num_columns
            ));
        }
        if witness.degree_bound() != degree_bound {
            return Err(anyhow!(
                "incorrect degree bound: got {}, want {degree_bound}",
                witness.degree_bound()
            ));
        }

        let active_row_set: Vec<BTreeSet<usize>> = self
            .selectors
            .iter()
            .map(|selector| {
                selector
                    .clone()
                    .decode2()
                    .into_iter()
                    .enumerate()
                    .filter(|(_, value)| *value != F::ZERO)
                    .map(|(index, _)| index)
                    .collect()
            })
            .collect();
        for (constraint, gate_instances) in &self.gates {
            for gate_instance in gate_instances {
                let variables = constraint.get_free_variables();
                for &row in &active_row_set[gate_instance.selector_index] {
                    let root_cell = Cell::new(row, gate_instance.column_index);
                    let substitution: BTreeMap<Variable, F> = variables
                        .iter()
                        .map(|variable| {
                            (
                                variable.clone(),
                                witness.get_at(variable.map_to_cell(root_cell)),
                            )
                        })
                        .collect();
                    if constraint.evaluate(&substitution) != F::ZERO {
                        return Err(anyhow!(
                            "gate constraint `{}` violated at row {}, column {}",
                            constraint,
                            row,
                            gate_instance.column_index
                        ));
                    }
                }
            }
        }

        let cell_by_identity_value: BTreeMap<F, Cell> = {
            let mut cell_by_identity_value = BTreeMap::default();
            let omega = Polynomial::<G>::domain_element2(1, degree_bound);
            let mut generator_power = F::ONE;
            for column in 0..self.num_columns {
                let mut omega_power = F::ONE;
                for row in 0..degree_bound {
                    cell_by_identity_value
                        .insert(generator_power * omega_power, Cell::new(row, column));
                    omega_power *= omega;
                }
                generator_power *= F::MULTIPLICATIVE_GENERATOR;
            }
            cell_by_identity_value
        };
        for column in 0..self.num_columns {
            for row in 0..self.num_rows {
                let source_cell = Cell::new(row, column);
                let target_cell = *cell_by_identity_value
                    .get(&self.sigma_values[column][row])
                    .unwrap();
                let source_value = witness.get_at(source_cell);
                let target_value = witness.get_at(target_cell);
                if source_value != target_value {
                    return Err(anyhow!(
                        "wire constraint violated: cell({}, {}) = {}, cell({}, {}) = {}",
                        source_cell.row(),
                        source_cell.column(),
                        source_value,
                        target_cell.row(),
                        target_cell.column(),
                        target_value,
                    ));
                }
            }
        }

        Ok(())
    }

    /// Converts a polynomial over F to one over G.
    fn embed_polynomial(polynomial: &Polynomial<F>) -> Polynomial<G> {
        Polynomial::with_coefficients(
            polynomial
                .coefficients()
                .iter()
                .copied()
                .map(G::from)
                .collect(),
        )
    }

    /// Converts a polynomial over F to one over G, scaling it by a factor in the process.
    ///
    /// This function is equivalent to `embed_polynomial(...) * scale`, but it's faster when G is an
    /// tower extension of F because multiplications between F and G are faster than those between G
    /// and G.
    fn embed_and_scale_polynomial(polynomial: &Polynomial<F>, scale: G) -> Polynomial<G> {
        Polynomial::with_coefficients(
            polynomial
                .coefficients()
                .iter()
                .copied()
                .map(|coefficient| scale * coefficient)
                .collect(),
        )
    }

    /// Builds a PCS committer with the circuit batch: the gate selectors, the sigma polynomials,
    /// and the circuit area selector, all over the base field `F`.
    fn make_committer<H: Hasher<G>>(&self) -> pcs::Committer<G, H> {
        let mut committer = pcs::Committer::<G, H>::new(self.pcs_config);
        committer.add_base_batch(
            self.selectors
                .iter()
                .chain(self.sigma.iter())
                .chain(std::iter::once(&self.circuit_area_selector))
                .cloned()
                .collect(),
        );
        committer
    }

    fn get_variable_substitution(
        &self,
        omega: F,
        omega_inv: F,
        columns: &[Polynomial<F>],
    ) -> BTreeMap<Variable, Polynomial<F>> {
        let mut substitution: BTreeMap<Variable, Polynomial<F>> = BTreeMap::default();
        for (constraint, instances) in &self.gates {
            for instance in instances {
                for variable in constraint
                    .get_free_variables()
                    .into_iter()
                    .map(|variable| variable.remap(instance.column_index as isize))
                {
                    if !substitution.contains_key(&variable) {
                        let column = variable.rotate_column(
                            columns[variable.column_index()].clone(),
                            omega,
                            omega_inv,
                        );
                        substitution.insert(variable, column);
                    }
                }
            }
        }
        substitution
    }

    /// `build_permutation_argument` builds the polynomials used in the LogUp permutation argument.
    /// The components of the returned tuple are, respectively: the accumulator `Z`, the start
    /// anchor constraint, the end anchor constraint, the helper columns `h_i`, the helper
    /// authentication constraints, and the recurrence constraint.
    ///
    /// The accumulator and helper columns will be committed in the PCS, while the start/end anchor
    /// constraints, helper authentication constraints, and the recurrence constraints are only
    /// included in the grand constraint and the computation of the PLONK quotient. Since `Z` and
    /// the `h_i`s are committed and witness-derived, if the circuit is blinded this function blinds
    /// them by filling the rows corresponding to the padding area with random G values so that they
    /// don't leak any information about the witness ([`padded_circuit_size`] ensures that the
    /// padding area has enough capacity to warrant effective blinding given the number of rotations
    /// used in the circuit and the required FRI queries). If the circuit is not blinded those rows
    /// are filled with zeros, so that proofs are deterministic.
    ///
    /// We use the Fiat-Shamir challenge `gamma` as the abstract variable for LogUp. The numerator
    /// factors of the standard PLONK permutation argument are:
    ///
    ///   N_i(X) = W_i(X) + beta * g^i * X + gamma
    ///
    /// where `W_i` is the i-th witness column and `g` is
    /// [`F::MULTIPLICATIVE_GENERATOR`](`Field::MULTIPLICATIVE_GENERATOR`). The denominator factors
    /// are:
    ///
    ///   D_i(X) = W_i(X) + beta * sigma_i(X) + gamma
    ///
    /// where `sigma_i` is the i-th [sigma column](Self::sigma) (note that the powers of the
    /// multiplicative generator are premultiplied in all sigma polynomials so that we can spare a
    /// few multiplications here).
    ///
    /// The full numerator and denominator products are, respectively:
    ///
    ///   N(X) = Prod(N_i(X))
    ///   D(X) = Prod(D_i(X))
    ///
    /// Log-deriving both in `gamma` we get the sums of inverses, respectively:
    ///
    ///   [log(N(X))]' = Sum(1 / (W_i(X) + beta * g^i * X + gamma))
    ///   [log(D(X))]' = Sum(1 / (W_i(X) + beta * sigma_i(X) + gamma))
    ///
    /// Rather than committing one helper column per fraction we pair the numerator and the
    /// denominator of every witness column, so that a single helper column per witness column
    /// interpolates the difference of the two rational values materialized at every row:
    ///
    ///   h_i(X) = FFT^-1(1 / N_i(X) - 1 / D_i(X))
    ///
    /// (`FFT^-1` is the inverse Fourier Transform, as implemented in [`Polynomial::encode2`].)
    ///
    /// Since `1 / N - 1 / D = (D - N) / (N * D)`, the helpers are authenticated by the following
    /// constraint:
    ///
    ///   h_i(X) * N_i(X) * D_i(X) + N_i(X) - D_i(X) = 0 mod H
    ///
    /// As mentioned above, the `h_i` columns are blinded in the padding area, so the random values
    /// stored in that area would break the above constraint. For this reason we gate it with the
    /// [circuit area selector](`make_circuit_area_selector`) `S(X)`:
    ///
    ///   S(X) * (h_i(X) * N_i(X) * D_i(X) + N_i(X) - D_i(X)) = 0 mod H
    ///
    /// The start anchor and recurrence constraints of the log-derived accumulator are:
    ///
    ///   Z(X) * L_0(X) = 0 mod H
    ///   Z(wX) - Z(X) - Sum(h_i(X)) = 0 mod H
    ///
    /// where `L_0` is the Lagrange basis that activates at `w^0` and vanishes everywhere else (see
    /// [`lagrange0`]).
    ///
    /// Since the expression of the recurrence constraint contains `h_i` it suffers from the same
    /// breakage in the padding area as the `h_i`s themselves, so this too needs to be gated by
    /// `S(X)`. Moreover, gating the recurrence constraint means it is no longer guaranteed to loop
    /// back to 0 at the end of the domain (it will be 0 only because `S(X)` vanishes at that
    /// location), so we need to add a new anchor constraint for the tail end. The original
    /// (non-randomized) helper columns `h_i` are always zero in the padding area because that area
    /// has no wires and all `N_i` / `D_i` pairs are identical, so Z never changes in that area;
    /// that means Z must loop back to 0 when the circuit area ends, on the first padding location.
    /// So we add the following end anchor constraint:
    ///
    ///   Z(X) * L_{num_rows}(X) = 0 mod H
    ///
    fn build_permutation_argument(
        &self,
        witness: &Witness<F>,
        columns: &[Polynomial<F>],
        beta: G,
        gamma: G,
    ) -> Result<(
        Polynomial<G>,      // accumulator
        Polynomial<G>,      // start anchor constraint
        Polynomial<G>,      // end anchor constraint
        Vec<Polynomial<G>>, // helper columns
        Vec<Polynomial<G>>, // helper constraints
        Polynomial<G>,      // recurrence constraint
    )> {
        let degree_bound = self.pcs_config.degree_bound;

        let omega = Polynomial::<G>::domain_element2(1, degree_bound);
        let omega_inv = omega.invert_vartime().unwrap();
        let circuit_area_selector = Self::embed_polynomial(&self.circuit_area_selector);

        // Helper columns are initially stored in these two flat arrays, expressed in the value
        // domain and laid out in column-major order. Later on they're converted to individual
        // column polynomials.
        let mut numerator_table = vec![G::ZERO; self.num_columns * degree_bound];
        let mut denominator_table = vec![G::ZERO; self.num_columns * degree_bound];

        let mut generator_power = F::ONE;
        for i in 0..self.num_columns {
            let offset = i * degree_bound;
            let mut omega_power = F::ONE;
            for j in 0..degree_bound {
                let witness_value: G = witness.get_at(Cell::new(j, i)).into();
                numerator_table[offset + j] =
                    witness_value + beta * (generator_power * omega_power) + gamma;
                denominator_table[offset + j] =
                    witness_value + beta * self.sigma_values[i][j] + gamma;
                omega_power *= omega;
            }
            generator_power *= F::MULTIPLICATIVE_GENERATOR;
        }

        G::invert_batch(numerator_table.as_mut_slice());
        G::invert_batch(denominator_table.as_mut_slice());

        let mut accumulator = vec![G::ZERO; degree_bound];
        for i in 0..self.num_rows {
            accumulator[(i + 1) % degree_bound] = accumulator[i]
                + (0..self.num_columns)
                    .map(|j| {
                        numerator_table[j * degree_bound + i]
                            - denominator_table[j * degree_bound + i]
                    })
                    .sum::<G>();
        }
        if self.blinded {
            for i in (self.num_rows + 1)..degree_bound {
                accumulator[i] = G::random_default();
            }
        }
        if accumulator[self.num_rows % degree_bound] != G::ZERO {
            return Err(anyhow!("permutation accumulator end anchor check failed"));
        }

        let accumulator = Polynomial::<G>::encode2(accumulator);

        let helpers: Vec<Polynomial<G>> = numerator_table
            .chunks(degree_bound)
            .zip(denominator_table.chunks(degree_bound))
            .map(|(numerators, denominators)| {
                let mut values: Vec<G> = numerators[0..self.num_rows]
                    .iter()
                    .zip(denominators[0..self.num_rows].iter())
                    .map(|(&numerator, &denominator)| numerator - denominator)
                    .collect();
                if self.blinded {
                    values.resize_with(degree_bound, G::random_default);
                } else {
                    values.resize(degree_bound, G::ZERO);
                }
                Polynomial::encode2(values)
            })
            .collect();

        let identity = Polynomial::with_coefficients(vec![F::ZERO, F::ONE]);
        let mut generator_power = F::ONE;
        let helper_constraints: Vec<Polynomial<G>> = helpers
            .iter()
            .zip(columns.iter().zip(self.sigma.iter()))
            .map(|(helper, (column, sigma))| {
                let witness_column = Self::embed_polynomial(column);
                let numerator = witness_column.clone()
                    + Polynomial::with_coefficients(vec![gamma, beta * generator_power]);
                let denominator =
                    witness_column + Self::embed_and_scale_polynomial(sigma, beta) + gamma;
                let constraint = Polynomial::multiply_batch([helper, &numerator, &denominator])
                    + Self::embed_and_scale_polynomial(
                        &(identity.clone() * generator_power - sigma.clone()),
                        beta,
                    );
                generator_power *= F::MULTIPLICATIVE_GENERATOR;
                constraint.multiply(circuit_area_selector.clone())
            })
            .collect();

        let start_anchor_constraint =
            accumulator.clone() * Polynomial::lagrange0_2(degree_bound).clone();
        let end_anchor_constraint = accumulator.clone()
            * Polynomial::<G>::lagrange0_2(degree_bound)
                .clone()
                .shift_domain_by(omega_inv.pow_small(self.num_rows).into());
        let recurrence_constraint = (accumulator.clone().shift_domain_by(omega.into())
            - accumulator.clone()
            - helpers.iter().sum::<Polynomial<G>>())
        .multiply(circuit_area_selector);

        Ok((
            accumulator,
            start_anchor_constraint,
            end_anchor_constraint,
            helpers,
            helper_constraints,
            recurrence_constraint,
        ))
    }

    /// Builds the constraint that binds every [public cell](Self::public_cells) to its claimed
    /// value.
    ///
    /// Writing `S_j` for the selector of the public cells of column `j`, `W_j` for the j-th witness
    /// column, and `v_{ij}` for the claimed value of cell `(i, j)`, the constraint of a single
    /// column is:
    ///
    ///   S_j(X) * W_j(X) - PI_j(X) = 0 mod H
    ///
    /// with `PI_j` interpolating `v_{ij}` at every public row of column `j` and zero elsewhere: at
    /// a public row that reduces to `W_j(w^i) = v_{ij}`, while at all other rows both terms vanish
    /// identically, leaving the rest of the row unconstrained (and therefore still private).
    ///
    /// All columns are batched in a single constraint with the powers of the
    /// [`psi`](`public_cell_challenge`) challenge:
    ///
    ///   Sum(psi^j * S_j(X) * W_j(X)) - PI(X) = 0 mod H
    ///
    /// where `PI` interpolates `Sum(psi^j * v_{ij})` at every row. Note that `S_j` is entirely
    /// determined by public information, so it's never committed: the verifier reconstructs
    /// `S_j(xi)` in closed form out of the Lagrange bases of the public rows.
    ///
    /// The resulting constraint has degree `2 * (N - 1)`, so it never exceeds the
    /// [quotient degree bound](`quotient_degree_bound`).
    fn build_public_cell_constraint(
        &self,
        columns: &[Polynomial<F>],
        public_inputs: &BTreeMap<Cell, F>,
        psi: G,
    ) -> Polynomial<G> {
        let mut cells_by_column: BTreeMap<usize, Vec<(usize, F)>> = BTreeMap::new();
        for (cell, &value) in public_inputs {
            cells_by_column
                .entry(cell.column())
                .or_default()
                .push((cell.row(), value));
        }
        let degree_bound = self.pcs_config.degree_bound;
        let mut public_inputs = vec![G::ZERO; degree_bound];
        let mut constraint = Polynomial::<G>::default();
        for (&column_index, cells) in &cells_by_column {
            let power = psi.pow_small(column_index);
            let mut selector = vec![F::ZERO; degree_bound];
            for &(row, value) in cells {
                selector[row] = F::ONE;
                public_inputs[row] += power * value;
            }
            constraint += Self::embed_and_scale_polynomial(
                &(Polynomial::encode2(selector) * columns[column_index].clone()),
                power,
            );
        }
        constraint - Polynomial::encode2(public_inputs)
    }

    /// Splits the quotient polynomial in chunks so that it can be batch-committed even though its
    /// degree is much higher than the bound configured in the underlying PCS.
    ///
    /// Writing `T` for the quotient and `M` for the quotient stride (ie. the degree bound of each
    /// chunk), the chunks satisfy:
    ///
    ///   T(X) = Sum(T_c(X) * X^(c * M))
    ///
    /// The verifier only ever needs `T(xi)`, but a proof reveals every committed polynomial at
    /// `s = num_rotations + num_queries` points; for the grand quotient that means `C * s` points
    /// are actually revealed, where `C` is the number of chunks. In other words, chunking increases
    /// the number of reveals by a factor of `C`. Since the quotient is a function of the witness,
    /// this is an information leak that must be compensated by the blinding system.
    ///
    /// Unlike every other committed polynomial the quotient can't simply be blinded in the padding
    /// area, because it has no "rows" of its own: the algebraic check pins it completely, and any
    /// mask committed alongside it would be revealed at the same points and cancel out. Therefore
    /// PLONK uses a telescoping mask technique: a pair of random polynomials are added to each
    /// chunk in such a way that the first one cancels with the second one of the previous chunk and
    /// the second one cancels with the first one of the next chunk.
    ///
    /// Let `C` be the number of chunks and `R_i` the i-th random mask, with `0 <= i <= C` and
    /// `R_0(X) = R_C(X) = 0`. Rather than committing the `T_c` chunks directly we commit:
    ///
    ///   T'_c(X) = T_c(X) + R_{c+1}(X) * X^M - R_c(X)
    ///
    /// The internal masks `R_1` through `R_{C-1}` telescope away in the recombination while the
    /// external masks `R_0` and `R_C` are zero, so `T(X) = Sum(T'_c(X) * X^(c * M))` still holds.
    ///
    /// The `R_i` masks must compensate `s = num_rotations + num_queries` reveals, so their degree
    /// bound must be `s + 1` (as computed by [`mask_degree_bound`]). That means the degree bound of
    /// each chunk will be `M = N - s - 1` rather than `N` itself, and that's always okay because
    /// [`padded_circuit_size`] rounds `num_rows + num_queries + T * (num_rotations + 1)` up to the
    /// next power of two, with `T >= 1` being the extension degree of `G` over `F`, so `N` is
    /// guaranteed to exceed `s + 1` by at least `num_rows`.
    fn split_quotient(&self, quotient: Polynomial<G>) -> Vec<Polynomial<G>> {
        let quotient_degree_bound = quotient_degree_bound(
            self.pcs_config.degree_bound,
            self.gates.iter().map(|(constraint, _)| constraint),
        );

        let mask_degree_bound = if self.blinded {
            mask_degree_bound(
                self.pcs_config.security_bits,
                self.pcs_config.blowup_log2,
                self.pcs_config.grinding_bits,
                self.gates.iter().map(|(constraint, _)| constraint),
            )
        } else {
            0
        };
        assert!(mask_degree_bound < self.pcs_config.degree_bound);
        let stride = self.pcs_config.degree_bound - mask_degree_bound;

        let mut coefficients = quotient.take();
        assert!(coefficients.len() <= quotient_degree_bound);
        coefficients.resize(quotient_degree_bound, G::ZERO);

        let num_chunks = quotient_degree_bound.div_ceil(stride);

        let mut chunks = vec![vec![G::ZERO; stride + mask_degree_bound]; num_chunks];
        for (c, chunk) in chunks.iter_mut().enumerate() {
            let start = c * stride;
            let end = std::cmp::min(start + stride, quotient_degree_bound);
            chunk[..(end - start)].copy_from_slice(&coefficients[start..end]);
        }

        for c in 1..num_chunks {
            for j in 0..mask_degree_bound {
                let mask = G::random_default();
                chunks[c - 1][stride + j] += mask;
                chunks[c][j] -= mask;
            }
        }

        chunks
            .into_iter()
            .map(Polynomial::with_coefficients)
            .collect()
    }

    /// Samples the uniformly random polynomial that makes the DEEP-FRI layer zero-knowledge.
    ///
    /// FRI is not zero-knowledge on its own: the cosets opened by the queries and the folded
    /// polynomial sent in the clear after the last round are linear functionals of the random
    /// linear combination of all committed polynomials, and they can be enough to determine that
    /// combination entirely -- one random linear combination of every witness column, helper, and
    /// quotient chunk. With zero rounds the folded polynomial is the whole DEEP quotient, which
    /// together with the opened values always determines it. Adding a polynomial with all its `N`
    /// coefficients uniformly random makes the combination itself uniform, so that FRI reveals
    /// nothing beyond the point openings and the query leaves, which the blinding rows of the
    /// witness take care of.
    ///
    /// The randomizer takes part in no constraint and the verifier ignores its openings; it only
    /// needs to be in the batch. Note that it has to be uniform over `G`, not `F`: it masks
    /// `G`-linear functionals of a polynomial over `G`.
    ///
    /// Since it serves no purpose other than zero-knowledge it's committed only when the circuit is
    /// [blinded](`Options::blind`), saving one low-degree extension, one Merkle column, one value
    /// in every query leaf and one evaluation at every opened point. It's the last polynomial of
    /// the quotient batch, so leaving it out shifts no other index.
    fn make_fri_randomizer(&self) -> Polynomial<G> {
        Polynomial::with_coefficients(
            (0..self.pcs_config.degree_bound)
                .map(|_| G::random_default())
                .collect(),
        )
    }

    /// Samples the [witness randomizers](`num_witness_randomizers`), uniformly random polynomials
    /// over `F` with all their `N` coefficients random, so that all their evaluations on the
    /// extended domain are random too.
    fn make_witness_randomizers(&self) -> Vec<Polynomial<F>> {
        (0..num_witness_randomizers::<G>(self.blinded, self.num_columns))
            .map(|_| {
                Polynomial::with_coefficients(
                    (0..self.pcs_config.degree_bound)
                        .map(|_| F::random_default())
                        .collect(),
                )
            })
            .collect()
    }

    /// Proves correctness of the given witness, or returns an error in case of a constraint
    /// violation.
    pub fn prove<H: Hasher<G>>(&self, mut witness: Witness<F>) -> Result<Proof<F, G, H>> {
        if self.blinded {
            witness.blind();
        }

        let degree_bound = self.pcs_config.degree_bound;

        if witness.num_rows() != self.num_rows {
            return Err(anyhow!(
                "incorrect witness size (got {} rows, want {})",
                witness.num_rows(),
                self.num_rows
            ));
        }
        if witness.degree_bound() != degree_bound {
            return Err(anyhow!(
                "incorrect witness degree bound (got {}, want {degree_bound})",
                witness.degree_bound()
            ));
        }
        if witness.num_columns() != self.num_columns {
            return Err(anyhow!(
                "incorrect witness size (got {} columns, want {})",
                witness.num_columns(),
                self.num_columns
            ));
        }

        let mut committer = self.make_committer::<H>();

        let columns = witness.clone().encode();
        committer.add_base_batch(
            columns
                .iter()
                .cloned()
                .chain(self.make_witness_randomizers())
                .collect(),
        );

        let omega = Polynomial::<G>::domain_element2(1, degree_bound);
        let omega_inv = omega.invert_vartime().unwrap();

        let gate_constraint = {
            let substitution = self.get_variable_substitution(omega, omega_inv, columns.as_slice());
            let delta = H::challenge(*DST_DELTA, &[committer.transcript_hash()]);
            let mut gate_constraint = Polynomial::<G>::default();
            let mut power = G::ONE;
            for (constraint, instances) in &self.gates {
                for instance in instances {
                    let constraint = if instance.column_index != 0 {
                        constraint
                            .clone()
                            .remap_variables(instance.column_index as isize)
                    } else {
                        constraint.clone()
                    };
                    gate_constraint += Self::embed_and_scale_polynomial(
                        &(self.selectors[instance.selector_index].clone()
                            * constraint.compose2(&substitution)),
                        power,
                    );
                    power *= delta;
                }
            }
            gate_constraint
        };

        let (
            permutation_accumulator,
            permutation_start_anchor_constraint,
            permutation_end_anchor_constraint,
            permutation_helpers,
            permutation_helper_constraints,
            permutation_recurrence_constraint,
        ) = {
            let beta = H::challenge(*DST_BETA, &[committer.transcript_hash()]);
            let gamma = H::challenge(*DST_GAMMA, &[committer.transcript_hash()]);
            self.build_permutation_argument(&witness, columns.as_slice(), beta, gamma)?
        };
        committer.add_batch(
            std::iter::once(permutation_accumulator)
                .chain(permutation_helpers.into_iter())
                .collect(),
        );

        let public_inputs: BTreeMap<Cell, F> = self
            .public_cells
            .iter()
            .map(|&cell| (cell, witness.get_at(cell)))
            .collect();

        let public_cell_constraint = {
            let psi = public_cell_challenge::<F, G, H>(committer.transcript_hash(), &public_inputs);
            self.build_public_cell_constraint(columns.as_slice(), &public_inputs, psi)
        };

        let alpha = H::challenge(*DST_ALPHA, &[committer.transcript_hash()]);

        let quotient = {
            let mut power = alpha;
            let mut constraint = gate_constraint + permutation_start_anchor_constraint * power;
            power *= alpha;
            constraint += permutation_end_anchor_constraint * power;
            power *= alpha;
            for permutation_helper_constraint in permutation_helper_constraints {
                constraint += permutation_helper_constraint * power;
                power *= alpha;
            }
            constraint += permutation_recurrence_constraint * power;
            power *= alpha;
            constraint += public_cell_constraint * power;
            constraint.divide_by_zero(degree_bound)?
        };
        committer.add_batch(
            self.split_quotient(quotient)
                .into_iter()
                .chain(self.blinded.then(|| self.make_fri_randomizer()))
                .collect(),
        );

        let xi = H::challenge(*DST_XI, &[committer.transcript_hash()]);

        let (commitment, prover) = committer.commit();
        let inner_proof = prover.prove(BTreeSet::from_iter(
            get_rotation_set(self.gates.iter().map(|(constraint, _)| constraint))
                .into_iter()
                .map(|rotation| {
                    xi * if rotation < 0 { omega_inv } else { omega }
                        .pow_small(rotation.unsigned_abs())
                }),
        ));

        Ok(Proof {
            public_inputs,
            commitment,
            inner_proof,
            _data: PhantomData,
        })
    }

    pub fn to_compressed<H: Hasher<G>>(self) -> CompressedCircuit<F, G, H> {
        let committer = self.make_committer::<H>();
        CompressedCircuit {
            num_rows: self.num_rows,
            num_columns: self.num_columns,
            blinded: self.blinded,
            pcs_config: self.pcs_config,
            gates: self
                .gates
                .into_iter()
                .map(|(constraint, instances)| (constraint, instances.into_iter().collect()))
                .collect(),
            public_cells: self.public_cells,
            circuit_commitment: committer.root_hash(COMMIT_INDEX_CIRCUIT),
            _data: PhantomData,
        }
    }

    pub fn as_compressed<H: Hasher<G>>(&self) -> CompressedCircuit<F, G, H> {
        let committer = self.make_committer::<H>();
        CompressedCircuit {
            num_rows: self.num_rows,
            num_columns: self.num_columns,
            blinded: self.blinded,
            pcs_config: self.pcs_config,
            gates: self
                .gates
                .iter()
                .map(|(constraint, instances)| {
                    (constraint.clone(), instances.iter().cloned().collect())
                })
                .collect(),
            public_cells: self.public_cells.clone(),
            circuit_commitment: committer.root_hash(COMMIT_INDEX_CIRCUIT),
            _data: Default::default(),
        }
    }

    pub fn verify<H: Hasher<G>>(&self, proof: &Proof<F, G, H>) -> Result<BTreeMap<Cell, F>> {
        self.as_compressed::<H>().verify(proof)
    }

    /// Serializes this circuit to a [`Circuit`](`proto::Circuit`) protobuf.
    pub fn to_proto(&self) -> proto::Circuit {
        proto::Circuit {
            num_rows: self.num_rows as u64,
            num_columns: self.num_columns as u64,
            blinded: self.blinded,
            pcs_config: Some(pcs_config_to_proto(&self.pcs_config)),
            selectors: self
                .selectors
                .iter()
                .map(|selector| proto::RowSelector {
                    coefficients: selector
                        .coefficients()
                        .iter()
                        .copied()
                        .map(utils::store_scalar)
                        .collect(),
                })
                .collect(),
            gates: self
                .gates
                .iter()
                .map(|(constraint, instances)| proto::Gate {
                    constraint: Some(constraint.to_proto()),
                    instances: instances
                        .iter()
                        .map(|instance| proto::GateInstance {
                            column_index: instance.column_index as u64,
                            selector_index: instance.selector_index as u64,
                        })
                        .collect(),
                })
                .collect(),
            sigma_table: self
                .sigma_values
                .iter()
                .flat_map(|values| values.iter().copied().map(utils::store_scalar))
                .collect(),
            public_cells: self
                .public_cells
                .iter()
                .map(|cell| proto::PublicCell {
                    row_index: cell.row() as u64,
                    column_index: cell.column() as u64,
                })
                .collect(),
        }
    }

    /// Deserializes a circuit from a [`Circuit`](`proto::Circuit`) protobuf.
    pub fn from_proto(proto: &proto::Circuit) -> Result<Self> {
        let num_rows = proto.num_rows as usize;
        if num_rows < 1 {
            return Err(anyhow!("circuits must have at least one row"));
        }

        let num_columns = proto.num_columns as usize;

        let pcs_config = match &proto.pcs_config {
            Some(proto) => pcs_config_from_proto::<G>(proto),
            None => Err(anyhow!("missing PCS settings")),
        }?;

        let degree_bound = pcs_config.degree_bound;
        if degree_bound < num_rows {
            return Err(anyhow!(
                "the degree bound {} is lower than the number of rows {}",
                degree_bound,
                num_rows
            ));
        }

        let selectors: Vec<Polynomial<F>> = proto
            .selectors
            .iter()
            .map(|selector| {
                if selector.coefficients.len() > degree_bound {
                    return Err(anyhow!(
                        "invalid selector with {} coefficients (the degree bound is {})",
                        selector.coefficients.len(),
                        degree_bound
                    ));
                }
                Ok(Polynomial::with_coefficients(
                    selector
                        .coefficients
                        .iter()
                        .map(|coefficient| utils::load_scalar(coefficient.as_slice()))
                        .collect::<Result<_>>()?,
                ))
            })
            .collect::<Result<_>>()?;

        let gates: BTreeMap<Constraint<F>, BTreeSet<GateInstance>> = proto
            .gates
            .iter()
            .map(|gate| {
                let constraint = match &gate.constraint {
                    Some(constraint) => Constraint::from_proto(constraint),
                    None => Err(anyhow!("missing gate constraint")),
                }?;
                let max_column_index = constraint.get_max_column_index();
                Ok((
                    constraint,
                    gate.instances
                        .iter()
                        .map(|instance| {
                            let column_index = instance.column_index as usize;
                            if column_index + max_column_index >= num_columns {
                                return Err(anyhow!(
                                    "invalid gate placement: column {} overflows",
                                    column_index + max_column_index
                                ));
                            }
                            let selector_index = instance.selector_index as usize;
                            if selector_index >= selectors.len() {
                                return Err(anyhow!(
                                    "invalid selector index {} in gate instance",
                                    selector_index
                                ));
                            }
                            Ok(GateInstance {
                                column_index,
                                selector_index,
                            })
                        })
                        .collect::<Result<_>>()?,
                ))
            })
            .collect::<Result<_>>()?;

        if proto.sigma_table.len() != num_columns * degree_bound {
            return Err(anyhow!(
                "invalid sigma table size {} (want {} columns of {} values)",
                proto.sigma_table.len(),
                num_columns,
                degree_bound
            ));
        }
        let sigma_values: Vec<Vec<F>> = proto
            .sigma_table
            .chunks_exact(degree_bound)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|value| utils::load_scalar(value.as_slice()))
                    .collect::<Result<_>>()
            })
            .collect::<Result<_>>()?;
        let sigma: Vec<Polynomial<F>> = sigma_values
            .iter()
            .map(|column| Polynomial::encode2(column.clone()))
            .collect();

        let public_cells: BTreeSet<Cell> = proto
            .public_cells
            .iter()
            .map(|cell| {
                let row_index = cell.row_index as usize;
                let column_index = cell.column_index as usize;
                if row_index < num_rows && column_index < num_columns {
                    Ok(Cell::new(row_index, column_index))
                } else {
                    Err(anyhow!(
                        "invalid public cell ({}, {})",
                        row_index,
                        column_index
                    ))
                }
            })
            .collect::<Result<_>>()?;

        Ok(Self {
            num_rows,
            num_columns,
            blinded: proto.blinded,
            pcs_config,
            selectors,
            gates,
            sigma,
            sigma_values,
            circuit_area_selector: make_circuit_area_selector(num_rows, degree_bound),
            public_cells,
            _data: PhantomData,
        })
    }
}

/// A PLONK circuit in committed form.
///
/// This struct is much smaller than the original circuit but still allows full verification of a
/// proof for the circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedCircuit<F: Field, G: Field256<BaseField = F>, H: Hasher<G>> {
    /// The raw number of rows of the circuit.
    ///
    /// Unlike [`Self::degree_bound`], this count doesn't include the blinding rows and is not
    /// padded to the next power of 2.
    num_rows: usize,

    /// Number of witness columns.
    num_columns: usize,

    /// Whether or not the circuit is blinded (see [`Options::blind`]).
    blinded: bool,

    /// Configuration parameters for the underlying PCS.
    pcs_config: pcs::Options,

    /// Gates used in the circuit: the first component of each pair is the gate constraint and the
    /// second component is the set of instances of that gate across the circuit.
    gates: Vec<(Constraint<F>, Vec<GateInstance>)>,

    /// List of witness cells that are revealed in the proofs.
    public_cells: BTreeSet<Cell>,

    /// Merkle root of the gate selectors, the sigma polynomials, and the circuit area selector.
    circuit_commitment: H256,

    _data: PhantomData<(G, H)>,
}

impl<F: Field, G: Field256<BaseField = F>, H: Hasher<G>> CompressedCircuit<F, G, H> {
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    pub fn is_blinded(&self) -> bool {
        self.blinded
    }

    pub fn pcs_config(&self) -> &pcs::Options {
        &self.pcs_config
    }

    pub fn public_cells(&self) -> &BTreeSet<Cell> {
        &self.public_cells
    }

    pub fn get_max_gate_degree(&self) -> usize {
        self.gates
            .iter()
            .map(|(constraint, _)| constraint.get_degree())
            .max()
            .unwrap_or(0)
    }

    pub fn commitment(&self) -> H256 {
        self.circuit_commitment
    }

    /// Calculates the stride of this circuit's quotient split.
    ///
    /// See [`Circuit::split_quotient`] for details.
    fn get_quotient_stride(&self) -> Result<usize> {
        let mask_degree_bound = if self.blinded {
            mask_degree_bound(
                self.pcs_config.security_bits,
                self.pcs_config.blowup_log2,
                self.pcs_config.grinding_bits,
                self.gates.iter().map(|(constraint, _)| constraint),
            )
        } else {
            0
        };
        if mask_degree_bound >= self.pcs_config.degree_bound {
            return Err(anyhow!(
                "degree bound {} too small for blinding (the quotient masks need {})",
                self.pcs_config.degree_bound,
                mask_degree_bound
            ));
        }
        Ok(self.pcs_config.degree_bound - mask_degree_bound)
    }

    /// Calculates the number of chunks the quotient was split into.
    ///
    /// See [`quotient_degree_bound`] for details.
    fn get_num_quotient_chunks(&self) -> Result<usize> {
        Ok(quotient_degree_bound(
            self.pcs_config.degree_bound,
            self.gates.iter().map(|(constraint, _)| constraint),
        )
        .div_ceil(self.get_quotient_stride()?))
    }

    /// Verifies a [`Proof`], returning the map of proven public inputs if successful or an error
    /// otherwise.
    ///
    /// The returned map has exactly one entry for every cell in [`Self::public_cells`]. Any extra
    /// opening the proof may carry is ignored.
    pub fn verify(&self, proof: &Proof<F, G, H>) -> Result<BTreeMap<Cell, F>> {
        let commitment = &proof.commitment;
        let inner_proof = &proof.inner_proof;

        let batches = commitment.batches();
        if batches.len() != NUM_COMMIT_INDICES {
            return Err(anyhow!(
                "wrong number of Merkle roots (got {}, want {})",
                batches.len(),
                NUM_COMMIT_INDICES
            ));
        }
        if batches[COMMIT_INDEX_CIRCUIT].root() != self.circuit_commitment {
            return Err(anyhow!(
                "wrong circuit commitment (got {}, want {})",
                batches[COMMIT_INDEX_CIRCUIT].root(),
                self.circuit_commitment
            ));
        }

        if !inner_proof.options().satisfies(&self.pcs_config) {
            return Err(anyhow!(
                "PCS settings mismatch (got {:?}, want {:?})",
                inner_proof.options(),
                self.pcs_config
            ));
        }
        let degree_bound = self.pcs_config.degree_bound;

        let num_gate_selectors = self
            .gates
            .iter()
            .flat_map(|(_, gate_instances)| {
                gate_instances
                    .iter()
                    .map(|instance| instance.selector_index)
            })
            .collect::<BTreeSet<usize>>()
            .len();
        let num_sigma_polynomials = self.num_columns;
        let num_circuit_area_selectors = 1;
        let num_witness_columns = self.num_columns;
        let num_witness_randomizers = num_witness_randomizers::<G>(self.blinded, self.num_columns);
        let num_permutation_accumulators = 1;
        let num_permutation_helpers = self.num_columns;
        let num_quotient_chunks = self.get_num_quotient_chunks()?;
        let num_fri_randomizers = if self.blinded { 1 } else { 0 };

        if !batches[COMMIT_INDEX_CIRCUIT].base_field() {
            return Err(anyhow!(
                "the circuit polynomials must be over the base field"
            ));
        }
        if batches[COMMIT_INDEX_CIRCUIT].num_polys()
            != num_gate_selectors + num_sigma_polynomials + num_circuit_area_selectors
        {
            return Err(anyhow!(
                "invalid number of circuit polynomials (got {}, want {})",
                batches[COMMIT_INDEX_CIRCUIT].num_polys(),
                num_gate_selectors + num_sigma_polynomials + num_circuit_area_selectors,
            ));
        }
        if !batches[COMMIT_INDEX_WITNESS].base_field() {
            return Err(anyhow!(
                "the witness polynomials must be over the base field"
            ));
        }
        if batches[COMMIT_INDEX_WITNESS].num_polys()
            != num_witness_columns + num_witness_randomizers
        {
            return Err(anyhow!(
                "invalid number of witness polynomials (got {}, want {})",
                batches[COMMIT_INDEX_WITNESS].num_polys(),
                num_witness_columns + num_witness_randomizers,
            ));
        }
        if batches[COMMIT_INDEX_PERMUTATION_ARGUMENT].base_field() {
            return Err(anyhow!(
                "the permutation argument polynomials must not be over the base field"
            ));
        }
        if batches[COMMIT_INDEX_PERMUTATION_ARGUMENT].num_polys()
            != num_permutation_accumulators + num_permutation_helpers
        {
            return Err(anyhow!(
                "invalid number of permutation argument polynomials (got {}, want {})",
                batches[COMMIT_INDEX_PERMUTATION_ARGUMENT].num_polys(),
                num_permutation_accumulators + num_permutation_helpers,
            ));
        }
        if batches[COMMIT_INDEX_QUOTIENT].base_field() {
            return Err(anyhow!(
                "the polynomials in the quotient batch must not be over the base field"
            ));
        }
        if batches[COMMIT_INDEX_QUOTIENT].num_polys() != num_quotient_chunks + num_fri_randomizers {
            return Err(anyhow!(
                "invalid number of polynomials in the quotient batch (got {}, want {})",
                batches[COMMIT_INDEX_QUOTIENT].num_polys(),
                num_quotient_chunks + num_fri_randomizers,
            ));
        }

        for (_, gate_instances) in &self.gates {
            for instance in gate_instances {
                if instance.selector_index >= num_gate_selectors {
                    return Err(anyhow!(
                        "invalid selector index {} (the circuit has {} selectors)",
                        instance.selector_index,
                        num_gate_selectors
                    ));
                }
            }
        }

        let omega = Polynomial::<G>::domain_element2(1, degree_bound);
        let omega_inv = omega.invert_vartime().unwrap();

        let xi = H::challenge(
            *DST_XI,
            &[commitment.transcript_hash(COMMIT_INDEX_QUOTIENT + 1)],
        );

        let points = inner_proof.points();
        for rotation in get_rotation_set(self.gates.iter().map(|(constraint, _)| constraint)) {
            let challenge = xi
                * if rotation < 0 { omega_inv } else { omega }
                    .pow_small_vartime(rotation.unsigned_abs());
            if !points.contains_key(&challenge) {
                return Err(anyhow!(
                    "the proof doesn't have an opening for the required rotation {}",
                    rotation
                ));
            }
        }

        let public_inputs: BTreeMap<Cell, F> = self
            .public_cells
            .iter()
            .map(|&cell| match proof.public_inputs().get(&cell) {
                Some(&value) => Ok((cell, value)),
                None => Err(anyhow!(
                    "the proof doesn't have an opening for public cell ({}, {})",
                    cell.row(),
                    cell.column()
                )),
            })
            .collect::<Result<_>>()?;

        inner_proof.verify(&commitment)?;

        let sigma: Vec<G> = {
            let offset = num_gate_selectors;
            (0..self.num_columns)
                .map(|i| points[&xi][offset + i])
                .collect()
        };

        let gate_constraint: G = {
            let selectors: Vec<G> = (0..num_gate_selectors).map(|i| points[&xi][i]).collect();
            let delta = H::challenge(
                *DST_DELTA,
                &[commitment.transcript_hash(COMMIT_INDEX_WITNESS + 1)],
            );
            let mut result = G::ZERO;
            let mut power = G::ONE;
            for (constraint, gate_instances) in &self.gates {
                for instance in gate_instances {
                    let constraint = constraint
                        .clone()
                        .remap_variables(instance.column_index as isize);
                    let substitution: BTreeMap<Variable, G> = constraint
                        .get_free_variables()
                        .into_iter()
                        .map(|variable| {
                            let offset = num_gate_selectors
                                + num_sigma_polynomials
                                + num_circuit_area_selectors;
                            let rotation = variable.rotation();
                            let challenge = xi
                                * if rotation < 0 { omega_inv } else { omega }
                                    .pow_small_vartime(rotation.unsigned_abs());
                            (
                                variable,
                                points[&challenge][offset + variable.column_index()],
                            )
                        })
                        .collect();
                    result += selectors[instance.selector_index]
                        * constraint.evaluate(&substitution)
                        * power;
                    power *= delta;
                }
            }
            result
        };

        let (permutation_accumulator, shifted_permutation_accumulator) = {
            let index = num_gate_selectors
                + num_sigma_polynomials
                + num_circuit_area_selectors
                + num_witness_columns
                + num_witness_randomizers;
            (points[&xi][index], points[&(xi * omega)][index])
        };
        let permutation_helpers: Vec<G> = {
            let offset = num_gate_selectors
                + num_sigma_polynomials
                + num_circuit_area_selectors
                + num_witness_columns
                + num_witness_randomizers
                + num_permutation_accumulators;
            (0..self.num_columns)
                .map(|i| points[&xi][offset + i])
                .collect()
        };

        let beta = H::challenge(
            *DST_BETA,
            &[commitment.transcript_hash(COMMIT_INDEX_WITNESS + 1)],
        );
        let gamma = H::challenge(
            *DST_GAMMA,
            &[commitment.transcript_hash(COMMIT_INDEX_WITNESS + 1)],
        );

        let witness_columns: Vec<G> = {
            let offset = num_gate_selectors + num_sigma_polynomials + num_circuit_area_selectors;
            (0..self.num_columns)
                .map(|i| points[&xi][offset + i])
                .collect()
        };

        let circuit_area_selector = points[&xi][num_gate_selectors + num_sigma_polynomials];

        let (permutation_helper_constraints, permutation_recurrence_constraint) = {
            let mut generator_power = F::ONE;
            let helper_constraints: Vec<G> = (0..self.num_columns)
                .map(|i| {
                    let constraint = (permutation_helpers[i]
                        * (witness_columns[i] + beta * generator_power * xi + gamma)
                        * (witness_columns[i] + beta * sigma[i] + gamma)
                        + beta * (xi * generator_power - sigma[i]))
                        * circuit_area_selector;
                    generator_power *= F::MULTIPLICATIVE_GENERATOR;
                    constraint
                })
                .collect();
            let recurrence_constraint = (shifted_permutation_accumulator
                - permutation_accumulator
                - permutation_helpers.into_iter().sum::<G>())
                * circuit_area_selector;
            (helper_constraints, recurrence_constraint)
        };

        let quotient: G = {
            let offset = num_gate_selectors
                + num_sigma_polynomials
                + num_circuit_area_selectors
                + num_witness_columns
                + num_witness_randomizers
                + num_permutation_accumulators
                + num_permutation_helpers;
            let stride = self.get_quotient_stride()?;
            (0..num_quotient_chunks)
                .map(|i| points[&xi][offset + i] * xi.pow_small(i * stride))
                .sum()
        };
        let zero = xi.pow_small(degree_bound) - G::ONE;

        let alpha = H::challenge(
            *DST_ALPHA,
            &[commitment.transcript_hash(COMMIT_INDEX_PERMUTATION_ARGUMENT + 1)],
        );

        let permutation_start_anchor_constraint =
            permutation_accumulator * lagrange0(xi, degree_bound);
        let permutation_end_anchor_constraint = permutation_accumulator
            * lagrange0(
                xi * omega_inv.pow_small_vartime(self.num_rows),
                degree_bound,
            );

        let public_cell_constraint: G = {
            let psi = public_cell_challenge::<F, G, H>(
                commitment.transcript_hash(COMMIT_INDEX_PERMUTATION_ARGUMENT + 1),
                &public_inputs,
            );
            let rows: Vec<usize> = public_inputs
                .keys()
                .map(Cell::row)
                .collect::<BTreeSet<usize>>()
                .into_iter()
                .collect();
            let mut lagrange: Vec<G> = rows
                .iter()
                .map(|&row| G::from(degree_bound as u64) * (xi - omega.pow_small_vartime(row)))
                .collect();
            G::invert_batch(lagrange.as_mut_slice());
            let lagrange: BTreeMap<usize, G> = rows
                .iter()
                .zip(lagrange)
                .map(|(&row, inverse)| (row, zero * omega.pow_small_vartime(row) * inverse))
                .collect();
            public_inputs
                .iter()
                .map(|(cell, &value)| {
                    psi.pow_small(cell.column())
                        * lagrange[&cell.row()]
                        * (witness_columns[cell.column()] - G::from(value))
                })
                .sum()
        };

        let full_constraint = {
            let mut power = alpha;
            let mut result = gate_constraint + power * permutation_start_anchor_constraint;
            power *= alpha;
            result += power * permutation_end_anchor_constraint;
            power *= alpha;
            for constraint in permutation_helper_constraints {
                result += power * constraint;
                power *= alpha;
            }
            result += power * permutation_recurrence_constraint;
            power *= alpha;
            result += power * public_cell_constraint;
            result
        };
        if full_constraint != quotient * zero {
            return Err(anyhow!("constraint violation"));
        }

        Ok(public_inputs)
    }

    /// Serializes this compressed circuit to a [`CompressedCircuit`](`proto::CompressedCircuit`)
    /// protobuf.
    pub fn to_proto(&self) -> proto::CompressedCircuit {
        proto::CompressedCircuit {
            hash_backend: Some(prost_types::Any::from_msg(&H::ProtoTag::default()).unwrap()),
            num_rows: self.num_rows as u64,
            num_columns: self.num_columns as u64,
            blinded: self.blinded,
            pcs_config: Some(pcs_config_to_proto(&self.pcs_config)),
            gates: self
                .gates
                .iter()
                .map(|(constraint, instances)| proto::Gate {
                    constraint: Some(constraint.to_proto()),
                    instances: instances
                        .iter()
                        .map(|instance| proto::GateInstance {
                            column_index: instance.column_index as u64,
                            selector_index: instance.selector_index as u64,
                        })
                        .collect(),
                })
                .collect(),
            public_cells: self
                .public_cells
                .iter()
                .map(|cell| proto::PublicCell {
                    row_index: cell.row() as u64,
                    column_index: cell.column() as u64,
                })
                .collect(),
            circuit_commitment: utils::store_hash(self.circuit_commitment),
        }
    }

    /// Deserializes a compressed circuit from a
    /// [`CompressedCircuit`](`proto::CompressedCircuit`) protobuf.
    pub fn from_proto(proto: &proto::CompressedCircuit) -> Result<Self> {
        match &proto.hash_backend {
            Some(hash_backend) => {
                hash_backend.to_msg::<H::ProtoTag>().map_err(|_| {
                    anyhow!(
                        "incorrect hash backend (got {}, want {})",
                        hash_backend.type_url,
                        H::ProtoTag::type_url()
                    )
                })?;
                Ok(())
            }
            None => Err(anyhow!("hash backend not specified")),
        }?;

        let num_rows = proto.num_rows as usize;
        let num_columns = proto.num_columns as usize;

        let pcs_config = match &proto.pcs_config {
            Some(proto) => pcs_config_from_proto::<G>(proto),
            None => Err(anyhow!("missing PCS settings")),
        }?;

        let degree_bound = pcs_config.degree_bound;
        if degree_bound < num_rows {
            return Err(anyhow!(
                "the degree bound {} is lower than the number of rows {}",
                degree_bound,
                num_rows
            ));
        }

        let gates: BTreeMap<Constraint<F>, BTreeSet<GateInstance>> = proto
            .gates
            .iter()
            .map(|gate| {
                let constraint = match &gate.constraint {
                    Some(constraint) => Constraint::from_proto(constraint),
                    None => Err(anyhow!("missing gate constraint")),
                }?;
                let max_column_index = constraint.get_max_column_index();
                Ok((
                    constraint,
                    gate.instances
                        .iter()
                        .map(|instance| {
                            let column_index = instance.column_index as usize;
                            if column_index + max_column_index >= num_columns {
                                return Err(anyhow!(
                                    "invalid gate placement: column {} overflows",
                                    column_index + max_column_index
                                ));
                            }
                            let selector_index = instance.selector_index as usize;
                            Ok(GateInstance {
                                column_index,
                                selector_index,
                            })
                        })
                        .collect::<Result<_>>()?,
                ))
            })
            .collect::<Result<_>>()?;
        let gates: Vec<(Constraint<F>, Vec<GateInstance>)> = gates
            .into_iter()
            .map(|(constraint, instances)| (constraint, instances.into_iter().collect()))
            .collect();

        let public_cells: BTreeSet<Cell> = proto
            .public_cells
            .iter()
            .map(|cell| {
                let row_index = cell.row_index as usize;
                let column_index = cell.column_index as usize;
                if row_index < num_rows && column_index < num_columns {
                    Ok(Cell::new(row_index, column_index))
                } else {
                    Err(anyhow!(
                        "invalid public cell ({}, {})",
                        row_index,
                        column_index
                    ))
                }
            })
            .collect::<Result<_>>()?;

        Ok(Self {
            num_rows,
            num_columns,
            blinded: proto.blinded,
            pcs_config,
            gates,
            public_cells,
            circuit_commitment: utils::load_hash(proto.circuit_commitment.as_slice())?,
            _data: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::var;
    use crate::witness::WitnessView;
    use starkom_bluesky::{Scalar as BS, from_const};
    use starkom_goldilocks::{GL, GL4};
    use starkom_pcs::hash::{Keccak256Hash, Sha2Hash};

    fn parse_hash(s: &'static str) -> H256 {
        s.parse().unwrap()
    }

    #[inline]
    fn cell(row: usize, column: usize) -> Cell {
        Cell::new(row, column)
    }

    // This function tests the circuit from Vitalik's PLONK tutorial,
    // https://vitalik.eth.limo/general/2019/09/22/plonk.html#how-plonk-works.
    fn test_vitalik_circuit_impl<F: Field, G: Field256<BaseField = F>, H: Hasher<G>>(
        canonicalize_constraints: bool,
        blowup_log2: usize,
        blind: bool,
        expected_degree_bound: usize,
        expected_num_polys: usize,
        commitment: H256,
    ) -> Result<()> {
        let mut builder = CircuitBuilder::<F, G>::default();
        builder.add_gate(0, (var(0) ^ 2) - var(1));
        builder.connect(cell(0, 0).into(), cell(1, 0).into());
        builder.connect(cell(0, 1).into(), cell(1, 1).into());
        builder.add_gate(1, var(0) * var(1) + var(0) + 5 - var(2));
        builder.connect(cell(0, 0).into(), cell(2, 0).into());
        builder.connect(cell(1, 2).into(), cell(2, 1).into());
        builder.add_gate(2, Constraint::nop());
        builder.declare_public_cells([cell(2, 0), cell(2, 1)]);
        let circuit = builder.build(Options {
            canonicalize_constraints,
            blowup_log2,
            blind,
            ..Options::default()
        })?;
        assert_eq!(circuit.num_rows(), 3);
        assert_eq!(circuit.pcs_config().degree_bound, expected_degree_bound);
        assert_eq!(circuit.num_columns(), 3);
        assert_eq!(circuit.pcs_config().blowup_log2, blowup_log2);
        assert_eq!(circuit.is_blinded(), blind);
        assert_eq!(circuit.num_gates(), 3);
        assert_eq!(circuit.get_max_gate_degree(), 2);
        let mut witness = circuit.make_witness();
        assert_eq!(witness.num_rows(), 3);
        assert_eq!(witness.degree_bound(), expected_degree_bound);
        assert_eq!(witness.num_columns(), 3);
        witness.set(cell(0, 0), 3u8.into());
        witness.set(cell(0, 1), 9u8.into());
        witness.set(cell(1, 0), 3u8.into());
        witness.set(cell(1, 1), 9u8.into());
        witness.set(cell(1, 2), 35u8.into());
        witness.set(cell(2, 0), 3u8.into());
        witness.set(cell(2, 1), 35u8.into());
        assert!(circuit.check_witness(&witness).is_ok());
        let proof = circuit.prove::<H>(witness)?;
        assert_eq!(proof.pcs_config().degree_bound, expected_degree_bound);
        assert_eq!(proof.pcs_config().blowup_log2, blowup_log2);
        assert_eq!(
            proof.pcs_config().extended_domain_size(),
            expected_degree_bound << blowup_log2
        );
        assert_eq!(proof.num_polys(), expected_num_polys);
        let circuit = circuit.to_compressed();
        assert_eq!(circuit.commitment(), commitment);
        let public_inputs = circuit.verify(&proof)?;
        assert_eq!(public_inputs[&cell(2, 0)], 3u8.into());
        assert_eq!(public_inputs[&cell(2, 1)], 35u8.into());
        Ok(())
    }

    #[test]
    fn test_vitalik_circuit_bluesky_sha2_blowup_2() {
        let c = parse_hash("0xa50d56f230b7d6aa3fad7d4155aeb9f81c1466209d744d0fbda7b899eccb6a85");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(false, 1, false, 4, 17, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(true, 1, false, 4, 17, c).is_ok()
        );
        let c = parse_hash("0x3874914bb99178730b56609b7cf8649c63521642f174f1c43e8a8c77decda9e7");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(false, 1, true, 256, 22, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(true, 1, true, 256, 22, c).is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_goldilocks_sha2_blowup_2() {
        let c = parse_hash("0xecae9813182a1afce0664ecfd0263989aad15f0df9fa15e90e58dcf0b34c32da");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(false, 1, false, 4, 17, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(true, 1, false, 4, 17, c).is_ok()
        );
        let c = parse_hash("0x68ee1cfdfa3a715d14e171b2cebf79fc8963eeb233faaa865b8b4f57d886ad8a");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(false, 1, true, 256, 23, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(true, 1, true, 256, 23, c).is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_bluesky_keccak256_blowup_2() {
        let c = parse_hash("0x2fdb065db07faf4e0f3e870f3ca5fa071d4cf723f0ec4cf7e166548c1c7459a6");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(false, 1, false, 4, 17, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(true, 1, false, 4, 17, c)
                .is_ok()
        );
        let c = parse_hash("0x221855b73b09f5c9e1a2e532534e101390550ae09449b1ed29eca8b18aafb9e2");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(false, 1, true, 256, 22, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(true, 1, true, 256, 22, c)
                .is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_goldilocks_keccak256_blowup_2() {
        let c = parse_hash("0x1cfdf49a4ad9c04773c4128e36c829ee6d56f3346da61a007455db3eca87fae6");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(false, 1, false, 4, 17, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(true, 1, false, 4, 17, c)
                .is_ok()
        );
        let c = parse_hash("0x2791a00c7ec8c5b7d4eac400e53d126db196d6120e3b082bac6403452ef9f444");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(false, 1, true, 256, 23, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(true, 1, true, 256, 23, c)
                .is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_bluesky_sha2_blowup_4() {
        let c = parse_hash("0x2eddc3df755eeeff8491387aa63c6c550b3a812fb01b203ccd0d73a503cde9fc");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(false, 2, false, 4, 17, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(true, 2, false, 4, 17, c).is_ok()
        );
        let c = parse_hash("0xca7f024475d882ab88e97cef2e34449a36857f01e30c3ba66c521738d1a3ad74");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(false, 2, true, 128, 22, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(true, 2, true, 128, 22, c).is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_goldilocks_sha2_blowup_4() {
        let c = parse_hash("0x0cb0571df8b19c5167198e7b67641235ab4305c4decc42e1e533d6ec0ab30b01");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(false, 2, false, 4, 17, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(true, 2, false, 4, 17, c).is_ok()
        );
        let c = parse_hash("0xe939cc71a8bc5951e3eb50859cb1c2c1f77c2e494728e936ff06f301b2d464c5");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(false, 2, true, 128, 23, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(true, 2, true, 128, 23, c).is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_bluesky_keccak256_blowup_4() {
        let c = parse_hash("0x74e0e2cbaf2170be82b0bc2f1d407f1ce6e38d531c427a7bf31be193f5e5257a");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(false, 2, false, 4, 17, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(true, 2, false, 4, 17, c)
                .is_ok()
        );
        let c = parse_hash("0xed353f3eec35bd94fe35ed6b738028cd213663446ee0a0eabc7a640b6de4c02c");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(false, 2, true, 128, 22, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(true, 2, true, 128, 22, c)
                .is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_goldilocks_keccak256_blowup_4() {
        let c = parse_hash("0x808cb587d04173c9993e688ff4c9bbd481028c9e63cb7c2597bd231b89c5a977");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(false, 2, false, 4, 17, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(true, 2, false, 4, 17, c)
                .is_ok()
        );
        let c = parse_hash("0x58d79bf08258beffed986ed8876ca43c85b43b5faf439940b921ff3ae915aa38");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(false, 2, true, 128, 23, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(true, 2, true, 128, 23, c)
                .is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_bluesky_sha2_blowup_8() {
        let c = parse_hash("0x8645a6a5c02132e38660559175a2f44645670d6ce2e681224dda072c69621b0e");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(false, 3, false, 4, 17, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(true, 3, false, 4, 17, c).is_ok()
        );
        let c = parse_hash("0x490c6c69659224c08a8bd8a1a7631a995cd698cae763d5ca4ddb9688d062fd62");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(false, 3, true, 64, 26, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(true, 3, true, 64, 26, c).is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_goldilocks_sha2_blowup_8() {
        let c = parse_hash("0x6fcdfcb37af00d5698120b5c79a417063daed0173f1a95cf27b28afca3f03f02");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(false, 3, false, 4, 17, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(true, 3, false, 4, 17, c).is_ok()
        );
        let c = parse_hash("0x649e0a30be12a68a298346ac25c37d956dd39cf90734b1fc77dde4ff6a4f5cc9");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(false, 3, true, 64, 27, c).is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(true, 3, true, 64, 27, c).is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_bluesky_keccak256_blowup_8() {
        let c = parse_hash("0xf7df798f43941cbb9b65dea803848702fa8c6ab71946c35fee2269d9a040443a");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(false, 3, false, 4, 17, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(true, 3, false, 4, 17, c)
                .is_ok()
        );
        let c = parse_hash("0x6cff59a9db40aa47a7c131662154ccc54fb4e35eeaee1f7efb41856e480bfc45");
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(false, 3, true, 64, 26, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(true, 3, true, 64, 26, c)
                .is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_goldilocks_keccak256_blowup_8() {
        let c = parse_hash("0x6668ecb9015d90696479cc2a26336e0cb25014a44ad35df7f03f515b3d92536b");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(false, 3, false, 4, 17, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(true, 3, false, 4, 17, c)
                .is_ok()
        );
        let c = parse_hash("0xec6aec9cb67616eccfe67e85f7e41e444d2ffae98fb9bf81c42716966ceebe28");
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(false, 3, true, 64, 27, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_impl::<GL, GL4, Keccak256Hash<GL4>>(true, 3, true, 64, 27, c)
                .is_ok()
        );
    }

    /// A slight variation of Vitalik's circuit. This one proves knowledge of three numbers x, y,
    /// and z such that x^3 + xy + 5 = z. Valid combinations are (3, 4, 44) and (4, 3, 81).
    fn test_vitalik_circuit_variation<F: Field, G: Field256<BaseField = F>, H: Hasher<G>>(
        canonicalize_constraints: bool,
        blowup_log2: usize,
        blind: bool,
        expected_degree_bound: usize,
        expected_num_polys: usize,
        commitment: H256,
    ) -> Result<()> {
        let mut builder = CircuitBuilder::<F, G>::default();
        builder.add_gate(0, (var(0) ^ 2) - var(1));
        let x = cell(0, 0);
        let square = cell(0, 1);
        builder.add_gate(1, var(0) * var(1) - var(2));
        builder.connect(x.into(), cell(1, 0).into());
        let y = cell(1, 1);
        let mul = cell(1, 2);
        builder.add_gate(2, var(0) * var(1) + var(2) + 5 - var(3));
        builder.connect(x.into(), cell(2, 0).into());
        builder.connect(square.into(), cell(2, 1).into());
        builder.connect(mul.into(), cell(2, 2).into());
        let result = cell(2, 3);
        let x_out = cell(3, 0);
        builder.connect(x.into(), x_out.into());
        let y_out = cell(3, 1);
        builder.connect(y.into(), y_out.into());
        let result_out = cell(3, 2);
        builder.connect(result.into(), result_out.into());
        builder.declare_public_cells([x_out, y_out, result_out]);
        let circuit = builder.build(Options {
            canonicalize_constraints,
            blowup_log2,
            blind,
            ..Options::default()
        })?;
        assert_eq!(circuit.num_rows(), 4);
        assert_eq!(circuit.pcs_config().degree_bound, expected_degree_bound);
        assert_eq!(circuit.num_columns(), 4);
        assert_eq!(circuit.pcs_config().blowup_log2, blowup_log2);
        assert_eq!(circuit.is_blinded(), blind);
        assert_eq!(circuit.num_gates(), 3);
        assert_eq!(circuit.get_max_gate_degree(), 2);
        let mut witness = circuit.make_witness();
        assert_eq!(witness.num_rows(), 4);
        assert_eq!(witness.degree_bound(), expected_degree_bound);
        assert_eq!(witness.num_columns(), 4);
        witness.set(cell(0, 0), 3u8.into());
        witness.set(cell(0, 1), 9u8.into());
        witness.set(cell(1, 0), 3u8.into());
        witness.set(cell(1, 1), 4u8.into());
        witness.set(cell(1, 2), 12u8.into());
        witness.set(cell(2, 0), 3u8.into());
        witness.set(cell(2, 1), 9u8.into());
        witness.set(cell(2, 2), 12u8.into());
        witness.set(cell(2, 3), 44u8.into());
        witness.set(cell(3, 0), 3u8.into());
        witness.set(cell(3, 1), 4u8.into());
        witness.set(cell(3, 2), 44u8.into());
        assert!(circuit.check_witness(&witness).is_ok());
        let proof = circuit.prove::<H>(witness)?;
        assert_eq!(proof.pcs_config().degree_bound, expected_degree_bound);
        assert_eq!(proof.pcs_config().blowup_log2, blowup_log2);
        assert_eq!(
            proof.pcs_config().extended_domain_size(),
            expected_degree_bound << blowup_log2
        );
        assert_eq!(proof.num_polys(), expected_num_polys);
        let circuit = circuit.to_compressed();
        assert_eq!(circuit.commitment(), commitment);
        let public_inputs = circuit.verify(&proof)?;
        assert_eq!(public_inputs[&x_out], 3u8.into());
        assert_eq!(public_inputs[&y_out], 4u8.into());
        assert_eq!(public_inputs[&result_out], 44u8.into());
        Ok(())
    }

    #[test]
    fn test_vitalik_circuit_variation_bluesky_blowup_2() {
        let c = parse_hash("0x14663a22fd17d00655866bc603dc22665a44c462944b02e944cc5c92066d353b");
        assert!(
            test_vitalik_circuit_variation::<BS, BS, Sha2Hash<BS>>(false, 1, false, 4, 20, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_variation::<BS, BS, Sha2Hash<BS>>(true, 1, false, 4, 20, c)
                .is_ok()
        );
        let c = parse_hash("0x278b49ff044bad2b27d0675183d892bdd9dd8583e260e33f2c09ed6713f3247c");
        assert!(
            test_vitalik_circuit_variation::<BS, BS, Sha2Hash<BS>>(false, 1, true, 256, 25, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_variation::<BS, BS, Sha2Hash<BS>>(true, 1, true, 256, 25, c)
                .is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_variation_goldilocks_blowup_2() {
        let c = parse_hash("0x2424fd36f2070b3236830ea5e1e7a45fcf432f4b92f00b216566c6f604f6148d");
        assert!(
            test_vitalik_circuit_variation::<GL, GL4, Sha2Hash<GL4>>(false, 1, false, 4, 20, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_variation::<GL, GL4, Sha2Hash<GL4>>(true, 1, false, 4, 20, c)
                .is_ok()
        );
        let c = parse_hash("0xd2d12ff566c124df52858f9e93f81744f4912da6b0f8f97e084982fcfef52982");
        assert!(
            test_vitalik_circuit_variation::<GL, GL4, Sha2Hash<GL4>>(false, 1, true, 256, 25, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_variation::<GL, GL4, Sha2Hash<GL4>>(true, 1, true, 256, 25, c)
                .is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_variation_bluesky_blowup_4() {
        let c = parse_hash("0xb1ddc8f1c8e08467da59ba69a419315dff00e34574eed905045f97252194c1d6");
        assert!(
            test_vitalik_circuit_variation::<BS, BS, Sha2Hash<BS>>(false, 2, false, 4, 20, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_variation::<BS, BS, Sha2Hash<BS>>(true, 2, false, 4, 20, c)
                .is_ok()
        );
        let c = parse_hash("0xaa379c376de2dc63538a2dfe80d566d73062b222a2c9fbd1109d9f37b9bba56a");
        assert!(
            test_vitalik_circuit_variation::<BS, BS, Sha2Hash<BS>>(false, 2, true, 128, 25, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_variation::<BS, BS, Sha2Hash<BS>>(true, 2, true, 128, 25, c)
                .is_ok()
        );
    }

    #[test]
    fn test_vitalik_circuit_variation_goldilocks_blowup_4() {
        let c = parse_hash("0xc98a1aa48a73f5a483e70f3b28987d8881f1defa4e975dbbbbf6c9feb6b3c2e0");
        assert!(
            test_vitalik_circuit_variation::<GL, GL4, Sha2Hash<GL4>>(false, 2, false, 4, 20, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_variation::<GL, GL4, Sha2Hash<GL4>>(true, 2, false, 4, 20, c)
                .is_ok()
        );
        let c = parse_hash("0x840a4ad7b540a0ba59e3d7851549527608958cd7f2251cdbd5b52213471d30eb");
        assert!(
            test_vitalik_circuit_variation::<GL, GL4, Sha2Hash<GL4>>(false, 2, true, 128, 25, c)
                .is_ok()
        );
        assert!(
            test_vitalik_circuit_variation::<GL, GL4, Sha2Hash<GL4>>(true, 2, true, 128, 25, c)
                .is_ok()
        );
    }

    fn vitalik_circuit_options() -> Options {
        Options {
            canonicalize_constraints: false,
            blowup_log2: 1,
            blind: false,
            ..Options::default()
        }
    }

    fn build_vitalik_circuit() -> Circuit<BS, BS> {
        build_vitalik_circuit_with_options(vitalik_circuit_options())
    }

    fn build_vitalik_circuit_with_options(options: Options) -> Circuit<BS, BS> {
        build_generic_vitalik_circuit(options)
    }

    /// Builds Vitalik's circuit over any field.
    fn build_generic_vitalik_circuit<F: Field, G: Field256<BaseField = F>>(
        options: Options,
    ) -> Circuit<F, G> {
        let mut builder = CircuitBuilder::<F, G>::default();
        builder.add_gate(0, (var(0) ^ 2) - var(1));
        builder.connect(cell(0, 0).into(), cell(1, 0).into());
        builder.connect(cell(0, 1).into(), cell(1, 1).into());
        builder.add_gate(1, var(0) * var(1) + var(0) + 5 - var(2));
        builder.connect(cell(0, 0).into(), cell(2, 0).into());
        builder.connect(cell(1, 2).into(), cell(2, 1).into());
        builder.add_gate(2, Constraint::nop());
        builder.declare_public_cells([cell(2, 0), cell(2, 1)]);
        builder.build(options).unwrap()
    }

    #[test]
    fn test_check_witness_detects_wrong_number_of_rows() {
        let circuit = build_vitalik_circuit();
        let witness = Witness::new(4, 3, circuit.pcs_config().degree_bound);
        let error = circuit.check_witness(&witness).unwrap_err();
        assert!(error.to_string().contains("wrong number of rows"));
    }

    #[test]
    fn test_check_witness_detects_wrong_number_of_columns() {
        let circuit = build_vitalik_circuit();
        let witness = Witness::new(3, 4, circuit.pcs_config().degree_bound);
        let error = circuit.check_witness(&witness).unwrap_err();
        assert!(error.to_string().contains("wrong number of columns"));
    }

    #[test]
    fn test_check_witness_detects_wrong_degree_bound() {
        let circuit = build_vitalik_circuit();
        let witness = Witness::new(3, 3, circuit.pcs_config().degree_bound * 2);
        let error = circuit.check_witness(&witness).unwrap_err();
        assert!(error.to_string().contains("incorrect degree bound"));
    }

    #[test]
    fn test_check_witness_detects_gate_constraint_violation() {
        let circuit = build_vitalik_circuit();
        let mut witness = circuit.make_witness();
        witness.set(cell(0, 0), from_const(3));
        witness.set(cell(0, 1), from_const(10));
        witness.set(cell(1, 0), from_const(3));
        witness.set(cell(1, 1), from_const(9));
        witness.set(cell(1, 2), from_const(35));
        witness.set(cell(2, 0), from_const(3));
        witness.set(cell(2, 1), from_const(35));
        let error = circuit.check_witness(&witness).unwrap_err();
        assert!(error.to_string().contains("gate constraint"));
    }

    #[test]
    fn test_check_witness_detects_direct_wire_constraint_violation() {
        let circuit = build_vitalik_circuit();
        let mut witness = circuit.make_witness();
        witness.set(cell(0, 0), from_const(3));
        witness.set(cell(0, 1), from_const(9));
        witness.set(cell(1, 0), from_const(4));
        witness.set(cell(1, 1), from_const(9));
        witness.set(cell(1, 2), from_const(45));
        witness.set(cell(2, 0), from_const(3));
        witness.set(cell(2, 1), from_const(45));
        let error = circuit.check_witness(&witness).unwrap_err();
        assert!(error.to_string().contains("wire constraint violated"));
    }

    #[test]
    fn test_check_witness_detects_transitive_wire_constraint_violation() {
        let circuit = build_vitalik_circuit();
        let mut witness = circuit.make_witness();
        witness.set(cell(0, 0), from_const(3));
        witness.set(cell(0, 1), from_const(9));
        witness.set(cell(1, 0), from_const(3));
        witness.set(cell(1, 1), from_const(9));
        witness.set(cell(1, 2), from_const(35));
        witness.set(cell(2, 0), from_const(99));
        witness.set(cell(2, 1), from_const(35));
        let error = circuit.check_witness(&witness).unwrap_err();
        assert!(error.to_string().contains("wire constraint violated"));
    }

    fn prove_vitalik_circuit() -> (
        CompressedCircuit<BS, BS, Sha2Hash<BS>>,
        Proof<BS, BS, Sha2Hash<BS>>,
    ) {
        prove_vitalik_circuit_with_options(vitalik_circuit_options())
    }

    fn prove_vitalik_circuit_with_options(
        options: Options,
    ) -> (
        CompressedCircuit<BS, BS, Sha2Hash<BS>>,
        Proof<BS, BS, Sha2Hash<BS>>,
    ) {
        prove_generic_vitalik_circuit(options)
    }

    /// Builds Vitalik's circuit over any field and proves it with a valid witness.
    fn prove_generic_vitalik_circuit<F: Field, G: Field256<BaseField = F>, H: Hasher<G>>(
        options: Options,
    ) -> (CompressedCircuit<F, G, H>, Proof<F, G, H>) {
        let circuit = build_generic_vitalik_circuit::<F, G>(options);
        let mut witness = circuit.make_witness();
        witness.set(cell(0, 0), 3u8.into());
        witness.set(cell(0, 1), 9u8.into());
        witness.set(cell(1, 0), 3u8.into());
        witness.set(cell(1, 1), 9u8.into());
        witness.set(cell(1, 2), 35u8.into());
        witness.set(cell(2, 0), 3u8.into());
        witness.set(cell(2, 1), 35u8.into());
        let proof = circuit.prove(witness).unwrap();
        (circuit.to_compressed(), proof)
    }

    #[test]
    fn test_public_cells_are_proven() {
        let (circuit, proof) = prove_vitalik_circuit();
        let public_inputs = circuit.verify(&proof).unwrap();
        assert_eq!(public_inputs.len(), 2);
        assert_eq!(public_inputs[&cell(2, 0)], from_const(3));
        assert_eq!(public_inputs[&cell(2, 1)], from_const(35));
    }

    #[test]
    fn test_tampered_public_cell_is_rejected() {
        let (circuit, mut proof) = prove_vitalik_circuit();
        proof.public_inputs.insert(cell(2, 1), from_const(36));
        let error = circuit.verify(&proof).unwrap_err();
        assert!(error.to_string().contains("constraint violation"));
    }

    #[test]
    fn test_missing_public_cell_is_rejected() {
        let (circuit, mut proof) = prove_vitalik_circuit();
        proof.public_inputs.remove(&cell(2, 1));
        let error = circuit.verify(&proof).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("doesn't have an opening for public cell")
        );
    }

    #[test]
    fn test_wrong_circuit_batch_size_is_rejected() {
        let (circuit, mut proof) = prove_vitalik_circuit();
        let mut commitment = proof.commitment.to_proto();
        commitment.batches[COMMIT_INDEX_CIRCUIT].num_polys -= 1;
        commitment.batches[COMMIT_INDEX_WITNESS].num_polys += 1;
        proof.commitment = pcs::Commitment::from_proto(&commitment).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid number of circuit polynomials")
        );
    }

    #[test]
    fn test_wrong_witness_batch_size_is_rejected() {
        let (circuit, mut proof) = prove_vitalik_circuit();
        let mut commitment = proof.commitment.to_proto();
        commitment.batches[COMMIT_INDEX_WITNESS].num_polys -= 1;
        commitment.batches[COMMIT_INDEX_PERMUTATION_ARGUMENT].num_polys += 1;
        proof.commitment = pcs::Commitment::from_proto(&commitment).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid number of witness polynomials")
        );
    }

    #[test]
    fn test_wrong_permutation_argument_batch_size_is_rejected() {
        let (circuit, mut proof) = prove_vitalik_circuit();
        let mut commitment = proof.commitment.to_proto();
        commitment.batches[COMMIT_INDEX_PERMUTATION_ARGUMENT].num_polys -= 1;
        commitment.batches[COMMIT_INDEX_QUOTIENT].num_polys += 1;
        proof.commitment = pcs::Commitment::from_proto(&commitment).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid number of permutation argument polynomials")
        );
    }

    #[test]
    fn test_wrong_quotient_batch_size_is_rejected() {
        let (circuit, mut proof) = prove_vitalik_circuit();
        let mut commitment = proof.commitment.to_proto();
        commitment.batches[COMMIT_INDEX_QUOTIENT].num_polys -= 1;
        proof.commitment = pcs::Commitment::from_proto(&commitment).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid number of polynomials in the quotient batch")
        );
    }

    #[test]
    fn test_circuit_batch_over_extension_field_is_rejected() {
        let (circuit, mut proof) = prove_vitalik_circuit();
        let mut commitment = proof.commitment.to_proto();
        commitment.batches[COMMIT_INDEX_CIRCUIT].base_field = false;
        proof.commitment = pcs::Commitment::from_proto(&commitment).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("the circuit polynomials must be over the base field")
        );
    }

    /// A witness committed over `G` rather than `F` wouldn't be proven to be over `F`.
    #[test]
    fn test_witness_batch_over_extension_field_is_rejected() {
        let (circuit, mut proof) = prove_vitalik_circuit();
        let mut commitment = proof.commitment.to_proto();
        commitment.batches[COMMIT_INDEX_WITNESS].base_field = false;
        proof.commitment = pcs::Commitment::from_proto(&commitment).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("the witness polynomials must be over the base field")
        );
    }

    #[test]
    fn test_permutation_argument_batch_over_base_field_is_rejected() {
        let (circuit, mut proof) = prove_vitalik_circuit();
        let mut commitment = proof.commitment.to_proto();
        commitment.batches[COMMIT_INDEX_PERMUTATION_ARGUMENT].base_field = true;
        proof.commitment = pcs::Commitment::from_proto(&commitment).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("the permutation argument polynomials must not be over the base field")
        );
    }

    #[test]
    fn test_quotient_batch_over_base_field_is_rejected() {
        let (circuit, mut proof) = prove_vitalik_circuit();
        let mut commitment = proof.commitment.to_proto();
        commitment.batches[COMMIT_INDEX_QUOTIENT].base_field = true;
        proof.commitment = pcs::Commitment::from_proto(&commitment).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("the polynomials in the quotient batch must not be over the base field")
        );
    }

    /// Vitalik's circuit has 3 witness columns, one fewer than the extension degree of
    /// Goldilocks^4, so blinding adds one witness randomizer.
    #[test]
    fn test_blinded_witness_batch_is_padded_to_extension_degree() {
        let (circuit, proof) = prove_generic_vitalik_circuit::<GL, GL4, Sha2Hash<GL4>>(Options {
            blind: true,
            ..vitalik_circuit_options()
        });
        let batch = proof.commitment.batches()[COMMIT_INDEX_WITNESS];
        assert!(batch.base_field());
        assert_eq!(batch.num_polys(), 4);
        assert!(circuit.verify(&proof).is_ok());
    }

    #[test]
    fn test_unblinded_witness_batch_is_not_padded() {
        let (circuit, proof) = prove_generic_vitalik_circuit::<GL, GL4, Sha2Hash<GL4>>(Options {
            blind: false,
            ..vitalik_circuit_options()
        });
        let batch = proof.commitment.batches()[COMMIT_INDEX_WITNESS];
        assert!(batch.base_field());
        assert_eq!(batch.num_polys(), 3);
        assert!(circuit.verify(&proof).is_ok());
    }

    /// The extension degree of BlueSky over itself is 1.
    #[test]
    fn test_blinded_witness_batch_is_not_padded_with_bluesky() {
        let (circuit, proof) = prove_vitalik_circuit_with_options(Options {
            blind: true,
            ..vitalik_circuit_options()
        });
        let batch = proof.commitment.batches()[COMMIT_INDEX_WITNESS];
        assert!(batch.base_field());
        assert_eq!(batch.num_polys(), 3);
        assert!(circuit.verify(&proof).is_ok());
    }

    #[test]
    fn test_witness_batch_without_randomizer_is_rejected() {
        let (circuit, mut proof) =
            prove_generic_vitalik_circuit::<GL, GL4, Sha2Hash<GL4>>(Options {
                blind: true,
                ..vitalik_circuit_options()
            });
        let mut commitment = proof.commitment.to_proto();
        commitment.batches[COMMIT_INDEX_WITNESS].num_polys -= 1;
        commitment.batches[COMMIT_INDEX_PERMUTATION_ARGUMENT].num_polys += 1;
        proof.commitment = pcs::Commitment::from_proto(&commitment).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid number of witness polynomials")
        );
    }

    #[test]
    fn test_extra_opening_is_not_reported_as_public() {
        let (circuit, mut proof) = prove_vitalik_circuit();
        proof.public_inputs.insert(cell(1, 2), from_const(35));
        let public_inputs = circuit.verify(&proof).unwrap();
        assert_eq!(public_inputs.len(), 2);
        assert!(!public_inputs.contains_key(&cell(1, 2)));
    }

    #[test]
    fn test_oversized_selector_is_rejected() {
        let mut proto = build_vitalik_circuit().to_proto();
        let degree_bound = proto.pcs_config.as_ref().unwrap().degree_bound as usize;
        proto.selectors[0]
            .coefficients
            .resize(degree_bound + 1, vec![1]);
        let error = Circuit::<BS, BS>::from_proto(&proto).unwrap_err();
        assert!(error.to_string().contains("invalid selector"));
    }

    #[test]
    fn test_truncated_sigma_table_is_rejected() {
        let mut proto = build_vitalik_circuit().to_proto();
        proto.sigma_table.pop();
        let error = Circuit::<BS, BS>::from_proto(&proto).unwrap_err();
        assert!(error.to_string().contains("invalid sigma table size"));
    }

    #[test]
    fn test_misplaced_gate_is_rejected() {
        let mut proto = build_vitalik_circuit().to_proto();
        proto.gates[1].instances[0].column_index = 1;
        let error = Circuit::<BS, BS>::from_proto(&proto).unwrap_err();
        assert!(error.to_string().contains("invalid gate placement"));
    }

    #[test]
    fn test_misplaced_compressed_gate_is_rejected() {
        let (circuit, _) = prove_vitalik_circuit();
        let mut proto = circuit.to_proto();
        proto.gates[1].instances[0].column_index = 1;
        let error = CompressedCircuit::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).unwrap_err();
        assert!(error.to_string().contains("invalid gate placement"));
    }

    #[test]
    fn test_sparse_selector_indices_are_rejected() {
        let (circuit, proof) = prove_vitalik_circuit();
        let mut proto = circuit.to_proto();
        for gate in &mut proto.gates {
            for instance in &mut gate.instances {
                if instance.selector_index == 2 {
                    instance.selector_index = 99;
                }
            }
        }
        let circuit = CompressedCircuit::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(error.to_string().contains("invalid selector index 99"));
    }

    #[test]
    fn test_circuit_serialization() {
        let circuit = build_vitalik_circuit();
        let mut witness = circuit.make_witness();
        witness.set(cell(0, 0), from_const(3));
        witness.set(cell(0, 1), from_const(9));
        witness.set(cell(1, 0), from_const(3));
        witness.set(cell(1, 1), from_const(9));
        witness.set(cell(1, 2), from_const(35));
        witness.set(cell(2, 0), from_const(3));
        witness.set(cell(2, 1), from_const(35));
        let proof = circuit.prove::<Sha2Hash<BS>>(witness).unwrap();
        let circuit = Circuit::<BS, BS>::from_proto(&circuit.to_proto()).unwrap();
        let public_inputs = circuit.verify(&proof).unwrap();
        assert_eq!(public_inputs.len(), 2);
        assert_eq!(public_inputs[&cell(2, 0)], from_const(3));
        assert_eq!(public_inputs[&cell(2, 1)], from_const(35));
    }

    #[test]
    fn test_compressed_circuit_serialization() {
        let (circuit, proof) = prove_vitalik_circuit();
        let circuit =
            CompressedCircuit::<BS, BS, Sha2Hash<BS>>::from_proto(&circuit.to_proto()).unwrap();
        let public_inputs = circuit.verify(&proof).unwrap();
        assert_eq!(public_inputs.len(), 2);
        assert_eq!(public_inputs[&cell(2, 0)], from_const(3));
        assert_eq!(public_inputs[&cell(2, 1)], from_const(35));
    }

    #[test]
    fn test_reordered_gates_are_accepted() {
        let (circuit, proof) = prove_vitalik_circuit();
        let mut proto = circuit.to_proto();
        proto.gates.reverse();
        let circuit = CompressedCircuit::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        let public_inputs = circuit.verify(&proof).unwrap();
        assert_eq!(public_inputs.len(), 2);
        assert_eq!(public_inputs[&cell(2, 0)], from_const(3));
        assert_eq!(public_inputs[&cell(2, 1)], from_const(35));
    }

    #[test]
    fn test_circuit_without_rows_is_rejected() {
        let mut proto = build_vitalik_circuit().to_proto();
        proto.num_rows = 0;
        let error = Circuit::<BS, BS>::from_proto(&proto).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("circuits must have at least one row")
        );
    }

    #[test]
    fn test_compressed_circuit_without_hash_backend_is_rejected() {
        let (circuit, _) = prove_vitalik_circuit();
        let mut proto = circuit.to_proto();
        proto.hash_backend = None;
        let error = CompressedCircuit::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).unwrap_err();
        assert!(error.to_string().contains("hash backend not specified"));
    }

    #[test]
    fn test_compressed_circuit_with_wrong_hash_backend_is_rejected() {
        let (circuit, _) = prove_vitalik_circuit();
        let error = CompressedCircuit::<BS, BS, Keccak256Hash<BS>>::from_proto(&circuit.to_proto())
            .unwrap_err();
        assert!(error.to_string().contains("incorrect hash backend"));
    }

    #[test]
    fn test_unblinded_proofs_are_deterministic() {
        let (_, proof1) = prove_vitalik_circuit();
        let (_, proof2) = prove_vitalik_circuit();
        assert_eq!(proof1.commitment.to_proto(), proof2.commitment.to_proto());
        assert_eq!(proof1.inner_proof.to_proto(), proof2.inner_proof.to_proto());
    }

    #[test]
    fn test_proof_serialization() {
        let (circuit, proof) = prove_vitalik_circuit();
        let proto = proof.to_proto();
        let proof = Proof::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        assert_eq!(proof.to_proto(), proto);
        let public_inputs = circuit.verify(&proof).unwrap();
        assert_eq!(public_inputs.len(), 2);
        assert_eq!(public_inputs[&cell(2, 0)], from_const(3));
        assert_eq!(public_inputs[&cell(2, 1)], from_const(35));
    }

    #[test]
    fn test_proof_without_commitment_is_rejected() {
        let (_, proof) = prove_vitalik_circuit();
        let mut proto = proof.to_proto();
        proto.commitment = None;
        let error = Proof::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).unwrap_err();
        assert!(error.to_string().contains("missing PCS commitment"));
    }

    #[test]
    fn test_proof_without_inner_proof_is_rejected() {
        let (_, proof) = prove_vitalik_circuit();
        let mut proto = proof.to_proto();
        proto.inner_proof = None;
        let error = Proof::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).unwrap_err();
        assert!(error.to_string().contains("missing PCS proof"));
    }

    #[test]
    fn test_proof_with_oversized_public_input_is_rejected() {
        let (_, proof) = prove_vitalik_circuit();
        let mut proto = proof.to_proto();
        proto.public_inputs[0].value.push(0);
        let error = Proof::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).unwrap_err();
        assert!(error.to_string().contains("invalid scalar length"));
    }

    #[test]
    fn test_proof_with_wrong_hash_backend_is_rejected() {
        let (_, proof) = prove_vitalik_circuit();
        let error = Proof::<BS, BS, Keccak256Hash<BS>>::from_proto(&proof.to_proto()).unwrap_err();
        assert!(error.to_string().contains("incorrect hash backend"));
    }

    #[test]
    fn test_tampered_serialized_public_input_is_rejected() {
        let (circuit, proof) = prove_vitalik_circuit();
        let mut proto = proof.to_proto();
        proto.public_inputs[1].value = utils::store_scalar(from_const(36));
        let proof = Proof::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(error.to_string().contains("constraint violation"));
    }

    #[test]
    fn test_blinded_circuit_without_room_for_masks_is_rejected() {
        let (circuit, proof) = prove_vitalik_circuit();
        let mut proto = circuit.to_proto();
        proto.blinded = true;
        let circuit = CompressedCircuit::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        let error = circuit.verify(&proof).unwrap_err();
        assert!(error.to_string().contains("too small for blinding"));
    }

    #[test]
    fn test_blinded_circuit_with_grinding() {
        // 8 grinding bits bring the queries from 128 down to 120, so the 3 rows and the 123
        // blinding rows fit in 128 rather than 256.
        let (circuit, proof) = prove_vitalik_circuit_with_options(Options {
            grinding_bits: 8,
            blind: true,
            ..vitalik_circuit_options()
        });
        assert_eq!(circuit.pcs_config().degree_bound, 128);
        assert_eq!(circuit.pcs_config().grinding_bits, 8);
        assert_eq!(proof.pcs_config(), circuit.pcs_config());
        let public_inputs = circuit.verify(&proof).unwrap();
        assert_eq!(public_inputs.len(), 2);
        assert_eq!(public_inputs[&cell(2, 0)], from_const(3));
        assert_eq!(public_inputs[&cell(2, 1)], from_const(35));
    }

    #[test]
    fn test_circuit_with_fri_rounds() {
        let (circuit, proof) = prove_vitalik_circuit_with_options(Options {
            fri_fold_rate: 2,
            max_fri_rounds: 1,
            ..vitalik_circuit_options()
        });
        assert_eq!(circuit.pcs_config().fold_rate, 2);
        assert_eq!(circuit.pcs_config().max_rounds, 1);
        assert_eq!(proof.pcs_config().num_rounds(), 1);
        let public_inputs = circuit.verify(&proof).unwrap();
        assert_eq!(public_inputs.len(), 2);
        assert_eq!(public_inputs[&cell(2, 0)], from_const(3));
        assert_eq!(public_inputs[&cell(2, 1)], from_const(35));
    }

    fn test_mismatched_pcs_config_is_rejected_impl(
        circuit_options: Options,
        proof_options: Options,
    ) {
        let (circuit, _) = prove_vitalik_circuit_with_options(circuit_options);
        let (_, proof) = prove_vitalik_circuit_with_options(proof_options);
        let error = circuit.verify(&proof).unwrap_err();
        assert!(error.to_string().contains("PCS settings mismatch"));
    }

    #[test]
    fn test_mismatched_pcs_config_is_rejected() {
        test_mismatched_pcs_config_is_rejected_impl(
            vitalik_circuit_options(),
            Options {
                fri_fold_rate: 2,
                ..vitalik_circuit_options()
            },
        );
        test_mismatched_pcs_config_is_rejected_impl(
            vitalik_circuit_options(),
            Options {
                max_fri_rounds: 1,
                ..vitalik_circuit_options()
            },
        );
        test_mismatched_pcs_config_is_rejected_impl(
            Options {
                grinding_bits: 8,
                ..vitalik_circuit_options()
            },
            vitalik_circuit_options(),
        );
    }

    #[test]
    fn test_proof_with_more_grinding_bits_is_accepted() {
        let (circuit, _) = prove_vitalik_circuit();
        let (_, proof) = prove_vitalik_circuit_with_options(Options {
            grinding_bits: 8,
            ..vitalik_circuit_options()
        });
        let public_inputs = circuit.verify(&proof).unwrap();
        assert_eq!(public_inputs.len(), 2);
        assert_eq!(public_inputs[&cell(2, 0)], from_const(3));
        assert_eq!(public_inputs[&cell(2, 1)], from_const(35));
    }

    #[test]
    fn test_pcs_config_serialization() {
        let circuit = build_vitalik_circuit_with_options(Options {
            fri_fold_rate: 2,
            max_fri_rounds: 1,
            grinding_bits: 8,
            ..vitalik_circuit_options()
        });
        let pcs_config = *circuit.pcs_config();
        let proto = circuit.to_proto();
        let compressed_circuit = circuit.to_compressed::<Sha2Hash<BS>>();
        let circuit = Circuit::<BS, BS>::from_proto(&proto).unwrap();
        let compressed_circuit =
            CompressedCircuit::<BS, BS, Sha2Hash<BS>>::from_proto(&compressed_circuit.to_proto())
                .unwrap();
        assert_eq!(*circuit.pcs_config(), pcs_config);
        assert_eq!(*compressed_circuit.pcs_config(), pcs_config);
    }

    #[test]
    fn test_circuit_without_pcs_config_is_rejected() {
        let mut proto = build_vitalik_circuit().to_proto();
        proto.pcs_config = None;
        let error = Circuit::<BS, BS>::from_proto(&proto).unwrap_err();
        assert!(error.to_string().contains("missing PCS settings"));
    }

    #[test]
    fn test_compressed_circuit_without_pcs_config_is_rejected() {
        let (circuit, _) = prove_vitalik_circuit();
        let mut proto = circuit.to_proto();
        proto.pcs_config = None;
        let error = CompressedCircuit::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).unwrap_err();
        assert!(error.to_string().contains("missing PCS settings"));
    }

    #[test]
    fn test_circuit_with_invalid_fold_rate_is_rejected() {
        let mut proto = build_vitalik_circuit().to_proto();
        proto.pcs_config.as_mut().unwrap().fold_rate = 0;
        assert!(Circuit::<BS, BS>::from_proto(&proto).is_err());
    }

    #[test]
    fn test_compressed_circuit_with_too_many_grinding_bits_is_rejected() {
        let (circuit, _) = prove_vitalik_circuit();
        let mut proto = circuit.to_proto();
        proto.pcs_config.as_mut().unwrap().grinding_bits = pcs::PROOF_OF_WORK_BITS as u32 + 1;
        assert!(CompressedCircuit::<BS, BS, Sha2Hash<BS>>::from_proto(&proto).is_err());
    }
}
