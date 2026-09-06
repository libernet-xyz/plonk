use crate::chip::Chip;
use crate::expr::{Constraint, Variable};
use crate::utils::{self, padded_circuit_size};
use crate::witness::{Cell, CellOffset, Partitioner, Witness, WitnessView};
use anyhow::{Result, anyhow};
use primitive_types::H256;
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
        .map(|constraint| {
            constraint
                .get_free_variables()
                .iter()
                .map(Variable::rotation)
                .collect::<BTreeSet<isize>>()
                .into_iter()
        })
        .flatten()
        .chain([0, 1])
        .collect::<BTreeSet<isize>>()
}

/// Calculates the degree bound of the PLONK quotient, typically much higher than the circuit's
/// general [degree bound](`Circuit::degree_bound`) `N` because the constraint equations involve
/// several polynomial multiplications, such as the gate selectors multiplied by the gate
/// constraints combined with the witness columns.
///
/// The algorithm uses the formula `(N - 1) * E`, where `E = max(max_gate_degree, num_columns)`.
/// The rationale behind it is:
///
/// * each column has degree less than or equal to `N - 1`;
/// * the grand gate constraint has degree less than or equal to
///   `(N - 1) * (1 + max_gate_degree)` (the selector contributes one factor, degree composition
///   with the constraint columns contributes `max_gate_degree` more);
/// * the recurrence constraint of the permutation argument has degree less than or equal to
///   `(N - 1) * (1 + num_columns)` (the accumulator/shifted term contributes one factor, one more
///   per column);
/// * the grand PLONK constraint (grand gate constraint + permutation argument fixpoint constraint
///   + permutation argument recurrence constraint) has degree less than or equal to
///   `(N - 1) * (1 + E)`;
/// * dividing that by the zero polynomial (`x^N - 1`, degree-N) yields a quotient with degree
///   `(N - 1) * (1 + E) - N`;
/// * so the degree bound of the quotient is `(N - 1) * (1 + E) - N + 1`
/// * ... which simplifies to `(N - 1) * E`.
fn quotient_degree_bound<'a, F: Field>(
    degree_bound: usize,
    num_columns: usize,
    gate_constraints: impl Iterator<Item = &'a Constraint<F>>,
) -> usize {
    let max_gate_degree = gate_constraints
        .map(|constraint| constraint.get_degree())
        .max()
        .unwrap_or(0);
    (degree_bound - 1) * std::cmp::max(max_gate_degree, num_columns)
}

/// Circuit compilation & proving options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilationOptions {
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
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            canonicalize_constraints: false,
        }
    }
}

/// Circuit compilation & proving options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvingOptions {
    /// Log2 of the blowup factor used to compute the low-degree extensions for the underlying PCS.
    pub blowup_log2: usize,
}

impl Default for ProvingOptions {
    fn default() -> Self {
        Self {
            blowup_log2: OPTIONS_DEFAULT_BLOWUP_LOG2,
        }
    }
}

pub trait CircuitView<F: Field, G: Field256 + From<F>> {
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
pub struct CircuitViewGenerator<'a, F: Field, G: Field256 + From<F>> {
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

impl<'a, F: Field, G: Field256 + From<F>> CircuitViewGenerator<'a, F, G> {
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
pub struct CircuitBuilder<F: Field, G: Field256 + From<F>> {
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

    /// List of rows that are revealed in the proofs.
    public_rows: BTreeSet<usize>,

    _data: PhantomData<G>,
}

impl<F: Field, G: Field256 + From<F>> CircuitBuilder<F, G> {
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
    pub fn declare_public_rows<I: IntoIterator<Item = usize>>(&mut self, gates: I) {
        self.public_rows = BTreeSet::from_iter(gates);
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
    pub fn build(mut self, options: CompilationOptions) -> Result<Circuit<F, G>> {
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

        let (degree_bound, num_blinding_rows) = padded_circuit_size(
            self.num_rows,
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
            let omega = Polynomial::<F>::domain_element2(1, degree_bound);
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
            num_blinding_rows,
            degree_bound,
            num_columns: self.num_columns,
            selectors,
            gates,
            sigma,
            sigma_values,
            public_rows: self.public_rows,
            _data: PhantomData,
        })
    }
}

impl<F: Field, G: Field256 + From<F>> CircuitView<F, G> for CircuitBuilder<F, G> {
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
pub struct CircuitSectionBuilder<'a, F: Field, G: Field256 + From<F>> {
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

impl<'a, F: Field, G: Field256 + From<F>> CircuitSectionBuilder<'a, F, G> {
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

impl<'a, F: Field, G: Field256 + From<F>> CircuitView<F, G> for CircuitSectionBuilder<'a, F, G> {
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
pub struct Proof<F: Field256, H: Hasher<F>> {
    commitment: pcs::Commitment<F, H>,
    inner_proof: pcs::Proof<F, H>,
}

impl<F: Field256, H: Hasher<F>> Proof<F, H> {
    /// Returns the proven degree bound.
    pub fn degree_bound(&self) -> usize {
        self.inner_proof.degree_bound()
    }

    /// Returns the base-2 logarithm of the blowup factor used in the proof.
    pub fn blowup_log2(&self) -> usize {
        self.inner_proof.blowup_log2()
    }

    /// Returns the size of the extended evaluation domain.
    pub fn extended_domain_size(&self) -> usize {
        self.inner_proof.extended_domain_size()
    }

    /// Returns the number of committed polynomials.
    ///
    /// These include the circuit selectors and sigma polynomials, the witness columns, and the
    /// chunks of the grand quotient.
    pub fn num_polys(&self) -> usize {
        self.inner_proof.num_polys()
    }
}

/// A PLONK circuit.
#[derive(Debug, Clone)]
pub struct Circuit<F: Field, G: Field256 + From<F>> {
    /// The raw number of rows of the circuit.
    ///
    /// Unlike [`Self::degree_bound`], this count doesn't include the blinding rows and is not
    /// padded to the next power of 2.
    num_rows: usize,

    /// Number of blinding rows used in the circuit.
    ///
    /// This is calculated by [`padded_circuit_size`] and depends on how many different variable
    /// rotations were used across all constraints.
    num_blinding_rows: usize,

    /// Number of witness rows (including the blinding rows) rounded up to the next power of 2.
    ///
    /// This is the direct result of [`padded_circuit_size`].
    degree_bound: usize,

    /// Number of witness columns.
    num_columns: usize,

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

    /// List of gates that are revealed in the proofs. Each element is a row index.
    public_rows: BTreeSet<usize>,

    _data: PhantomData<G>,
}

impl<F: Field, G: Field256 + From<F>> Circuit<F, G> {
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    pub fn num_gates(&self) -> usize {
        self.gates
            .iter()
            .map(|(_, instances)| instances.len())
            .sum()
    }

    pub fn public_rows(&self) -> &BTreeSet<usize> {
        &self.public_rows
    }

    /// Makes an empty [`Witness`] objects suitable for use with this circuit.
    pub fn make_witness(&self) -> Witness<F> {
        Witness::new(
            self.num_rows,
            self.num_columns,
            self.gates.iter().flat_map(|(constraint, _)| {
                constraint
                    .get_free_variables()
                    .iter()
                    .map(Variable::rotation)
                    .collect::<BTreeSet<isize>>()
            }),
        )
    }

    /// Performs various checks to verify that the provided `witness` is compatible with this
    /// circuit and doesn't violate any constraints.
    pub fn check_witness(&self, witness: &Witness<F>) -> Result<()> {
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
        if witness.degree_bound() != self.degree_bound {
            return Err(anyhow!(
                "incorrect degree bound: got {}, want {}",
                witness.degree_bound(),
                self.degree_bound
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
            let omega = Polynomial::<F>::domain_element2(1, self.degree_bound);
            let mut generator_power = F::ONE;
            for column in 0..self.num_columns {
                let mut omega_power = F::ONE;
                for row in 0..self.degree_bound {
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

    fn make_committer<H: Hasher<G>>(&self, options: &ProvingOptions) -> pcs::Committer<G, H> {
        let circuit_polynomials = self
            .selectors
            .iter()
            .map(Self::embed_polynomial)
            .chain(self.sigma.iter().map(Self::embed_polynomial))
            .collect();
        pcs::Committer::<G, H>::new(self.degree_bound, options.blowup_log2, circuit_polynomials)
    }

    fn get_variable_substitution(
        &self,
        omega: G,
        omega_inv: G,
        columns: &[Polynomial<F>],
    ) -> BTreeMap<Variable, Polynomial<G>> {
        let mut substitution: BTreeMap<Variable, Polynomial<G>> = BTreeMap::default();
        for (constraint, instances) in &self.gates {
            for instance in instances {
                for variable in constraint
                    .get_free_variables()
                    .into_iter()
                    .map(|variable| variable.remap(instance.column_index as isize))
                {
                    if !substitution.contains_key(&variable) {
                        let column = variable.rotate_column(
                            Self::embed_polynomial(&columns[variable.column_index()]),
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

    /// Builds the three polynomials used in the permutation argument. The components of the
    /// returned tuple are, respectively: the coordinate pair accumulator, the fixpoint constraint,
    /// and the recurrence constraint.
    fn build_permutation_argument(
        &self,
        witness: &Witness<F>,
        columns: &[Polynomial<F>],
        beta: G,
        gamma: G,
    ) -> Result<(Polynomial<G>, Polynomial<G>, Polynomial<G>)> {
        let omega = Polynomial::<G>::domain_element2(1, self.degree_bound);

        let accumulator = {
            let mut accumulator = vec![G::ZERO; self.degree_bound + 1];

            accumulator[0] = G::ONE;
            let mut omega_power = G::ONE;
            for i in 0..self.degree_bound {
                let mut generator_power = G::ONE;
                accumulator[i + 1] = accumulator[i];
                for j in 0..self.num_columns {
                    let witness_value: G = witness.get_at(Cell::new(i, j)).into();
                    accumulator[i + 1] *=
                        witness_value + beta * generator_power * omega_power + gamma;
                    accumulator[i + 1] *=
                        (witness_value + beta * G::from(self.sigma_values[j][i]) + gamma)
                            .invert_unwrap();
                    generator_power *= G::MULTIPLICATIVE_GENERATOR;
                }
                omega_power *= omega;
            }

            if accumulator.pop().unwrap() != G::ONE {
                return Err(anyhow!("permutation accumulator wraparound check failed"));
            }

            Polynomial::encode2(accumulator)
        };

        let shifted = accumulator.clone().shift_domain_by(omega);

        let recurrence_constraint = {
            let lhs = Polynomial::multiply_batch(
                std::iter::once(shifted)
                    .chain(
                        columns
                            .iter()
                            .zip(self.sigma.iter())
                            .map(|(column, sigma)| {
                                Self::embed_polynomial(column)
                                    + Self::embed_polynomial(sigma) * beta
                                    + gamma
                            }),
                    )
                    .collect::<Vec<_>>()
                    .as_slice(),
            );

            let mut power = G::ONE;
            let rhs = Polynomial::multiply_batch(
                std::iter::once(&accumulator).chain(
                    columns
                        .iter()
                        .map(|column| {
                            let column = Self::embed_polynomial(column)
                                + Polynomial::with_coefficients(vec![gamma, beta * power]);
                            power *= G::MULTIPLICATIVE_GENERATOR;
                            column
                        })
                        .collect::<Vec<_>>()
                        .as_slice(),
                ),
            );

            lhs - rhs
        };

        let fixpoint_constraint =
            (accumulator.clone() - G::ONE) * Polynomial::lagrange0_2(self.degree_bound).clone();

        Ok((accumulator, fixpoint_constraint, recurrence_constraint))
    }

    /// Splits the quotient polynomial in chunks so that it can be batch-committed even though its
    /// degree is much higher than the bound configured in the underlying PCS.
    fn split_quotient(&self, quotient: Polynomial<G>) -> Vec<Polynomial<G>> {
        let degree_bound = quotient_degree_bound(
            self.degree_bound,
            self.num_columns,
            self.gates.iter().map(|(constraint, _)| constraint),
        );
        let mut coefficients = quotient.take();
        assert!(coefficients.len() <= degree_bound);
        coefficients.resize(degree_bound, G::ZERO);
        coefficients
            .chunks(self.degree_bound)
            .map(|coefficients| Polynomial::with_coefficients(coefficients.to_vec()))
            .collect()
    }

    /// Proves correctness of the given witness, or returns an error in case of a constraint
    /// violation.
    pub fn prove<H: Hasher<G>>(
        &self,
        mut witness: Witness<F>,
        options: ProvingOptions,
    ) -> Result<Proof<G, H>> {
        witness.blind();
        if witness.num_rows() != self.num_rows {
            return Err(anyhow!(
                "incorrect witness size (got {} rows, want {})",
                witness.num_rows(),
                self.num_rows
            ));
        }
        if witness.degree_bound() != self.degree_bound {
            return Err(anyhow!(
                "incorrect witness degree bound (got {}, want {})",
                witness.degree_bound(),
                self.degree_bound
            ));
        }
        if witness.num_columns() != self.num_columns {
            return Err(anyhow!(
                "incorrect witness size (got {} columns, want {})",
                witness.num_columns(),
                self.num_columns
            ));
        }

        let mut committer = self.make_committer::<H>(&options);

        let columns = witness.clone().encode();
        committer.add_batch(columns.iter().map(Self::embed_polynomial).collect());

        let omega = Polynomial::<G>::domain_element2(1, self.degree_bound);
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
                    let selector = self.selectors[instance.selector_index].clone();
                    gate_constraint += selector * constraint.compose2(&substitution) * power;
                    power *= delta;
                }
            }
            gate_constraint
        };

        let (
            permutation_accumulator,
            permutation_fixpoint_constraint,
            permutation_recurrence_constraint,
        ) = {
            let beta = H::challenge(*DST_BETA, &[committer.transcript_hash()]);
            let gamma = H::challenge(*DST_GAMMA, &[committer.transcript_hash()]);
            self.build_permutation_argument(&witness, columns.as_slice(), beta, gamma)?
        };
        committer.add_batch(vec![permutation_accumulator]);

        let alpha = H::challenge(*DST_ALPHA, &[committer.transcript_hash()]);

        let quotient = (gate_constraint
            + permutation_fixpoint_constraint * alpha
            + permutation_recurrence_constraint * alpha.square())
        .divide_by_zero(self.degree_bound)?;
        committer.add_batch(self.split_quotient(quotient));

        let xi = H::challenge(*DST_XI, &[committer.transcript_hash()]);

        let (commitment, prover) = committer.commit(BTreeSet::from_iter(
            get_rotation_set(self.gates.iter().map(|(constraint, _)| constraint))
                .into_iter()
                .map(|rotation| {
                    xi * if rotation < 0 { omega_inv } else { omega }
                        .pow_small(rotation.unsigned_abs())
                })
                .chain(self.public_rows.iter().map(|&row| omega.pow_small(row))),
        ));
        let inner_proof = prover.prove(&commitment);

        Ok(Proof {
            commitment,
            inner_proof,
        })
    }

    pub fn to_compressed<H: Hasher<G>>(
        self,
        options: ProvingOptions,
    ) -> CompressedCircuit<F, G, H> {
        let committer = self.make_committer::<H>(&options);
        CompressedCircuit {
            num_rows: self.num_rows,
            num_blinding_rows: self.num_blinding_rows,
            degree_bound: self.degree_bound,
            num_columns: self.num_columns,
            options,
            gates: self
                .gates
                .into_iter()
                .map(|(constraint, instances)| (constraint, instances.into_iter().collect()))
                .collect(),
            public_rows: self.public_rows,
            circuit_commitment: committer.root_hash(COMMIT_INDEX_CIRCUIT),
            _data: PhantomData,
        }
    }

    pub fn as_compressed<H: Hasher<G>>(
        &self,
        options: ProvingOptions,
    ) -> CompressedCircuit<F, G, H> {
        let committer = self.make_committer::<H>(&options);
        CompressedCircuit {
            num_rows: self.num_rows,
            num_blinding_rows: self.num_blinding_rows,
            degree_bound: self.degree_bound,
            num_columns: self.num_columns,
            options,
            gates: self
                .gates
                .iter()
                .map(|(constraint, instances)| {
                    (constraint.clone(), instances.iter().cloned().collect())
                })
                .collect(),
            public_rows: self.public_rows.clone(),
            circuit_commitment: committer.root_hash(COMMIT_INDEX_CIRCUIT),
            _data: Default::default(),
        }
    }

    pub fn verify<H: Hasher<G>>(
        &self,
        proof: &Proof<G, H>,
        options: ProvingOptions,
    ) -> Result<BTreeMap<Cell, F>> {
        self.as_compressed::<H>(options).verify(proof)
    }
}

/// A PLONK circuit in committed form.
///
/// This struct is much smaller than the original circuit but still allows full verification of a
/// proof for the circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedCircuit<F: Field, G: Field256 + From<F>, H: Hasher<G>> {
    /// The raw number of rows of the circuit.
    ///
    /// Unlike [`Self::degree_bound`], this count doesn't include the blinding rows and is not
    /// padded to the next power of 2.
    num_rows: usize,

    /// Number of blinding rows used in the circuit.
    num_blinding_rows: usize,

    /// Number of witness rows (including the blinding rows) rounded up to the next power of 2.
    degree_bound: usize,

    /// Number of witness columns.
    num_columns: usize,

    /// Proving options used to commit to this circuit (in [`Circuit::as_compressed`] or
    /// [`Circuit::to_compressed`]).
    options: ProvingOptions,

    /// Gates used in the circuit: the first component of each pair is the gate constraint and the
    /// second component is the set of instances of that gate across the circuit.
    gates: Vec<(Constraint<F>, Vec<GateInstance>)>,

    /// List of rows that are revealed in the proofs.
    public_rows: BTreeSet<usize>,

    /// Merkle root of the circuit selectors and sigma polynomials.
    circuit_commitment: H256,

    _data: PhantomData<(G, H)>,
}

impl<F: Field, G: Field256 + From<F>, H: Hasher<G>> CompressedCircuit<F, G, H> {
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    pub fn public_rows(&self) -> &BTreeSet<usize> {
        &self.public_rows
    }

    pub fn commitment(&self) -> H256 {
        self.circuit_commitment
    }

    /// Calculates the number of chunks the quotient was split into.
    ///
    /// See [`quotient_degree_bound`] for details.
    fn get_num_quotient_chunks(&self) -> usize {
        quotient_degree_bound(
            self.degree_bound,
            self.num_columns,
            self.gates.iter().map(|(constraint, _)| constraint),
        )
        .div_ceil(self.degree_bound)
    }

    fn lagrange0(x: Scalar, n: usize) -> Scalar {
        (x.pow_small(n) - Scalar::ONE)
            * (Scalar::from(n as u64) * (x - Scalar::ONE)).invert_unwrap()
    }

    /// Verifies a [`Proof`], returning the map of proven public values if successful or an error
    /// otherwise.
    ///
    /// The returned map will have exactly `N` entries for every row in [`Self::public_rows`], with
    /// `N` being the [number of columns](`Self::num_columns`).
    pub fn verify(&self, proof: &Proof<G, H>) -> Result<BTreeMap<Cell, F>> {
        let commitment = &proof.commitment;
        let inner_proof = &proof.inner_proof;

        if commitment.tree_roots().len() != NUM_COMMIT_INDICES {
            return Err(anyhow!(
                "wrong number of Merkle roots (got {}, want {})",
                commitment.tree_roots().len(),
                NUM_COMMIT_INDICES
            ));
        }
        if commitment.tree_roots()[COMMIT_INDEX_CIRCUIT] != self.circuit_commitment {
            return Err(anyhow!(
                "wrong circuit commitment (got {}, want {})",
                commitment.tree_roots()[COMMIT_INDEX_CIRCUIT],
                self.circuit_commitment
            ));
        }

        if inner_proof.degree_bound() != self.degree_bound {
            return Err(anyhow!(
                "wrong degree bound (got {}, want {})",
                inner_proof.degree_bound(),
                self.degree_bound
            ));
        }
        if inner_proof.blowup_log2() != self.options.blowup_log2 {
            return Err(anyhow!(
                "blowup factor mismatch (got {}, want {})",
                1usize << inner_proof.blowup_log2(),
                1usize << self.options.blowup_log2
            ));
        }

        let num_gate_selectors = self
            .gates
            .iter()
            .map(|(_, gate_instances)| {
                gate_instances
                    .iter()
                    .map(|instance| instance.selector_index)
            })
            .flatten()
            .collect::<BTreeSet<usize>>()
            .len();
        let num_sigma_polynomials = self.num_columns;
        let num_witness_columns = self.num_columns;
        let num_permutation_accumulator_polynomial = 1usize;
        let num_quotient_chunks = self.get_num_quotient_chunks();
        let expected_polynomials = num_gate_selectors
            + num_sigma_polynomials
            + num_witness_columns
            + num_permutation_accumulator_polynomial
            + num_quotient_chunks;

        if inner_proof.num_polys() != expected_polynomials {
            return Err(anyhow!(
                "incorrect number of committed polynomials (got {}, want {})",
                inner_proof.num_polys(),
                expected_polynomials,
            ));
        }

        let omega = Polynomial::<G>::domain_element2(1, self.degree_bound);
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
        for &row in &self.public_rows {
            let z = omega.pow_small(row);
            if !points.contains_key(&z) {
                return Err(anyhow!(
                    "the proof doesn't have an opening for public row {row}"
                ));
            }
        }

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
                            let offset = num_gate_selectors + num_sigma_polynomials;
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
            let offset = num_gate_selectors + num_sigma_polynomials + num_witness_columns;
            (points[&xi][offset], points[&(xi * omega)][offset])
        };

        let beta = H::challenge(
            *DST_BETA,
            &[commitment.transcript_hash(COMMIT_INDEX_WITNESS + 1)],
        );
        let gamma = H::challenge(
            *DST_GAMMA,
            &[commitment.transcript_hash(COMMIT_INDEX_WITNESS + 1)],
        );

        let (permutation_numerator, permutation_denominator) = {
            let mut numerator = G::ONE;
            let mut denominator = G::ONE;
            let mut generator_power = G::ONE;
            let offset = num_gate_selectors + num_sigma_polynomials;
            for column_index in 0..self.num_columns {
                let variable = points[&xi][offset + column_index];
                let sigma = sigma[column_index];
                numerator *= variable + beta * generator_power * xi + gamma;
                denominator *= variable + beta * sigma + gamma;
                generator_power *= G::MULTIPLICATIVE_GENERATOR;
            }
            (numerator, denominator)
        };

        let quotient: G = {
            let offset = num_gate_selectors
                + num_sigma_polynomials
                + num_witness_columns
                + num_permutation_accumulator_polynomial;
            (0..num_quotient_chunks)
                .map(|i| points[&xi][offset + i] * xi.pow_small(i * self.degree_bound))
                .sum()
        };
        let zero = xi.pow_small(self.degree_bound) - G::ONE;

        let alpha = H::challenge(
            *DST_ALPHA,
            &[commitment.transcript_hash(COMMIT_INDEX_PERMUTATION_ARGUMENT + 1)],
        );

        let permutation_recurrence_constraint = shifted_permutation_accumulator
            * permutation_denominator
            - permutation_accumulator * permutation_numerator;
        let permutation_fixpoint_constraint =
            (permutation_accumulator - G::ONE) * Self::lagrange0(xi, self.degree_bound);

        let full_constraint = gate_constraint
            + alpha * permutation_fixpoint_constraint
            + alpha.square() * permutation_recurrence_constraint;
        if full_constraint != quotient * zero {
            return Err(anyhow!("constraint violation"));
        }

        Ok(self
            .public_rows
            .iter()
            .map(|&row| {
                let offset = num_gate_selectors + num_sigma_polynomials;
                (0..self.num_columns).into_iter().map(move |column| {
                    let x = omega.pow_small_vartime(row);
                    (Cell::new(row, column), points[&x][offset + column])
                })
            })
            .flatten()
            .collect())
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
    fn test_vitalik_circuit_impl<F: Field, G: Field256 + From<F>, H: Hasher<G>>(
        canonicalize_constraints: bool,
        blowup_log2: usize,
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
        builder.declare_public_rows([2]);
        let circuit = builder.build(CompilationOptions {
            canonicalize_constraints,
        })?;
        assert_eq!(circuit.num_rows(), 3);
        assert_eq!(circuit.degree_bound(), 8);
        assert_eq!(circuit.num_columns(), 3);
        assert_eq!(circuit.num_gates(), 3);
        let mut witness = circuit.make_witness();
        assert_eq!(witness.num_rows(), 3);
        assert_eq!(witness.degree_bound(), 8);
        assert_eq!(witness.num_columns(), 3);
        witness.set(cell(0, 0), 3u8.into());
        witness.set(cell(0, 1), 9u8.into());
        witness.set(cell(1, 0), 3u8.into());
        witness.set(cell(1, 1), 9u8.into());
        witness.set(cell(1, 2), 35u8.into());
        witness.set(cell(2, 0), 3u8.into());
        witness.set(cell(2, 1), 35u8.into());
        assert!(circuit.check_witness(&witness).is_ok());
        let options = ProvingOptions { blowup_log2 };
        let proof = circuit.prove::<H>(witness, options.clone())?;
        assert_eq!(proof.degree_bound(), 8);
        assert_eq!(proof.blowup_log2(), blowup_log2);
        assert_eq!(proof.extended_domain_size(), 8 << blowup_log2);
        assert_eq!(proof.num_polys(), 13);
        let circuit = circuit.to_compressed(options);
        assert_eq!(circuit.commitment(), commitment);
        let public_inputs = circuit.verify(&proof)?;
        assert_eq!(public_inputs[&cell(2, 0)], 3u8.into());
        assert_eq!(public_inputs[&cell(2, 1)], 35u8.into());
        Ok(())
    }

    #[test]
    fn test_vitalik_circuit_bluesky_sha2_blowup_2() {
        let c = parse_hash("0x2732a0cce5e03109e372c39515b4e3cc7aae87adfde949418c7f5918e4cea1f9");
        assert!(test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(false, 1, c).is_ok());
        assert!(test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(true, 1, c).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_goldilocks_sha2_blowup_2() {
        let c = parse_hash("0x2732a0cce5e03109e372c39515b4e3cc7aae87adfde949418c7f5918e4cea1f9");
        assert!(test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(false, 1, c).is_ok());
        assert!(test_vitalik_circuit_impl::<GL, GL4, Sha2Hash<GL4>>(true, 1, c).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_keccak256_blowup_2() {
        let c = parse_hash("0xa4e2f7c1507afba18f2996fd89dd570f1bd466fdf14c8bc0d65a015c88aa8e20");
        assert!(test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(false, 1, c).is_ok());
        assert!(test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(true, 1, c).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_sha2_blowup_4() {
        let c = parse_hash("0xd81086a421590d6b5517c86495fcc3f52c983ff49e9d7e9cbc034fdb7cb77782");
        assert!(test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(false, 2, c).is_ok());
        assert!(test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(true, 2, c).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_keccak256_blowup_4() {
        let c = parse_hash("0x44197f3984298e3c6d20bc0a5be4ff0dca39b54f0c601f67367c3f72cfb1bb93");
        assert!(test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(false, 2, c).is_ok());
        assert!(test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(true, 2, c).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_sha2_blowup_8() {
        let c = parse_hash("0xdb14b315aa52e49402a85544104520629b245a2fcecf33b15eebb39e0c711aa5");
        assert!(test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(false, 3, c).is_ok());
        assert!(test_vitalik_circuit_impl::<BS, BS, Sha2Hash<BS>>(true, 3, c).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_keccak256_blowup_8() {
        let c = parse_hash("0x4fa1ea6a1b435bb0309c7432be6087bd326962a13709669be5695ddf6049fc22");
        assert!(test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(false, 3, c).is_ok());
        assert!(test_vitalik_circuit_impl::<BS, BS, Keccak256Hash<BS>>(true, 3, c).is_ok());
    }

    /// A slight variation of Vitalik's circuit. This one proves knowledge of three numbers x, y,
    /// and z such that x^3 + xy + 5 = z. Valid combinations are (3, 4, 44) and (4, 3, 81).
    fn test_vitalik_circuit_variation_impl<F: Field, G: Field256 + From<F>, H: Hasher<G>>(
        canonicalize_constraints: bool,
        blowup_log2: usize,
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
        builder.declare_public_rows([3]);
        let circuit = builder.build(CompilationOptions {
            canonicalize_constraints,
        })?;
        assert_eq!(circuit.num_rows(), 4);
        assert_eq!(circuit.degree_bound(), 8);
        assert_eq!(circuit.num_columns(), 4);
        assert_eq!(circuit.num_gates(), 3);
        let mut witness = circuit.make_witness();
        assert_eq!(witness.num_rows(), 4);
        assert_eq!(witness.degree_bound(), 8);
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
        let options = ProvingOptions { blowup_log2 };
        let proof = circuit.prove::<H>(witness, options.clone())?;
        assert_eq!(proof.degree_bound(), 8);
        assert_eq!(proof.blowup_log2(), blowup_log2);
        assert_eq!(proof.extended_domain_size(), 8 << blowup_log2);
        assert_eq!(proof.num_polys(), 16);
        let circuit = circuit.to_compressed(options);
        assert_eq!(circuit.commitment(), commitment);
        let public_inputs = circuit.verify(&proof)?;
        assert_eq!(public_inputs[&x_out], 3u8.into());
        assert_eq!(public_inputs[&y_out], 4u8.into());
        assert_eq!(public_inputs[&result_out], 44u8.into());
        Ok(())
    }

    #[test]
    fn test_vitalik_circuit_variation_blowup_2() {
        let c = parse_hash("0x6f793d1bfd1349adb0981b299dae730991836d950d53d4f2df08b649db552d29");
        assert!(test_vitalik_circuit_variation_impl::<Sha2Hash<Scalar>>(false, 1, c).is_ok());
        assert!(test_vitalik_circuit_variation_impl::<Sha2Hash<Scalar>>(true, 1, c).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_variation_blowup_4() {
        let c = parse_hash("0xb50cc61c1a3f557f97d42bb6233ba4d133338bf86577615e4cbae10bbb90ccb6");
        assert!(test_vitalik_circuit_variation_impl::<Sha2Hash<Scalar>>(false, 2, c).is_ok());
        assert!(test_vitalik_circuit_variation_impl::<Sha2Hash<Scalar>>(true, 2, c).is_ok());
    }

    fn build_vitalik_circuit() -> Circuit {
        let mut builder = CircuitBuilder::default();
        builder.add_gate(0, (var(0) ^ 2) - var(1));
        builder.connect(cell(0, 0).into(), cell(1, 0).into());
        builder.connect(cell(0, 1).into(), cell(1, 1).into());
        builder.add_gate(1, var(0) * var(1) + var(0) + 5 - var(2));
        builder.connect(cell(0, 0).into(), cell(2, 0).into());
        builder.connect(cell(1, 2).into(), cell(2, 1).into());
        builder.add_gate(2, Constraint::nop());
        builder.declare_public_rows([2]);
        builder
            .build(CompilationOptions {
                canonicalize_constraints: false,
            })
            .unwrap()
    }

    #[test]
    fn test_check_witness_detects_wrong_number_of_rows() {
        let circuit = build_vitalik_circuit();
        let witness = Witness::new(4, 3, [0, 1]);
        let error = circuit.check_witness(&witness).unwrap_err();
        assert!(error.to_string().contains("wrong number of rows"));
    }

    #[test]
    fn test_check_witness_detects_wrong_number_of_columns() {
        let circuit = build_vitalik_circuit();
        let witness = Witness::new(3, 4, [0, 1]);
        let error = circuit.check_witness(&witness).unwrap_err();
        assert!(error.to_string().contains("wrong number of columns"));
    }

    #[test]
    fn test_check_witness_detects_wrong_degree_bound() {
        let circuit = build_vitalik_circuit();
        let witness = Witness::new(3, 3, [-2, -1, 0, 1, 2]);
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
}
