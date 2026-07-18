use crate::expr::{Constraint, Variable};
use crate::utils::{hash_to_scalar, padded_circuit_size};
use crate::witness::{Cell, Partitioner, Witness, WitnessView, cell};
use anyhow::{Result, anyhow};
use starkom_bluesky::Scalar;
use starkom_ff::{Field, PrimeField};
use starkom_pcs::{self as pcs, hash::Hash};
use starkom_poly;
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Default blowup factor (16) in logarithmic form.
///
/// Used with the underlying PCS to compute low-degree extensions.
pub const OPTIONS_DEFAULT_BLOWUP_LOG2: usize = 4;

const COMMIT_INDEX_CIRCUIT: usize = 0;
const COMMIT_INDEX_WITNESS: usize = 1;
const COMMIT_INDEX_PERMUTATION_ARGUMENT: usize = 2;
const COMMIT_INDEX_QUOTIENT: usize = 3;
const NUM_COMMIT_INDICES: usize = 4;

const FIAT_SHAMIR_INDEX_ALPHA: Scalar = Scalar::from_const(0);
const FIAT_SHAMIR_INDEX_BETA: Scalar = Scalar::from_const(1);
const FIAT_SHAMIR_INDEX_GAMMA: Scalar = Scalar::from_const(2);
const FIAT_SHAMIR_INDEX_DELTA: Scalar = Scalar::from_const(3);
const FIAT_SHAMIR_INDEX_XI: Scalar = Scalar::from_const(4);

/// Domain separator tag used for the main Fiat-Shamir challenge.
static DST: LazyLock<Scalar> = LazyLock::new(|| hash_to_scalar(b"starkom/plonk/challenge"));

/// Builds the set of all rotations used in a circuit.
///
/// NOTE: we're always including 0 and +1 because we need to open xi and xi*omega regardless;
/// they're needed for the final algebraic check and for the shifted permutation argument,
/// respectively.
fn get_rotation_set<'a, I: IntoIterator<Item = &'a Constraint>>(gates: I) -> BTreeSet<isize> {
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
fn quotient_degree_bound<'a, I: Iterator<Item = &'a Constraint>>(
    degree_bound: usize,
    num_columns: usize,
    gate_constraints: I,
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

mod internal {
    use super::*;

    /// Provides access to the internal state of a circuit view.
    ///
    /// This is used to implement the provided methods of the [`CircuitView`] trait.
    pub trait CircuitViewState {
        /// Returns a reference to the [`CircuitBuilder`].
        fn builder(&self) -> &CircuitBuilder;

        /// Returns a mutable reference to the [`CircuitBuilder`].
        fn builder_mut(&mut self) -> &mut CircuitBuilder;

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
            cell(self.row_offset(), self.column_offset())
        }

        /// Returns the next root cell where an [auto gate](`CircuitView::auto_gate`) can be placed,
        /// advancing the internal state to the next row.
        fn step_row(&mut self) -> Cell;
    }
}

pub trait CircuitView: internal::CircuitViewState {
    /// Returns the number of columns included in the view.
    ///
    /// If this is the root view, that is the raw [`CircuitBuilder`] instance, the width is
    /// unbounded and `None` is returned.
    fn width(&self) -> Option<usize>;

    /// Adds a gate to the circuit.
    fn add_gate(&mut self, row: usize, constraint: Constraint) {
        let root_cell = cell(row, 0).remap(self.root_cell());
        self.builder_mut().add_gate_internal(root_cell, constraint);
    }

    /// "Connects" two circuit [`Cell`]s, meaning they will be constrained to have the same value.
    fn connect(&mut self, cell1: Option<Cell>, cell2: Option<Cell>) {
        let root_cell = self.root_cell();
        self.builder_mut().connect_internal(
            cell1.map(|cell| cell.remap(root_cell)),
            cell2.map(|cell| cell.remap(root_cell)),
        );
    }

    /// Adds a gate with `N` inputs and `M` outputs.
    ///
    /// The provided `constraint` must use exactly `N+M` variables, or the function will panic. The
    /// provided input cells are wrapped in `Option`s because `None` means the corresponding input
    /// is unconstrained.
    ///
    /// The first `N` variables used in the constraint (those with the lowest column numbers) will
    /// be automatically connected to the specified `inputs` unless they're unconstrained / None,
    /// while the last `M` variables (those with the highest column numbers) will be returned as
    /// outputs.
    ///
    /// NOTE: variables with the same column number but different rotations are considered different
    /// variables for the purpose of counting against `N` and `M`. For example, the constraint
    /// `var(0,+1) == var(0,-1) + var(0)` uses three variables, not one. Variables with the same
    /// column number are ordered by rotation, so for example if you set `N=2` and `M=1` the above
    /// constraint would associate the 2 inputs to `var(0,-1)` and `var(0)`, and the output to
    /// `var(0,+1)`.
    fn auto_gate<const N: usize, const M: usize>(
        &mut self,
        constraint: Constraint,
        inputs: [Option<Cell>; N],
    ) -> [Cell; M] {
        let variables: Vec<Variable> = constraint.get_free_variables().into_iter().collect();
        assert_eq!(variables.len(), N + M);

        let root_cell = self.step_row();

        for i in 0..N {
            if let Some(input) = inputs[i] {
                self.builder_mut()
                    .connect_internal(Some(input), Some(variables[i].map_to_cell(root_cell)));
            }
        }

        self.builder_mut().add_gate_internal(root_cell, constraint);

        std::array::from_fn(|i| variables[N + i].map_to_cell(root_cell))
    }

    /// Adds a gate with `N` terminations.
    ///
    /// Unlike [`Self::auto_gate`] this method doesn't make a distinction between input and output
    /// terminations; it simply enforces a polynomial relation among `N` terminations.
    ///
    /// The provided `constraint` must use exactly `N` variables, or the function will panic. The
    /// provided input cells are wrapped in `Option`s because `None` means the corresponding input
    /// is unconstrained.
    ///
    /// The `N` variables used in the constraint will be automatically connected to the specified
    /// `inputs` unless they're unconstrained / None.
    ///
    /// NOTE: variables with the same column number but different rotations are considered different
    /// variables for the purpose of counting against `N`. For example, the constraint
    /// `var(0,+1) == var(0,-1) + var(0)` uses three variables, not one. Variables with the same
    /// column number are ordered by rotation, so for example the above constraint would associate
    /// the 3 inputs to `var(0,-1)`, `var(0)`, and `var(0,+1)`, respectively.
    fn auto_constraint<const N: usize>(
        &mut self,
        constraint: Constraint,
        inputs: [Option<Cell>; N],
    ) -> [Cell; N] {
        let variables: Vec<Variable> = constraint.get_free_variables().into_iter().collect();
        assert_eq!(variables.len(), N);

        let root_cell = self.step_row();

        for i in 0..N {
            if let Some(input) = inputs[i] {
                self.connect(Some(input), Some(variables[i].map_to_cell(root_cell)));
            }
        }

        self.builder_mut().add_gate_internal(root_cell, constraint);

        std::array::from_fn(|i| variables[i].map_to_cell(root_cell))
    }

    /// Adds a NOP gate with `N` terminations; the constraint of the gate is `0 == 0`.
    ///
    /// This is conceptually equivalent to calling:
    ///
    /// ```ignore
    /// witness.auto_constraint::<N>(Constraint::nop(), inputs);
    /// ```
    ///
    /// except that the above call wouldn't work because the nop constraint uses 0 variables, so it
    /// would panic because `0 != N`.
    fn add_nop_gate<const N: usize>(&mut self, inputs: [Option<Cell>; N]) -> [Cell; N] {
        let root_cell = self.step_row();
        let outputs = std::array::from_fn(|i| cell(0, i).remap(root_cell));

        for i in 0..N {
            if let Some(input) = inputs[i] {
                self.connect(Some(input), Some(outputs[i]));
            }
        }

        self.builder_mut()
            .add_gate_internal(root_cell, Constraint::nop());

        outputs
    }

    /// Spawns a child `CircuitView` at the given coordinates.
    fn spawn(&mut self, row_offset: usize, column_offset: usize, width: usize) -> impl CircuitView;
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
pub struct CircuitBuilder {
    /// Current number of rows in the circuit.
    num_rows: usize,

    /// Current number of columns in the circuit.
    num_columns: usize,

    /// Used by [`Self::auto_gate`] to keep track of the current row;
    row_counter: usize,

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
    gates: BTreeMap<Constraint, Vec<Cell>>,

    /// Cell partitioning inferred from the connections made with [`Self::connect`].
    partitioner: Partitioner,

    /// List of rows that are revealed in the proofs.
    public_rows: BTreeSet<usize>,
}

impl CircuitBuilder {
    fn add_gate_internal(&mut self, root_cell: Cell, constraint: Constraint) {
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

    fn make_selector(degree_bound: usize, activation_row_set: BTreeSet<usize>) -> Polynomial {
        let mut selector_values = vec![Scalar::ZERO; degree_bound];
        for row in activation_row_set {
            selector_values[row] = Scalar::ONE;
        }
        Polynomial::encode2(selector_values)
    }

    fn build_gates_and_selectors(
        &self,
        degree_bound: usize,
    ) -> (
        BTreeMap<Constraint, BTreeSet<GateInstance>>,
        Vec<Polynomial>,
    ) {
        // Keys are (constraint, column_index) pairs; values are activation row sets.
        let mut row_set_map: BTreeMap<(Constraint, usize), BTreeSet<usize>> = BTreeMap::default();
        for (constraint, root_cells) in &self.gates {
            for root_cell in root_cells.as_slice() {
                let key = (constraint.clone(), root_cell.column());
                let row = root_cell.row();
                row_set_map.entry(key).or_default().insert(row);
            }
        }

        // Roughly the inverse of `row_set_map`: keys are activation row sets, values are the list
        // of (constraint, column_index) instances that activate at those rows.
        let mut gates_by_row_set: BTreeMap<BTreeSet<usize>, Vec<(Constraint, usize)>> =
            BTreeMap::default();
        for ((constraint, column_index), row_set) in row_set_map {
            gates_by_row_set
                .entry(row_set)
                .or_default()
                .push((constraint, column_index));
        }

        let mut gates: BTreeMap<Constraint, BTreeSet<GateInstance>> = BTreeMap::default();
        let mut selectors: Vec<Polynomial> = vec![];

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
    pub fn build(mut self, options: CompilationOptions) -> Result<Circuit> {
        if options.canonicalize_constraints {
            let mut old_gates: BTreeMap<Constraint, Vec<Cell>> = BTreeMap::default();
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

        let sigma_values: Vec<Vec<Scalar>> = {
            let mut sigma = vec![Scalar::ZERO; degree_bound * self.num_columns];
            let omega = Polynomial::domain_element2(1, degree_bound);
            let mut k = Scalar::ONE;
            for i in 0..self.num_columns {
                let offset = i * degree_bound;
                sigma[offset] = k;
                for j in 1..degree_bound {
                    sigma[offset + j] = sigma[offset + j - 1] * omega;
                }
                k *= Scalar::MULTIPLICATIVE_GENERATOR;
            }
            for node in self.partitioner.iter_nodes() {
                let indices: Vec<usize> = node
                    .iter()
                    .map(|cell| cell.column() * degree_bound + cell.row())
                    .collect();
                let mut permuted: Vec<Scalar> = indices.iter().map(|&i| sigma[i]).collect();
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
        })
    }
}

impl internal::CircuitViewState for CircuitBuilder {
    fn builder(&self) -> &CircuitBuilder {
        self
    }

    fn builder_mut(&mut self) -> &mut CircuitBuilder {
        self
    }

    fn row_offset(&self) -> usize {
        0
    }

    fn column_offset(&self) -> usize {
        0
    }

    fn step_row(&mut self) -> Cell {
        let root_cell = cell(self.row_counter, 0);
        self.row_counter += 1;
        root_cell
    }
}

impl CircuitView for CircuitBuilder {
    fn width(&self) -> Option<usize> {
        None
    }

    fn spawn(&mut self, row_offset: usize, column_offset: usize, width: usize) -> impl CircuitView {
        CircuitSectionBuilder::new(self, row_offset, column_offset, width)
    }
}

/// Implements [`CircuitView`] for a sub-section of the circuit.
#[derive(Debug)]
struct CircuitSectionBuilder<'a> {
    /// Reference to the parent [`CircuitBuilder`].
    builder: &'a mut CircuitBuilder,

    /// Row offset of the sub-section.
    row_offset: usize,

    /// Column offset of the sub-section.
    column_offset: usize,

    /// Width (number of columns) of the sub-section.
    width: usize,

    /// Row counter for [`Self::auto_gate`].
    row_counter: usize,
}

impl<'a> CircuitSectionBuilder<'a> {
    fn new(
        builder: &'a mut CircuitBuilder,
        row_offset: usize,
        column_offset: usize,
        width: usize,
    ) -> Self {
        Self {
            builder,
            row_offset,
            column_offset,
            width,
            row_counter: 0,
        }
    }
}

impl<'a> internal::CircuitViewState for CircuitSectionBuilder<'a> {
    fn builder(&self) -> &CircuitBuilder {
        self.builder
    }

    fn builder_mut(&mut self) -> &mut CircuitBuilder {
        self.builder
    }

    fn row_offset(&self) -> usize {
        self.row_offset
    }

    fn column_offset(&self) -> usize {
        self.column_offset
    }

    fn step_row(&mut self) -> Cell {
        let root_cell = cell(self.row_counter, 0).remap(self.root_cell());
        self.row_counter += 1;
        root_cell
    }
}

impl<'a> CircuitView for CircuitSectionBuilder<'a> {
    fn width(&self) -> Option<usize> {
        Some(self.width)
    }

    fn spawn(&mut self, row_offset: usize, column_offset: usize, width: usize) -> impl CircuitView {
        CircuitSectionBuilder::new(
            self.builder,
            self.row_offset + row_offset,
            self.column_offset + column_offset,
            width,
        )
    }
}

impl<'a> Drop for CircuitSectionBuilder<'a> {
    fn drop(&mut self) {
        self.builder.row_counter =
            std::cmp::max(self.builder.row_counter, self.row_offset + self.row_counter);
    }
}

/// A PLONK proof.
///
/// The API in the implementation mostly mirrors that of the underlying PCS proof.
#[derive(Debug, Clone)]
pub struct Proof<H: Hash<Scalar>> {
    commitment: pcs::Commitment<H>,
    inner_proof: pcs::Proof<H>,
}

impl<H: Hash<Scalar>> Proof<H> {
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

#[derive(Debug, Clone)]
pub struct Circuit {
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
    selectors: Vec<Polynomial>,

    /// Gates used in the circuit: the first component of each pair is the gate constraint and the
    /// second component is the set of instances of that gate across the circuit.
    gates: BTreeMap<Constraint, BTreeSet<GateInstance>>,

    /// Sigma polynomials of the permutation argument, one for every witness column.
    sigma: Vec<Polynomial>,

    /// The [sigma polynomials](`Self::sigma`) expressed on the value domain.
    ///
    /// The layout is analogous to [`Self::sigma`] itself: the values are indexed column-first.
    sigma_values: Vec<Vec<Scalar>>,

    /// List of gates that are revealed in the proofs. Each element is a row index.
    public_rows: BTreeSet<usize>,
}

impl Circuit {
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

    /// Makes an empty [`Witness`] objects suitable for use with this circuit.
    pub fn make_witness(&self) -> Witness {
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

    /// Builds the three polynomials used in the permutation argument. The components of the
    /// returned tuple are, respectively: the coordinate pair accumulator, the fixpoint constraint,
    /// and the recurrence constraint.
    fn build_permutation_argument(
        &self,
        witness: &Witness,
        columns: &[Polynomial],
        beta: Scalar,
        gamma: Scalar,
    ) -> Result<(Polynomial, Polynomial, Polynomial)> {
        let omega = Polynomial::domain_element2(1, self.degree_bound);

        let accumulator = {
            let mut accumulator = vec![Scalar::ZERO; self.degree_bound + 1];

            accumulator[0] = Scalar::ONE;
            let mut omega_pow = Scalar::ONE;
            for i in 0..self.degree_bound {
                let mut generator_pow = Scalar::ONE;
                accumulator[i + 1] = accumulator[i];
                for j in 0..self.num_columns {
                    let witness_value = witness.get(cell(i, j));
                    accumulator[i + 1] *= witness_value + beta * generator_pow * omega_pow + gamma;
                    accumulator[i + 1] *=
                        (witness_value + beta * self.sigma_values[j][i] + gamma).invert_unwrap();
                    generator_pow *= Scalar::MULTIPLICATIVE_GENERATOR;
                }
                omega_pow *= omega;
            }

            if accumulator.pop().unwrap() != Scalar::ONE {
                return Err(anyhow!("permutation accumulator wraparound check failed"));
            }

            Polynomial::encode2(accumulator)
        };

        let shifted = accumulator.clone().shift_domain_by(omega);

        let recurrence_constraint = {
            let mut lhs = shifted;
            for (column, sigma) in columns.iter().zip(self.sigma.iter()) {
                lhs *= column.clone() + sigma.clone() * beta + gamma;
            }
            let mut rhs = accumulator.clone();
            let mut pow = Scalar::ONE;
            for column in columns {
                rhs *= column.clone() + Polynomial::with_coefficients(vec![gamma, beta * pow]);
                pow *= Scalar::MULTIPLICATIVE_GENERATOR;
            }
            lhs - rhs
        };

        let fixpoint_constraint =
            (accumulator.clone() - Scalar::ONE) * Polynomial::lagrange0(self.degree_bound).clone();

        Ok((accumulator, fixpoint_constraint, recurrence_constraint))
    }

    /// Splits the quotient polynomial in chunks so that it can be batch-committed even though its
    /// degree is much higher than the bound configured in the underlying PCS.
    fn split_quotient(&self, quotient: Polynomial) -> Vec<Polynomial> {
        let degree_bound = quotient_degree_bound(
            self.degree_bound,
            self.num_columns,
            self.gates.iter().map(|(constraint, _)| constraint),
        );
        let mut coefficients = quotient.take();
        assert!(coefficients.len() <= degree_bound);
        coefficients.resize(degree_bound, Scalar::ZERO);
        coefficients
            .chunks(self.degree_bound)
            .map(|coefficients| Polynomial::with_coefficients(coefficients.to_vec()))
            .collect()
    }

    /// Proves correctness of the given witness, or returns an error in case of a constraint
    /// violation.
    pub fn prove<H: Hash<Scalar>>(
        &self,
        mut witness: Witness,
        options: ProvingOptions,
    ) -> Result<Proof<H>> {
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

        let circuit_polynomials = self
            .selectors
            .iter()
            .cloned()
            .chain(self.sigma.iter().cloned())
            .collect();

        let mut committer =
            pcs::Committer::<H>::new(self.degree_bound, options.blowup_log2, circuit_polynomials);

        let columns = witness.clone().encode();
        committer.add_batch(columns.clone());

        let omega = Polynomial::domain_element2(1, self.degree_bound);

        let gate_constraint = {
            let delta = H::hash_two(*DST, committer.transcript_hash(), FIAT_SHAMIR_INDEX_DELTA);
            let mut gate_constraint = Polynomial::default();
            let mut pow = Scalar::ONE;
            for (constraint, instances) in &self.gates {
                for instance in instances {
                    let constraint = constraint.clone().remap_variables(instance.column_index);
                    let selector = self.selectors[instance.selector_index].clone();
                    gate_constraint +=
                        selector * constraint.compose(omega, columns.as_slice()) * pow;
                    pow *= delta;
                }
            }
            gate_constraint
        };

        let (
            permutation_accumulator,
            permutation_fixpoint_constraint,
            permutation_recurrence_constraint,
        ) = {
            let beta = H::hash_two(*DST, committer.transcript_hash(), FIAT_SHAMIR_INDEX_BETA);
            let gamma = H::hash_two(*DST, committer.transcript_hash(), FIAT_SHAMIR_INDEX_GAMMA);
            self.build_permutation_argument(&witness, columns.as_slice(), beta, gamma)?
        };
        committer.add_batch(vec![permutation_accumulator]);

        let alpha = H::hash_two(*DST, committer.transcript_hash(), FIAT_SHAMIR_INDEX_ALPHA);

        let quotient = (gate_constraint
            + permutation_fixpoint_constraint * alpha
            + permutation_recurrence_constraint * alpha.square())
        .divide_by_zero(self.degree_bound)?;
        committer.add_batch(self.split_quotient(quotient));

        let xi = H::hash_two(*DST, committer.transcript_hash(), FIAT_SHAMIR_INDEX_XI);

        let omega_inv = omega.invert_unwrap();
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

    pub fn to_compressed<H: Hash<Scalar>>(self, options: ProvingOptions) -> CompressedCircuit<H> {
        let committer = pcs::Committer::<H>::new(
            self.degree_bound,
            options.blowup_log2,
            self.selectors
                .into_iter()
                .chain(self.sigma.into_iter())
                .collect(),
        );
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
            _data: Default::default(),
        }
    }

    pub fn as_compressed<H: Hash<Scalar>>(&self, options: ProvingOptions) -> CompressedCircuit<H> {
        let committer = pcs::Committer::<H>::new(
            self.degree_bound,
            options.blowup_log2,
            self.selectors
                .iter()
                .cloned()
                .chain(self.sigma.iter().cloned())
                .collect(),
        );
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

    pub fn verify<H: Hash<Scalar>>(
        &self,
        proof: &Proof<H>,
        options: ProvingOptions,
    ) -> Result<BTreeMap<Cell, Scalar>> {
        self.as_compressed::<H>(options).verify(proof)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedCircuit<H: Hash<Scalar>> {
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
    gates: Vec<(Constraint, Vec<GateInstance>)>,

    /// List of rows that are revealed in the proofs.
    public_rows: BTreeSet<usize>,

    /// Merkle root of the circuit selectors.
    circuit_commitment: Scalar,

    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> CompressedCircuit<H> {
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
    pub fn verify(&self, proof: &Proof<H>) -> Result<BTreeMap<Cell, Scalar>> {
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

        let omega = Polynomial::domain_element2(1, self.degree_bound);
        let omega_inv = omega.invert_vartime().unwrap();

        let xi = H::hash_two(
            *DST,
            commitment.transcript_hash(COMMIT_INDEX_QUOTIENT + 1),
            FIAT_SHAMIR_INDEX_XI,
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

        let sigma: Vec<Scalar> = {
            let offset = num_gate_selectors;
            (0..self.num_columns)
                .map(|i| points[&xi][offset + i])
                .collect()
        };

        let gate_constraint: Scalar = {
            let selectors: Vec<Scalar> = (0..num_gate_selectors).map(|i| points[&xi][i]).collect();
            let delta = H::hash_two(
                *DST,
                commitment.transcript_hash(COMMIT_INDEX_WITNESS + 1),
                FIAT_SHAMIR_INDEX_DELTA,
            );
            let mut result = Scalar::ZERO;
            let mut pow = Scalar::ONE;
            for (constraint, gate_instances) in &self.gates {
                for instance in gate_instances {
                    let constraint = constraint.clone().remap_variables(instance.column_index);
                    let substitution: BTreeMap<Variable, Scalar> = constraint
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
                        * pow;
                    pow *= delta;
                }
            }
            result
        };

        let (permutation_accumulator, shifted_permutation_accumulator) = {
            let offset = num_gate_selectors + num_sigma_polynomials + num_witness_columns;
            (points[&xi][offset], points[&(xi * omega)][offset])
        };

        let beta = H::hash_two(
            *DST,
            commitment.transcript_hash(COMMIT_INDEX_WITNESS + 1),
            FIAT_SHAMIR_INDEX_BETA,
        );
        let gamma = H::hash_two(
            *DST,
            commitment.transcript_hash(COMMIT_INDEX_WITNESS + 1),
            FIAT_SHAMIR_INDEX_GAMMA,
        );

        let (permutation_numerator, permutation_denominator) = {
            let mut numerator = Scalar::ONE;
            let mut denominator = Scalar::ONE;
            let mut generator_pow = Scalar::ONE;
            let offset = num_gate_selectors + num_sigma_polynomials;
            for column_index in 0..self.num_columns {
                let variable = points[&xi][offset + column_index];
                let sigma = sigma[column_index];
                numerator *= variable + beta * generator_pow * xi + gamma;
                denominator *= variable + beta * sigma + gamma;
                generator_pow *= Scalar::MULTIPLICATIVE_GENERATOR;
            }
            (numerator, denominator)
        };

        let quotient: Scalar = {
            let offset = num_gate_selectors
                + num_sigma_polynomials
                + num_witness_columns
                + num_permutation_accumulator_polynomial;
            (0..num_quotient_chunks)
                .map(|i| points[&xi][offset + i] * xi.pow_small(i * self.degree_bound))
                .sum()
        };
        let zero = xi.pow_small(self.degree_bound) - Scalar::ONE;

        let alpha = H::hash_two(
            *DST,
            commitment.transcript_hash(COMMIT_INDEX_PERMUTATION_ARGUMENT + 1),
            FIAT_SHAMIR_INDEX_ALPHA,
        );

        let permutation_recurrence_constraint = shifted_permutation_accumulator
            * permutation_denominator
            - permutation_accumulator * permutation_numerator;
        let permutation_fixpoint_constraint = (permutation_accumulator - Scalar::from_const(1))
            * Self::lagrange0(xi, self.degree_bound);

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
                    (cell(row, column), points[&x][offset + column])
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
    use starkom_bluesky::from_const;
    use starkom_pcs::hash::{Poseidon2Hash, Sha2Hash};

    // This function tests the circuit from Vitalik's PLONK tutorial,
    // https://vitalik.eth.limo/general/2019/09/22/plonk.html#how-plonk-works.
    fn test_vitalik_circuit_impl<H: Hash<Scalar>>(
        canonicalize_constraints: bool,
        blowup_log2: usize,
    ) -> Result<()> {
        let mut builder = CircuitBuilder::default();
        let [x, square] = builder.auto_gate((var(0) ^ 2) - var(1), []);
        let [result] = builder.auto_gate(
            var(0) * var(1) + var(0) + 5 - var(2),
            [x.into(), square.into()],
        );
        let [_, result] = builder.add_nop_gate([x.into(), result.into()]);
        builder.declare_public_rows([result.row()]);
        let circuit = builder.build(CompilationOptions {
            canonicalize_constraints,
        })?;
        assert_eq!(circuit.num_rows(), 3);
        assert_eq!(circuit.degree_bound(), 8);
        assert_eq!(circuit.num_columns(), 3);
        let mut witness = circuit.make_witness();
        assert_eq!(witness.num_rows(), 3);
        assert_eq!(witness.degree_bound(), 8);
        assert_eq!(witness.num_columns(), 3);
        let square = witness.auto_set_one(var(1), var(0) ^ 2, [from_const(3)]);
        let result = witness.auto_set_one(var(2), var(0) * var(1) + var(0) + 5, [x, square]);
        let [x, result] = witness.nop([x, result]);
        let options = ProvingOptions { blowup_log2 };
        let proof = circuit.prove::<H>(witness, options.clone())?;
        assert_eq!(proof.degree_bound(), 8);
        assert_eq!(proof.blowup_log2(), blowup_log2);
        assert_eq!(proof.extended_domain_size(), 8 << blowup_log2);
        assert_eq!(proof.num_polys(), 13);
        let public_inputs = circuit.verify(&proof, options)?;
        assert_eq!(public_inputs[&x], from_const(3));
        assert_eq!(public_inputs[&result], from_const(35));
        Ok(())
    }

    #[test]
    fn test_vitalik_circuit_sha2_blowup_2() {
        assert!(test_vitalik_circuit_impl::<Sha2Hash<Scalar>>(false, 1).is_ok());
        assert!(test_vitalik_circuit_impl::<Sha2Hash<Scalar>>(true, 1).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_poseidon2_blowup_2() {
        assert!(test_vitalik_circuit_impl::<Poseidon2Hash<Scalar>>(false, 1).is_ok());
        assert!(test_vitalik_circuit_impl::<Poseidon2Hash<Scalar>>(true, 1).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_sha2_blowup_4() {
        assert!(test_vitalik_circuit_impl::<Sha2Hash<Scalar>>(false, 2).is_ok());
        assert!(test_vitalik_circuit_impl::<Sha2Hash<Scalar>>(true, 2).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_poseidon2_blowup_4() {
        assert!(test_vitalik_circuit_impl::<Poseidon2Hash<Scalar>>(false, 2).is_ok());
        assert!(test_vitalik_circuit_impl::<Poseidon2Hash<Scalar>>(true, 2).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_sha2_blowup_8() {
        assert!(test_vitalik_circuit_impl::<Sha2Hash<Scalar>>(false, 3).is_ok());
        assert!(test_vitalik_circuit_impl::<Sha2Hash<Scalar>>(true, 3).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_poseidon2_blowup_8() {
        assert!(test_vitalik_circuit_impl::<Poseidon2Hash<Scalar>>(false, 3).is_ok());
        assert!(test_vitalik_circuit_impl::<Poseidon2Hash<Scalar>>(true, 3).is_ok());
    }

    // TODO
}
