use crate::expr::{Constraint, Variable};
use crate::utils::padded_circuit_size;
use crate::witness::{Cell, Partitioner, Witness, cell};
use anyhow::Result;
use starkom_bluesky::Scalar;
use starkom_pcs::{self as pcs, hash::Hash};
use starkom_poly;
use std::collections::{BTreeMap, BTreeSet};

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Default blowup factor (16) in logarithmic form.
///
/// Used with the underlying PCS to compute low-degree extensions.
pub const OPTIONS_DEFAULT_BLOWUP_LOG2: usize = 4;

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

pub trait CircuitView {
    /// Returns the number of columns included in the view.
    ///
    /// If this is the root view, that is the raw [`CircuitBuilder`] instance, the width is
    /// unbounded and `None` is returned.
    fn width(&self) -> Option<usize>;

    /// INTERNAL ONLY. Not for public usage.
    fn add_gate_internal(&mut self, row: usize, column: usize, constraint: Constraint);

    /// Adds a gate to the circuit.
    fn add_gate(&mut self, row: usize, constraint: Constraint) {
        self.add_gate_internal(row, 0, constraint);
    }

    /// "Connects" two circuit [`Cell`]s, meaning they will be constrained to have the same value.
    fn connect(&mut self, wire1: Option<Cell>, wire2: Option<Cell>);

    /// Adds a gate with `N` inputs and `M` outputs.
    ///
    /// The provided `constraint` must use exactly `N+M` variables, or the function will panic. The
    /// provided input cells are wrapped in `Option`s because `None` means the corresponding input
    /// must remain unconstrained.
    ///
    /// The first `N` variables used in the constraint (those with the lowest column numbers) will
    /// be automatically connected to the specified `inputs` unless they're unconstrained / None,
    /// while the last `M` variables (those with the highest column numbers) will be returned as
    /// outputs.
    fn auto_gate<const N: usize, const M: usize>(
        &mut self,
        constraint: Constraint,
        inputs: [Option<Cell>; N],
    ) -> [Cell; M];

    fn add_nop_gate<const N: usize, const M: usize>(
        &mut self,
        inputs: [Option<Cell>; N],
    ) -> [Cell; M] {
        self.auto_gate(Constraint::nop(), inputs)
    }

    /// Spawns a child `CircuitView` at the given coordinates.
    fn spawn(&mut self, row_offset: usize, column_offset: usize, width: usize) -> impl CircuitView;
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

    /// Compiles the circuit built so far into a [`Circuit`] object.
    pub fn build(mut self) -> Result<Circuit> {
        let (degree_bound, _) = padded_circuit_size(
            self.num_rows,
            self.gates.iter().flat_map(|(constraint, _)| {
                constraint
                    .get_free_variables()
                    .iter()
                    .map(Variable::rotation)
                    .collect::<BTreeSet<isize>>()
            }),
        );

        // TODO
        todo!()
    }
}

impl CircuitView for CircuitBuilder {
    fn width(&self) -> Option<usize> {
        None
    }

    fn add_gate_internal(&mut self, row: usize, column: usize, constraint: Constraint) {
        let root_cell = cell(row, column);
        {
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
        match self.gates.get_mut(&constraint) {
            Some(root_cells) => {
                root_cells.push(root_cell);
            }
            None => {
                self.gates.insert(constraint, vec![root_cell]);
            }
        }
    }

    fn connect(&mut self, cell1: Option<Cell>, cell2: Option<Cell>) {
        match (cell1, cell2) {
            (Some(cell1), Some(cell2)) => {
                self.partitioner.connect(cell1, cell2);
            }
            _ => {}
        }
    }

    fn auto_gate<const N: usize, const M: usize>(
        &mut self,
        constraint: Constraint,
        inputs: [Option<Cell>; N],
    ) -> [Cell; M] {
        let variables: Vec<Variable> = constraint.get_free_variables().into_iter().collect();
        assert_eq!(variables.len(), N + M);

        let root_cell = cell(self.row_counter, 0);
        self.row_counter += 1;

        for i in 0..N {
            if let Some(input) = inputs[i] {
                self.connect(Some(input), Some(variables[i].map_to_cell(root_cell)));
            }
        }

        self.add_gate_internal(root_cell.row(), root_cell.column(), constraint);

        std::array::from_fn(|i| variables[N + i].map_to_cell(root_cell))
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

impl<'a> CircuitSectionBuilder<'a> {
    fn map_cell(&self, cell: Cell) -> Cell {
        Cell::new(
            self.row_offset + cell.row(),
            self.column_offset + cell.column(),
        )
    }
}

impl<'a> CircuitView for CircuitSectionBuilder<'a> {
    fn width(&self) -> Option<usize> {
        Some(self.width)
    }

    fn add_gate_internal(&mut self, row: usize, column: usize, constraint: Constraint) {
        self.builder.add_gate_internal(
            self.row_offset + row,
            self.column_offset + column,
            constraint,
        );
    }

    fn auto_gate<const N: usize, const M: usize>(
        &mut self,
        constraint: Constraint,
        inputs: [Option<Cell>; N],
    ) -> [Cell; M] {
        let variables: Vec<Variable> = constraint.get_free_variables().into_iter().collect();
        assert_eq!(variables.len(), N + M);

        let root_cell = cell(self.row_counter, 0);
        self.row_counter += 1;

        for i in 0..N {
            if let Some(input) = inputs[i] {
                self.connect(Some(input), Some(variables[i].map_to_cell(root_cell)));
            }
        }

        self.add_gate_internal(root_cell.row(), root_cell.column(), constraint);

        std::array::from_fn(|i| variables[N + i].map_to_cell(root_cell))
    }

    fn connect(&mut self, cell1: Option<Cell>, cell2: Option<Cell>) {
        let mapper = |cell| self.map_cell(cell);
        self.builder.connect(cell1.map(mapper), cell2.map(mapper))
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

#[derive(Debug, Clone)]
pub struct Circuit {
    /// The raw number of rows of the circuit.
    ///
    /// Unlike [`Self::degree_bound`], this count doesn't include the blinding rows and is not
    /// padded to the next power of 2.
    num_rows: usize,

    /// Number of witness rows (including the blinding rows) rounded up to the next power of 2.
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

    /// Proves correctness for the given witness, or returns an error in case of a constraint
    /// violation.
    pub fn prove<H: Hash<Scalar>>(
        &self,
        mut witness: Witness,
        options: ProvingOptions,
    ) -> Result<Proof<H>> {
        // TODO
        todo!()
    }

    // TODO
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
