use crate::expr::{Constraint, Variable};
use crate::utils::padded_circuit_size;
use starkom_bluesky::Scalar;
use starkom_ff::Field;
use std::collections::{BTreeMap, BTreeSet, btree_map};

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// A cell in a circuit or witness, uniquely identified by a row number and a column number.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cell {
    row: usize,
    column: usize,
}

impl Cell {
    /// Creates a new cell.
    pub const fn new(row: usize, column: usize) -> Self {
        Self { row, column }
    }

    /// Returns the row number of the cell.
    pub const fn row(&self) -> usize {
        self.row
    }

    /// Returns the column number of the cell.
    pub const fn column(&self) -> usize {
        self.column
    }

    pub(crate) const fn remap(self, root_cell: Cell) -> Self {
        Self {
            row: root_cell.row + self.row,
            column: root_cell.column + self.column,
        }
    }
}

/// Shorthand for [`Cell::new`].
#[inline]
pub const fn cell(row: usize, column: usize) -> Cell {
    Cell::new(row, column)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CellOrUnconstrained {
    Cell(Cell),
    Unconstrained(Scalar),
}

impl From<Cell> for CellOrUnconstrained {
    fn from(cell: Cell) -> Self {
        CellOrUnconstrained::Cell(cell)
    }
}

impl From<Scalar> for CellOrUnconstrained {
    fn from(value: Scalar) -> Self {
        CellOrUnconstrained::Unconstrained(value)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NodeIterator<'a> {
    inner: btree_map::Iter<'a, usize, BTreeSet<Cell>>,
}

impl<'a> Iterator for NodeIterator<'a> {
    type Item = &'a BTreeSet<Cell>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, node)| node)
    }
}

/// Keeps the cells of a circuit organized in partitions.
///
/// A partition is a set of interconnected witness cells that will be constrained to have the same
/// value. We sometimes use the term "node" to refer to a partition because in circuit theory a node
/// is a set of interconnected wires, equivalent to one of our partitions.
///
/// This data structure allows determining the subsets of the sigma polynomials to permute.
#[derive(Debug, Default, Clone)]
pub(crate) struct Partitioner {
    /// Next available node ID.
    next_id: usize,

    /// Keys are incremental node IDs, values are nodes.
    nodes: BTreeMap<usize, BTreeSet<Cell>>,

    /// Keys are cells, values are the ID of the node that cell belongs to.
    ///
    /// If a cell is not found here it's implied that it belongs to a partition containing only
    /// that cell, i.e. it's unconstrained.
    node_by_cell: BTreeMap<Cell, usize>,
}

impl Partitioner {
    pub(crate) fn connect(&mut self, cell1: Cell, cell2: Cell) {
        if let Some(node_id1) = self.node_by_cell.get(&cell1) {
            if let Some(node_id2) = self.node_by_cell.get(&cell2) {
                if *node_id1 != *node_id2 {
                    let mut node2 = self.nodes.remove(&node_id2).unwrap();
                    let node1 = self.nodes.get_mut(node_id1).unwrap();
                    node1.append(&mut node2);
                    self.node_by_cell.insert(cell2, *node_id1);
                }
            } else {
                let node = self.nodes.get_mut(node_id1).unwrap();
                node.insert(cell2);
                self.node_by_cell.insert(cell2, *node_id1);
            }
        } else {
            if let Some(node_id) = self.node_by_cell.get(&cell2) {
                let node = self.nodes.get_mut(node_id).unwrap();
                node.insert(cell1);
                self.node_by_cell.insert(cell1, *node_id);
            } else {
                let id = self.next_id;
                self.next_id += 1;
                self.nodes.insert(id, BTreeSet::from([cell1, cell2]));
                self.node_by_cell.insert(cell1, id);
                self.node_by_cell.insert(cell2, id);
            }
        }
    }

    pub(crate) fn iter_nodes(&self) -> NodeIterator<'_> {
        NodeIterator {
            inner: self.nodes.iter(),
        }
    }
}

mod internal {
    use super::*;

    /// Provides access to the internal state of a circuit view.
    ///
    /// This is used to implement the provided methods of the [`WitnessView`] trait.
    pub trait WitnessViewState {
        /// Returns a reference to the [`Witness`].
        fn witness(&self) -> &Witness;

        /// Returns a mutable reference to the [`Witness`].
        fn witness_mut(&mut self) -> &mut Witness;

        /// Returns the row offset of the view (0 for the [`Witness`] itself).
        ///
        /// This is always an absolute value even for transitive sub-views. It is not relative to
        /// the parent view.
        fn row_offset(&self) -> usize;

        /// Returns the column offset of the view (0 for the [`Witness`] itself).
        ///
        /// This is always an absolute value even for transitive sub-views. It is not relative to
        /// the parent view.
        fn column_offset(&self) -> usize;

        /// Returns [`Self::row_offset()`] and [`Self::column_offset()`] as a [`Cell`].
        fn root_cell(&self) -> Cell {
            cell(self.row_offset(), self.column_offset())
        }

        /// Returns the next root cell for [setting auto gates](`Witness::auto_set`), advancing the
        /// internal state to the next row.
        fn step_row(&mut self) -> Cell;
    }
}

pub trait WitnessView: internal::WitnessViewState {
    fn width(&self) -> usize;

    /// Reads a witness cell.
    fn get(&self, cell: Cell) -> Scalar {
        self.witness().get_internal(cell)
    }

    /// Updates a witness cell.
    fn set(&mut self, cell: Cell, value: Scalar) {
        self.witness_mut().set_internal(cell, value);
    }

    /// Copies a witness cell to another.
    fn copy(&mut self, src_cell: Cell, dst_cell: Cell) -> Scalar {
        self.witness_mut().copy_internal(src_cell, dst_cell)
    }

    fn auto_set<C: Into<CellOrUnconstrained>, const N: usize, const M: usize>(
        &mut self,
        expressions: &BTreeMap<Variable, Constraint>,
        inputs: [C; N],
    ) -> [Cell; M] {
        assert_eq!(expressions.len(), M);

        let root_cell = self.step_row();

        let variables: BTreeSet<Variable> = expressions
            .iter()
            .map(|(_, constraint)| constraint.get_free_variables())
            .fold(BTreeSet::default(), |mut accumulator, mut variables| {
                accumulator.append(&mut variables);
                accumulator
            });
        assert_eq!(variables.len(), N);

        let substitution: BTreeMap<Variable, Scalar> = variables
            .into_iter()
            .zip(inputs.into_iter())
            .map(|(variable, input)| {
                let dst_cell = variable.map_to_cell(root_cell);
                let value = match input.into() {
                    CellOrUnconstrained::Cell(cell) => self.copy(cell, dst_cell),
                    CellOrUnconstrained::Unconstrained(value) => {
                        self.set(dst_cell, value);
                        value
                    }
                };
                (variable, value)
            })
            .collect();

        expressions
            .iter()
            .map(|(variable, expression)| {
                let cell = variable.map_to_cell(root_cell);
                self.set(cell, expression.evaluate(&substitution));
                cell
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn auto_set_one<C: Into<CellOrUnconstrained>, const N: usize>(
        &mut self,
        output: Constraint,
        expression: Constraint,
        inputs: [C; N],
    ) -> Cell {
        let [result] = self.auto_set(
            &BTreeMap::from([(output.get_variable().unwrap(), expression)]),
            inputs,
        );
        result
    }

    fn nop<C: Copy + Into<CellOrUnconstrained>, const N: usize>(
        &mut self,
        inputs: [C; N],
    ) -> [Cell; N] {
        let root_cell = self.step_row();
        std::array::from_fn(|i| {
            let cell = cell(0, i).remap(root_cell);
            match inputs[i].into() {
                CellOrUnconstrained::Cell(input) => {
                    self.witness_mut().copy_internal(input, cell);
                }
                CellOrUnconstrained::Unconstrained(value) => {
                    self.witness_mut().set_internal(cell, value);
                }
            }
            cell
        })
    }

    fn spawn(&mut self, row_offset: usize, column_offset: usize, width: usize) -> impl WitnessView;
}

#[derive(Debug, Clone)]
pub struct Witness {
    /// The number of witness rows *not* including the blinding rows.
    num_rows: usize,

    /// Number of blinding rows used in the circuit.
    ///
    /// This is calculated by [`padded_circuit_size`] and depends on how many different variable
    /// rotations were used across all constraints.
    num_blinding_rows: usize,

    /// Padded circuit size, including the blinding rows and rounded up to the next power of 2.
    degree_bound: usize,

    /// Used by [`Self::auto_set`] to identify the row to update.
    row_counter: usize,

    /// Witness table cells, indexed column-first.
    ///
    /// The column-first indexing allows quickly interpolating polynomials for the columns.
    data: Vec<Vec<Scalar>>,
}

impl Witness {
    pub(crate) fn new<R: IntoIterator<Item = isize>>(
        num_rows: usize,
        num_columns: usize,
        rotations: R,
    ) -> Self {
        let (degree_bound, num_blinding_rows) = padded_circuit_size(num_rows, rotations);
        Self {
            num_rows,
            num_blinding_rows,
            degree_bound,
            row_counter: 0,
            data: vec![vec![Scalar::ZERO; degree_bound]; num_columns],
        }
    }

    /// Returns the number of witness rows *not* including the blinding rows.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Returns the padded circuit size, including the blinding rows and rounded up to the next
    /// power of 2.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the number of witness columns.
    pub fn num_columns(&self) -> usize {
        self.data.len()
    }

    fn get_internal(&self, cell: Cell) -> Scalar {
        self.data[cell.column()][cell.row()]
    }

    fn set_internal(&mut self, cell: Cell, value: Scalar) {
        self.data[cell.column()][cell.row()] = value;
    }

    fn copy_internal(&mut self, src_cell: Cell, dst_cell: Cell) -> Scalar {
        let value = self.data[src_cell.column()][src_cell.row()];
        self.data[dst_cell.column()][dst_cell.row()] = value;
        value
    }

    /// Fills in the blinding rows of the witness with random values.
    ///
    /// The affected rows are the last [`Self::num_blinding_rows`] before the power-of-two padding.
    pub(crate) fn blind(&mut self) {
        for i in 0..self.num_columns() {
            for j in 0..self.num_blinding_rows {
                self.data[i][self.num_rows + j] = Scalar::random_default();
            }
        }
    }

    pub(crate) fn encode(self) -> Vec<Polynomial> {
        self.data
            .iter()
            .map(|data| Polynomial::encode2(data.clone()))
            .collect()
    }
}

impl internal::WitnessViewState for Witness {
    fn witness(&self) -> &Witness {
        self
    }

    fn witness_mut(&mut self) -> &mut Witness {
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

impl WitnessView for Witness {
    fn width(&self) -> usize {
        self.data.len()
    }

    fn spawn(&mut self, row_offset: usize, column_offset: usize, width: usize) -> impl WitnessView {
        WitnessSection::new(self, row_offset, column_offset, width)
    }
}

#[derive(Debug)]
pub struct WitnessSection<'a> {
    witness: &'a mut Witness,
    row_offset: usize,
    column_offset: usize,
    width: usize,
    row_counter: usize,
}

impl<'a> WitnessSection<'a> {
    fn new(
        witness: &'a mut Witness,
        row_offset: usize,
        column_offset: usize,
        width: usize,
    ) -> Self {
        Self {
            witness,
            row_offset,
            column_offset,
            width,
            row_counter: 0,
        }
    }
}

impl<'a> internal::WitnessViewState for WitnessSection<'a> {
    fn witness(&self) -> &Witness {
        self.witness
    }

    fn witness_mut(&mut self) -> &mut Witness {
        self.witness
    }

    fn row_offset(&self) -> usize {
        self.row_offset
    }

    fn column_offset(&self) -> usize {
        self.column_offset
    }

    fn step_row(&mut self) -> Cell {
        let root_cell = cell(self.row_counter, 0);
        self.row_counter += 1;
        root_cell
    }
}

impl<'a> WitnessView for WitnessSection<'a> {
    fn width(&self) -> usize {
        self.width
    }

    fn spawn(&mut self, row_offset: usize, column_offset: usize, width: usize) -> impl WitnessView {
        WitnessSection::new(
            self.witness,
            self.row_offset + row_offset,
            self.column_offset + column_offset,
            width,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn node<const N: usize>(cells: [Cell; N]) -> BTreeSet<Cell> {
        BTreeSet::from(cells)
    }

    fn collect(partitioner: &Partitioner) -> BTreeSet<BTreeSet<Cell>> {
        partitioner.iter_nodes().cloned().collect()
    }

    #[test]
    fn test_cell_1() {
        let cell = Cell::new(12, 34);
        assert_eq!(cell.row(), 12);
        assert_eq!(cell.column(), 34);
    }

    #[test]
    fn test_cell_2() {
        let cell = Cell::new(56, 78);
        assert_eq!(cell.row(), 56);
        assert_eq!(cell.column(), 78);
    }

    #[test]
    fn test_cell_shorthand_1() {
        let cell = cell(12, 34);
        assert_eq!(cell.row(), 12);
        assert_eq!(cell.column(), 34);
    }

    #[test]
    fn test_cell_shorthand_2() {
        let cell = cell(56, 78);
        assert_eq!(cell.row(), 56);
        assert_eq!(cell.column(), 78);
    }

    #[test]
    fn test_remap_cell_1() {
        let cell = cell(12, 34).remap(cell(56, 78));
        assert_eq!(cell.row(), 68);
        assert_eq!(cell.column(), 112);
    }

    #[test]
    fn test_remap_cell_2() {
        let cell = cell(56, 78).remap(cell(42, 43));
        assert_eq!(cell.row(), 98);
        assert_eq!(cell.column(), 121);
    }

    #[test]
    fn test_empty() {
        let partitioner = Partitioner::default();
        assert_eq!(collect(&partitioner), BTreeSet::default());
    }

    #[test]
    fn test_one_partition_two_wires_1() {
        let w1 = cell(12, 34);
        let w2 = cell(34, 56);
        let mut partitioner = Partitioner::default();
        partitioner.connect(w1, w2);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2])]));
    }

    #[test]
    fn test_one_partition_two_wires_2() {
        let w1 = cell(12, 34);
        let w2 = cell(34, 56);
        let mut partitioner = Partitioner::default();
        partitioner.connect(w2, w1);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2])]));
    }

    #[test]
    fn test_one_partition_three_wires_1() {
        let w1 = cell(12, 34);
        let w2 = cell(34, 56);
        let w3 = cell(12, 56);
        let mut partitioner = Partitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w1, w3);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2, w3])]));
    }

    #[test]
    fn test_one_partition_three_wires_2() {
        let w1 = cell(12, 34);
        let w2 = cell(34, 56);
        let w3 = cell(12, 56);
        let mut partitioner = Partitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w2, w3);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2, w3])]));
    }

    #[test]
    fn test_one_partition_three_wires_3() {
        let w1 = cell(12, 34);
        let w2 = cell(34, 56);
        let w3 = cell(12, 56);
        let mut partitioner = Partitioner::default();
        partitioner.connect(w2, w1);
        partitioner.connect(w1, w3);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2, w3])]));
    }

    #[test]
    fn test_one_partition_three_wires_4() {
        let w1 = cell(12, 34);
        let w2 = cell(34, 56);
        let w3 = cell(12, 56);
        let mut partitioner = Partitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w3, w1);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2, w3])]));
    }

    #[test]
    fn test_one_partition_three_wires_5() {
        let w1 = cell(12, 34);
        let w2 = cell(34, 56);
        let w3 = cell(12, 56);
        let mut partitioner = Partitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w3, w2);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2, w3])]));
    }

    #[test]
    fn test_two_partitions_1() {
        let w1 = cell(12, 34);
        let w2 = cell(34, 56);
        let w3 = cell(56, 78);
        let w4 = cell(78, 90);
        let mut partitioner = Partitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w3, w4);
        assert_eq!(
            collect(&partitioner),
            BTreeSet::from([node([w1, w2]), node([w3, w4])])
        );
    }

    #[test]
    fn test_two_partitions_2() {
        let w1 = cell(12, 34);
        let w2 = cell(34, 56);
        let w3 = cell(56, 78);
        let w4 = cell(78, 90);
        let mut partitioner = Partitioner::default();
        partitioner.connect(w1, w3);
        partitioner.connect(w2, w4);
        assert_eq!(
            collect(&partitioner),
            BTreeSet::from([node([w1, w3]), node([w2, w4])])
        );
    }

    #[test]
    fn test_two_partitions_3() {
        let w1 = cell(12, 34);
        let w2 = cell(34, 56);
        let w3 = cell(56, 78);
        let w4 = cell(78, 90);
        let w5 = cell(90, 12);
        let mut partitioner = Partitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w1, w3);
        partitioner.connect(w4, w5);
        assert_eq!(
            collect(&partitioner),
            BTreeSet::from([node([w1, w2, w3]), node([w4, w5])])
        );
    }

    #[test]
    fn test_two_partitions_4() {
        let w1 = cell(12, 34);
        let w2 = cell(34, 56);
        let w3 = cell(56, 78);
        let w4 = cell(78, 90);
        let w5 = cell(90, 12);
        let mut partitioner = Partitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w4, w5);
        partitioner.connect(w1, w3);
        assert_eq!(
            collect(&partitioner),
            BTreeSet::from([node([w1, w2, w3]), node([w4, w5])])
        );
    }

    // TODO
}
