use starkom_bluesky::Scalar;
use std::collections::{BTreeMap, BTreeSet, btree_map};

/// A "wire" is a termination of a gate, identified by a row index and a column index.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Wire {
    row: usize,
    column: usize,
}

impl Wire {
    pub const fn new(row: usize, column: usize) -> Self {
        Self { row, column }
    }

    pub const fn row(&self) -> usize {
        self.row
    }

    pub const fn column(&self) -> usize {
        self.column
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum WireOrUnconstrained {
    Wire(Wire),
    Unconstrained(Scalar),
}

impl From<Wire> for WireOrUnconstrained {
    fn from(wire: Wire) -> Self {
        WireOrUnconstrained::Wire(wire)
    }
}

impl From<Scalar> for WireOrUnconstrained {
    fn from(value: Scalar) -> Self {
        WireOrUnconstrained::Unconstrained(value)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NodeIterator<'a> {
    inner: btree_map::Iter<'a, usize, BTreeSet<Wire>>,
}

impl<'a> Iterator for NodeIterator<'a> {
    type Item = &'a BTreeSet<Wire>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, node)| node)
    }
}

/// Keeps all the wires of a circuit organized in partitions, i.e. sets of interconnected wires.
///
/// Since all the wires in a partition are connected to each other, in this context a partition
/// represents a node of the circuit, so we call partitions "nodes".
///
/// This data structure allows determining the subsets of the sigma polynomials to permute.
#[derive(Debug, Default, Clone)]
pub(crate) struct WirePartitioner {
    /// Next available node ID.
    next_id: usize,

    /// Keys are incremental node IDs, values are nodes.
    nodes: BTreeMap<usize, BTreeSet<Wire>>,

    /// Keys are wires, values are the ID of the node that wire is connected to.
    ///
    /// If a wire is not found here it's implied that it belongs to a partition containing only
    /// that wire, i.e. it's unconstrained.
    node_by_wire: BTreeMap<Wire, usize>,
}

impl WirePartitioner {
    pub(crate) fn connect(&mut self, wire1: Wire, wire2: Wire) {
        if let Some(node_id1) = self.node_by_wire.get(&wire1) {
            if let Some(node_id2) = self.node_by_wire.get(&wire2) {
                if *node_id1 != *node_id2 {
                    let mut node2 = self.nodes.remove(&node_id2).unwrap();
                    let node1 = self.nodes.get_mut(node_id1).unwrap();
                    node1.append(&mut node2);
                    self.node_by_wire.insert(wire2, *node_id1);
                }
            } else {
                let node = self.nodes.get_mut(node_id1).unwrap();
                node.insert(wire2);
                self.node_by_wire.insert(wire2, *node_id1);
            }
        } else {
            if let Some(node_id) = self.node_by_wire.get(&wire2) {
                let node = self.nodes.get_mut(node_id).unwrap();
                node.insert(wire1);
                self.node_by_wire.insert(wire1, *node_id);
            } else {
                let id = self.next_id;
                self.next_id += 1;
                self.nodes.insert(id, BTreeSet::from([wire1, wire2]));
                self.node_by_wire.insert(wire1, id);
                self.node_by_wire.insert(wire2, id);
            }
        }
    }

    pub(crate) fn iter_nodes(&self) -> NodeIterator<'_> {
        NodeIterator {
            inner: self.nodes.iter(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline(always)]
    fn wire(row: usize, column: usize) -> Wire {
        Wire::new(row, column)
    }

    #[inline]
    fn node<const N: usize>(wires: [Wire; N]) -> BTreeSet<Wire> {
        BTreeSet::from(wires)
    }

    fn collect(partitioner: &WirePartitioner) -> BTreeSet<BTreeSet<Wire>> {
        partitioner.iter_nodes().cloned().collect()
    }

    #[test]
    fn test_wire_1() {
        let wire = wire(12, 34);
        assert_eq!(wire.row(), 12);
        assert_eq!(wire.column(), 34);
    }

    #[test]
    fn test_wire_2() {
        let wire = wire(56, 78);
        assert_eq!(wire.row(), 56);
        assert_eq!(wire.column(), 78);
    }

    #[test]
    fn test_empty() {
        let partitioner = WirePartitioner::default();
        assert_eq!(collect(&partitioner), BTreeSet::default());
    }

    #[test]
    fn test_one_partition_two_wires_1() {
        let w1 = wire(12, 34);
        let w2 = wire(34, 56);
        let mut partitioner = WirePartitioner::default();
        partitioner.connect(w1, w2);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2])]));
    }

    #[test]
    fn test_one_partition_two_wires_2() {
        let w1 = wire(12, 34);
        let w2 = wire(34, 56);
        let mut partitioner = WirePartitioner::default();
        partitioner.connect(w2, w1);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2])]));
    }

    #[test]
    fn test_one_partition_three_wires_1() {
        let w1 = wire(12, 34);
        let w2 = wire(34, 56);
        let w3 = wire(12, 56);
        let mut partitioner = WirePartitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w1, w3);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2, w3])]));
    }

    #[test]
    fn test_one_partition_three_wires_2() {
        let w1 = wire(12, 34);
        let w2 = wire(34, 56);
        let w3 = wire(12, 56);
        let mut partitioner = WirePartitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w2, w3);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2, w3])]));
    }

    #[test]
    fn test_one_partition_three_wires_3() {
        let w1 = wire(12, 34);
        let w2 = wire(34, 56);
        let w3 = wire(12, 56);
        let mut partitioner = WirePartitioner::default();
        partitioner.connect(w2, w1);
        partitioner.connect(w1, w3);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2, w3])]));
    }

    #[test]
    fn test_one_partition_three_wires_4() {
        let w1 = wire(12, 34);
        let w2 = wire(34, 56);
        let w3 = wire(12, 56);
        let mut partitioner = WirePartitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w3, w1);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2, w3])]));
    }

    #[test]
    fn test_one_partition_three_wires_5() {
        let w1 = wire(12, 34);
        let w2 = wire(34, 56);
        let w3 = wire(12, 56);
        let mut partitioner = WirePartitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w3, w2);
        assert_eq!(collect(&partitioner), BTreeSet::from([node([w1, w2, w3])]));
    }

    #[test]
    fn test_two_partitions_1() {
        let w1 = wire(12, 34);
        let w2 = wire(34, 56);
        let w3 = wire(56, 78);
        let w4 = wire(78, 90);
        let mut partitioner = WirePartitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w3, w4);
        assert_eq!(
            collect(&partitioner),
            BTreeSet::from([node([w1, w2]), node([w3, w4])])
        );
    }

    #[test]
    fn test_two_partitions_2() {
        let w1 = wire(12, 34);
        let w2 = wire(34, 56);
        let w3 = wire(56, 78);
        let w4 = wire(78, 90);
        let mut partitioner = WirePartitioner::default();
        partitioner.connect(w1, w3);
        partitioner.connect(w2, w4);
        assert_eq!(
            collect(&partitioner),
            BTreeSet::from([node([w1, w3]), node([w2, w4])])
        );
    }

    #[test]
    fn test_two_partitions_3() {
        let w1 = wire(12, 34);
        let w2 = wire(34, 56);
        let w3 = wire(56, 78);
        let w4 = wire(78, 90);
        let w5 = wire(90, 12);
        let mut partitioner = WirePartitioner::default();
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
        let w1 = wire(12, 34);
        let w2 = wire(34, 56);
        let w3 = wire(56, 78);
        let w4 = wire(78, 90);
        let w5 = wire(90, 12);
        let mut partitioner = WirePartitioner::default();
        partitioner.connect(w1, w2);
        partitioner.connect(w4, w5);
        partitioner.connect(w1, w3);
        assert_eq!(
            collect(&partitioner),
            BTreeSet::from([node([w1, w2, w3]), node([w4, w5])])
        );
    }
}
