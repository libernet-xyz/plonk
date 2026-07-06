use crate::Constraint;
use crate::wires::{Wire, WirePartitioner};
use anyhow::Result;
use starkom_bluesky::Scalar;
use starkom_ff::Field;
use starkom_pcs::{self as pcs, hash::Hash};
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Default blowup factor in logarithmic form.
///
/// Used with the underlying PCS to compute low-degree extensions.
pub const OPTIONS_DEFAULT_BLOWUP_LOG2: usize = 3;

/// Circuit compilation & proving options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// Log2 of the blowup factor used to compute the low-degree extensions for the underlying PCS.
    pub blowup_log2: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            blowup_log2: OPTIONS_DEFAULT_BLOWUP_LOG2,
        }
    }
}

/// Allows building PLONK [`Circuit`]s.
#[derive(Debug, Default, Clone)]
pub struct CircuitBuilder {
    /// Current number of gates/rows in the circuit.
    num_rows: usize,

    /// Current number of columns in the circuit.
    num_columns: usize,

    /// The gates of the circuit, indexed by constraint.
    ///
    /// For every gate type (that is, for every unique gate constraint) this map associates the list
    /// of rows where the gate is active. During circuit compilation (triggered by
    /// [`Self::build`]) each list of rows will be converted to a Lagrange basis that activates on
    /// those rows, aka a "selector".
    gates: BTreeMap<Constraint, Vec<usize>>,

    /// Wire partitioning inferred from the connections made with [`Self::connect`].
    wires: WirePartitioner,

    /// List of gates that are revealed in the proofs. Each element is a row index.
    public_gates: BTreeSet<usize>,
}

impl CircuitBuilder {
    /// Creates a variable representing a witness column.
    ///
    /// These variables can be combined with Rust operators to construct gate constraints. Supported
    /// operations on variables are: addition (`+`), subtraction (binary `-`), negation (unary `-`),
    /// multiplication (`*`), and exponentiation by a constant (`^` followed by a constant).
    pub fn var(&mut self, column_index: usize) -> Constraint {
        self.num_columns = std::cmp::max(self.num_columns, column_index + 1);
        Constraint::make_var(column_index)
    }

    /// Adds a gate to the circuit.
    pub fn add_gate(&mut self, constraint: Constraint) -> usize {
        let row = self.num_rows;
        self.num_rows += 1;
        match self.gates.get_mut(&constraint) {
            Some(rows) => {
                rows.push(row);
            }
            None => {
                self.gates.insert(constraint, vec![row]);
            }
        }
        row
    }

    /// Connects two [`Wire`]s of the circuit.
    pub fn connect(&mut self, wire1: Wire, wire2: Wire) {
        self.wires.connect(wire1, wire2);
    }

    /// Updates the list of witness rows that are revealed.
    ///
    /// This method drops any previously provided lists, so if it's called multiple times only the
    /// list provided in the last call is used.
    ///
    /// Ideally you should call this method only once after adding all gates, right before
    /// [`CircuitBuilder::build`].
    pub fn declare_public_gates<I: IntoIterator<Item = usize>>(&mut self, gates: I) {
        self.public_gates = BTreeSet::from_iter(gates);
    }

    /// Compiles the circuit built so far into a [`Circuit`] object.
    pub fn build(self) -> Circuit {
        // TODO
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct Witness {
    /// Witness table cells, indexed column-first.
    ///
    /// The column-first indexing allows quickly interpolating polynomials for the columns.
    data: Vec<Vec<Scalar>>,
}

impl Witness {
    /// Reads a witness cell.
    pub fn get(&self, wire: Wire) -> Scalar {
        self.data[wire.column()][wire.row()]
    }

    /// Updates a witness cell.
    pub fn set(&mut self, wire: Wire, value: Scalar) {
        self.data[wire.column()][wire.row()] = value;
    }

    /// Copies a witness cell to another.
    pub fn copy(&mut self, src_wire: Wire, dst_wire: Wire) -> Scalar {
        let value = self.data[src_wire.column()][src_wire.row()];
        self.data[dst_wire.column()][dst_wire.row()] = value;
        value
    }
}

impl Index<Wire> for Witness {
    type Output = Scalar;

    fn index(&self, index: Wire) -> &Self::Output {
        &self.data[index.column()][index.row()]
    }
}

impl IndexMut<Wire> for Witness {
    fn index_mut(&mut self, index: Wire) -> &mut Self::Output {
        &mut self.data[index.column()][index.row()]
    }
}

#[derive(Debug, Clone)]
pub struct Proof<H: Hash<Scalar>> {
    inner: pcs::Proof<H>,
    public_inputs: BTreeMap<Wire, Scalar>,
}

/// A PLONK circuit.
#[derive(Debug, Clone)]
pub struct Circuit {
    /// Number of witness rows.
    num_rows: usize,

    /// Number of witness columns.
    num_columns: usize,

    /// Gates of the circuit indexed by their respective constraints. The values of the map are
    /// selectors that activate on the rows where the gate was used.
    gates: BTreeMap<Constraint, Polynomial>,

    /// Sigma polynomials of the permutation argument, one for every witness column.
    sigma: Vec<Polynomial>,
}

impl Circuit {
    /// Makes an empty [`Witness`] objects suitable for use with this circuit.
    pub fn make_witness(&self) -> Witness {
        Witness {
            data: vec![vec![Scalar::ZERO; self.num_rows]; self.num_columns],
        }
    }

    /// Proves correctness for the given witness, or returns an error in case of a constraint
    /// violation.
    pub fn prove<H: Hash<Scalar>>(&self, witness: Witness, options: Options) -> Result<Proof<H>> {
        // TODO
        todo!()
    }
}

/// A PLONK circuit in committed form.
///
/// Having logarithmic size this struct is very small compared to the original circuit, but it still
/// allows full verification of a proof for the circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedCircuit<H: Hash<Scalar>> {
    /// Number of witness rows.
    num_rows: usize,

    /// Number of witness columns.
    num_columns: usize,

    /// Polynomial commitment for the circuit.
    commitment: pcs::Commitment,

    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> CompressedCircuit<H> {
    pub fn verify(&self, proof: &Proof<H>) -> Result<()> {
        // TODO
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
