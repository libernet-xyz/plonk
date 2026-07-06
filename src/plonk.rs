use crate::Constraint;
use crate::utils;
use crate::wires::{Wire, WirePartitioner};
use anyhow::Result;
use starkom_bluesky::Scalar;
use starkom_ff::Field;
use starkom_pcs::{self as pcs, hash::Hash};
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Default blowup factor in logarithmic form.
///
/// Used with the underlying PCS to compute low-degree extensions.
pub const OPTIONS_DEFAULT_BLOWUP_LOG2: usize = 3;

/// Number of extra rows that are implicitly added to all circuits and witnesses for blinding.
///
/// Blinding rows are appended at the end using NOP gates and random scalars in the witness.
///
/// The reason why PLONK requires 3 of them is that they must be strictly more than the number of
/// off-domain locations opened in the underlying polynomial commitment scheme, and PLONK requires
/// opening two such locations: the Fiat-Shamir challenge xi and the shifted point xi*omega (the
/// latter is for the coordinate pair accumulator polynomial of the permutation argument, which
/// contains the witness columns in its definition).
pub const NUM_BLINDING_ROWS: usize = 3;

/// Domain separator tag used for the main Fiat-Shamir challenge.
static DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/plonk/challenge"));

fn padded_size(n: usize) -> usize {
    std::cmp::max(2, n.next_power_of_two())
}

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
        let degree_bound = padded_size(self.num_rows);
        Circuit {
            degree_bound,
            num_columns: self.num_columns,
            gates: self
                .gates
                .into_iter()
                .map(|(constraint, rows)| {
                    let mut data = vec![Scalar::ZERO; degree_bound];
                    for row in rows {
                        data[row] = Scalar::ONE;
                    }
                    (constraint, Polynomial::encode2(data))
                })
                .collect(),
            sigma: vec![], // TODO
            public_gates: self.public_gates,
        }
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
    /// Number of witness rows (including the blinding rows) padded to the next power of 2.
    degree_bound: usize,

    /// Number of witness columns.
    num_columns: usize,

    /// Gates of the circuit indexed by their respective constraints. The values of the map are
    /// selectors that activate on the rows where the gate was used.
    gates: BTreeMap<Constraint, Polynomial>,

    /// Sigma polynomials of the permutation argument, one for every witness column.
    sigma: Vec<Polynomial>,

    /// List of gates that are revealed in the proofs. Each element is a row index.
    public_gates: BTreeSet<usize>,
}

impl Circuit {
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// Makes an empty [`Witness`] objects suitable for use with this circuit.
    pub fn make_witness(&self) -> Witness {
        Witness {
            data: vec![vec![Scalar::ZERO; self.degree_bound]; self.num_columns],
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

    #[inline]
    fn wire(row: usize, column: usize) -> Wire {
        Wire::new(row, column)
    }

    #[test]
    fn test_vitalik_circuit() {
        let mut builder = CircuitBuilder::default();
        let r0 = builder.var(0);
        let r1 = builder.var(1);
        let r2 = builder.var(2);
        let square = builder.add_gate((r0.clone() ^ 2) - r1.clone());
        let result = builder.add_gate(r0.clone() * r1.clone() + r0 + 5 - r2);
        builder.connect(wire(square, 0), wire(result, 0));
        builder.connect(wire(square, 1), wire(result, 1));
        let nop = builder.add_gate(Constraint::default());
        builder.connect(wire(result, 2), wire(nop, 0));
        builder.declare_public_gates([nop]);
        let circuit = builder.build();
        // TODO
    }

    // TODO
}
