use crate::Constraint;
use crate::utils;
use crate::wires::{Wire, WirePartitioner};
use anyhow::{Result, anyhow};
use starkom_bluesky::Scalar;
use starkom_ff::Field;
use starkom_pcs::{self as pcs, hash::Hash};
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Default blowup factor (16) in logarithmic form.
///
/// Used with the underlying PCS to compute low-degree extensions.
pub const OPTIONS_DEFAULT_BLOWUP_LOG2: usize = 4;

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

const COMMIT_INDEX_CIRCUIT: usize = 0;
const COMMIT_INDEX_WITNESS: usize = 1;
const COMMIT_INDEX_PERMUTATION_ARGUMENT: usize = 2;

const COMMIT_INDEX_QUOTIENT: usize = 2; // TODO: update
const NUM_COMMIT_INDICES: usize = 3; // TODO: update

const FIAT_SHAMIR_INDEX_ALPHA: Scalar = Scalar::from_const(0);
const FIAT_SHAMIR_INDEX_BETA: Scalar = Scalar::from_const(1);
const FIAT_SHAMIR_INDEX_GAMMA: Scalar = Scalar::from_const(2);
const FIAT_SHAMIR_INDEX_DELTA: Scalar = Scalar::from_const(3);
const FIAT_SHAMIR_INDEX_XI: Scalar = Scalar::from_const(4);

/// Domain separator tag used for the main Fiat-Shamir challenge.
static DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/plonk/challenge"));

fn padded_size(mut n: usize) -> usize {
    n += NUM_BLINDING_ROWS;
    std::cmp::max(2, n.next_power_of_two())
}

/// Circuit compilation & proving options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilationOptions {
    /// Normalizes all constraints using [`Constraint::normalize`].
    ///
    /// When disabled, proving errors out if there are negative exponents rather than attempting
    /// normalization.
    ///
    /// Normalization is carried out inside [`CircuitBuilder::build`].
    ///
    /// WARNING: normalized constraints may be more permissive than their original form because a
    /// negative exponent requires the variable to be different from zero. Starkom does not allow
    /// proving with negative exponents, so enable this flag only if your circuit is correctly
    /// constrained even when those variables are zero.
    pub normalize_constraints: bool,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            normalize_constraints: false,
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
    pub fn build(self, options: CompilationOptions) -> Result<Circuit> {
        let degree_bound = padded_size(self.num_rows);

        let gates = self
            .gates
            .into_iter()
            .map(|(mut constraint, rows)| {
                if !constraint.is_normal() {
                    if options.normalize_constraints {
                        constraint = constraint.normalize();
                    } else {
                        return Err(anyhow!("constraint `{}` is not in normal form", constraint));
                    }
                }
                let mut data = vec![Scalar::ZERO; degree_bound];
                for row in rows {
                    data[row] = Scalar::ONE;
                }
                Ok((constraint, Polynomial::encode2(data)))
            })
            .collect::<Result<_>>()?;

        Ok(Circuit {
            num_rows: self.num_rows,
            degree_bound,
            num_columns: self.num_columns,
            gates,
            sigma: vec![], // TODO
            public_gates: self.public_gates,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Witness {
    /// The number of witness rows *not* including the blinding rows.
    num_rows: usize,

    /// Witness table cells, indexed column-first.
    ///
    /// The column-first indexing allows quickly interpolating polynomials for the columns.
    data: Vec<Vec<Scalar>>,
}

impl Witness {
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn degree_bound(&self) -> usize {
        padded_size(self.num_rows)
    }

    pub fn num_columns(&self) -> usize {
        self.data.len()
    }

    /// Reads a witness cell.
    pub fn get(&self, wire: Wire) -> Scalar {
        let row = wire.row();
        assert!(row < self.num_rows);
        self.data[wire.column()][row]
    }

    /// Updates a witness cell.
    pub fn set(&mut self, wire: Wire, value: Scalar) {
        let row = wire.row();
        assert!(row < self.num_rows);
        self.data[wire.column()][row] = value;
    }

    /// Copies a witness cell to another.
    pub fn copy(&mut self, src_wire: Wire, dst_wire: Wire) -> Scalar {
        let src_row = src_wire.row();
        let dst_row = dst_wire.row();
        assert!(src_row < self.num_rows);
        assert!(dst_row < self.num_rows);
        let value = self.data[src_wire.column()][src_row];
        self.data[dst_wire.column()][dst_row] = value;
        value
    }

    /// Adds blinding rows to the polynomial.
    ///
    /// This is for internal use, [`Circuit::prove`] calls it automatically.
    fn blind(&mut self) {
        for column in &mut self.data {
            for i in 0..NUM_BLINDING_ROWS {
                column[self.num_rows + i] = Scalar::random_default();
            }
        }
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
    commitment: pcs::Commitment,
    inner_proof: pcs::Proof<H>,
}

/// A PLONK circuit.
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

    /// Gates used in the circuit: the first component of each pair is the gate constraint and the
    /// second component is the selector / Lagrange basis polynomial that activates on the rows
    /// where that gate was used.
    gates: Vec<(Constraint, Polynomial)>,

    /// Sigma polynomials of the permutation argument, one for every witness column.
    sigma: Vec<Polynomial>,

    /// List of gates that are revealed in the proofs. Each element is a row index.
    public_gates: BTreeSet<usize>,
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
        Witness {
            num_rows: self.num_rows,
            data: vec![vec![Scalar::ZERO; self.degree_bound]; self.num_columns],
        }
    }

    /// Calculates the degree bound of the PLONK quotient, typically much higher than
    /// [`Self::degree_bound()`] because the constraint equations involve several polynomial
    /// multiplications such as the gate selectors by the gate constraints combined with the witness
    /// columns.
    ///
    /// This function is used to calculate exactly how many chunks the quotient needs to be split
    /// into before getting committed.
    ///
    /// The algorithm assumes that all gate selectors have degree<N, where N is the general
    /// [degree bound](`Self::degree_bound`) of the circuit.
    fn get_quotient_degree_bound(&self) -> usize {
        (self.degree_bound - 1)
            * (1 + self
                .gates
                .iter()
                .map(|(constraint, _)| constraint.get_degree())
                .max()
                .unwrap_or(0))
            + 1
    }

    /// Splits the quotient polynomial in chunks so that it can be batch-committed even if its
    /// degree is much higher than the bound configured in the underlying PCS.
    fn split_quotient(&self, quotient: Polynomial) -> Vec<Polynomial> {
        let degree_bound = self.get_quotient_degree_bound();
        let mut coefficients = quotient.take();
        assert!(coefficients.len() <= degree_bound);
        coefficients.resize(degree_bound, Scalar::ZERO);
        coefficients
            .chunks(self.degree_bound)
            .map(|coefficients| Polynomial::with_coefficients(coefficients.to_vec()))
            .collect()
    }

    /// Proves correctness for the given witness, or returns an error in case of a constraint
    /// violation.
    pub fn prove<H: Hash<Scalar>>(
        &self,
        mut witness: Witness,
        options: ProvingOptions,
    ) -> Result<Proof<H>> {
        witness.blind();
        if witness.degree_bound() != self.degree_bound {
            return Err(anyhow!(
                "incorrect witness size (got {}, want {})",
                witness.degree_bound(),
                self.degree_bound
            ));
        }

        let selectors = self
            .gates
            .iter()
            .map(|(_, selector)| selector.clone())
            .collect();

        let mut committer =
            pcs::Committer::<H>::new(self.degree_bound, options.blowup_log2, selectors);

        let columns: Vec<Polynomial> = witness
            .data
            .into_iter()
            .map(|data| Polynomial::encode2(data))
            .collect();

        committer.add_batch(columns.clone());

        let gate_constraint = {
            let delta = H::hash_two(
                *DST,
                committer.root_hash(COMMIT_INDEX_WITNESS),
                FIAT_SHAMIR_INDEX_DELTA,
            );
            let mut gate_constraint = Polynomial::default();
            let mut pow = Scalar::ONE;
            for (constraint, selector) in &self.gates {
                gate_constraint += selector.clone() * constraint.compose(columns.as_slice()) * pow;
                pow *= delta;
            }
            gate_constraint
        };

        // TODO: prove wire constraints.

        let quotient = gate_constraint.divide_by_zero(self.degree_bound)?;
        committer.add_batch(self.split_quotient(quotient));

        let xi = H::hash_two(
            *DST,
            committer.root_hash(COMMIT_INDEX_QUOTIENT),
            FIAT_SHAMIR_INDEX_XI,
        );

        let omega = Polynomial::domain_element2(1, self.degree_bound);

        let (commitment, prover) = committer.commit(BTreeSet::from_iter(
            [xi, xi * omega]
                .into_iter()
                .chain(self.public_gates.iter().map(|&row| omega.pow_small(row))),
        ));
        let inner_proof = prover.prove(&commitment);

        Ok(Proof {
            commitment,
            inner_proof,
        })
    }

    pub fn to_compressed<H: Hash<Scalar>>(self, options: ProvingOptions) -> CompressedCircuit<H> {
        let (gates, selectors) = self.gates.into_iter().unzip();
        let committer = pcs::Committer::<H>::new(self.degree_bound, options.blowup_log2, selectors);
        CompressedCircuit {
            num_rows: self.num_rows,
            degree_bound: self.degree_bound,
            num_columns: self.num_columns,
            options,
            gates,
            public_gates: self.public_gates,
            circuit_commitment: committer.root_hash(COMMIT_INDEX_CIRCUIT),
            _data: Default::default(),
        }
    }

    pub fn as_compressed<H: Hash<Scalar>>(&self, options: ProvingOptions) -> CompressedCircuit<H> {
        let (gates, selectors) = self
            .gates
            .iter()
            .map(|(constraint, selector)| (constraint.clone(), selector.clone()))
            .unzip();
        let committer = pcs::Committer::<H>::new(self.degree_bound, options.blowup_log2, selectors);
        CompressedCircuit {
            num_rows: self.num_rows,
            degree_bound: self.degree_bound,
            num_columns: self.num_columns,
            options,
            gates,
            public_gates: self.public_gates.clone(),
            circuit_commitment: committer.root_hash(COMMIT_INDEX_CIRCUIT),
            _data: Default::default(),
        }
    }

    pub fn verify<H: Hash<Scalar>>(&self, proof: &Proof<H>, options: ProvingOptions) -> Result<()> {
        self.as_compressed::<H>(options).verify(proof)
    }
}

/// A PLONK circuit in committed form.
///
/// This struct is much smaller than the original circuit but still allows full verification of a
/// proof for the circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedCircuit<H: Hash<Scalar>> {
    /// The raw number of rows of the circuit.
    ///
    /// Unlike [`Self::degree_bound`], this count doesn't include the blinding rows and is not
    /// padded to the next power of 2.
    num_rows: usize,

    /// Number of witness rows (including the blinding rows) rounded up to the next power of 2.
    degree_bound: usize,

    /// Number of witness columns.
    num_columns: usize,

    /// Proving options used to commit to this circuit (in [`Circuit::as_compressed`] or
    /// [`Circuit::to_compressed`]).
    options: ProvingOptions,

    /// Gates used in the original circuit.
    gates: Vec<Constraint>,

    /// List of gates that are revealed in the proofs.
    public_gates: BTreeSet<usize>,

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

    /// Calculates the number of chunks the quotient was split into.
    ///
    /// See [`Circuit::get_quotient_degree_bound`] for details.
    fn get_num_quotient_chunks(&self) -> usize {
        1 + self
            .gates
            .iter()
            .map(|constraint| constraint.get_degree())
            .max()
            .unwrap_or(0)
    }

    pub fn verify(&self, proof: &Proof<H>) -> Result<()> {
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

        let num_quotient_chunks = self.get_num_quotient_chunks();
        let expected_polynomials = self.gates.len() + self.num_columns + num_quotient_chunks;
        if inner_proof.num_polys() != expected_polynomials {
            return Err(anyhow!(
                "incorrect number of committed polynomials (got {}, want {})",
                inner_proof.num_polys(),
                expected_polynomials,
            ));
        }

        let omega = Polynomial::domain_element2(1, self.degree_bound);

        let xi = H::hash_two(
            *DST,
            commitment.tree_roots()[COMMIT_INDEX_QUOTIENT],
            FIAT_SHAMIR_INDEX_XI,
        );

        let points = inner_proof.points();
        if !points.contains_key(&xi) {
            return Err(anyhow!(
                "the proof doesn't have an opening for the main Fiat-Shamir challenge"
            ));
        }
        if !points.contains_key(&(xi * omega)) {
            return Err(anyhow!(
                "the proof doesn't have an opening for the shifted Fiat-Shamir challenge"
            ));
        }
        for &gate in &self.public_gates {
            let z = omega.pow_small(gate);
            if !points.contains_key(&z) {
                return Err(anyhow!(
                    "the proof doesn't have an opening for public gate {gate}"
                ));
            }
        }

        inner_proof.verify(&commitment)?;

        let selectors: Vec<Scalar> = self
            .gates
            .iter()
            .enumerate()
            .map(|(i, _)| points[&xi][i])
            .collect();

        let constraints: Vec<Scalar> = {
            let offset = self.gates.len();
            let variables: Vec<Scalar> = (0..self.num_columns)
                .map(|i| points[&xi][offset + i])
                .collect();
            self.gates
                .iter()
                .map(|constraint| constraint.evaluate(variables.as_slice()))
                .collect()
        };

        let gate_constraint: Scalar = selectors
            .into_iter()
            .zip(constraints.into_iter())
            .map(|(selector, constraint)| selector * constraint)
            .sum::<Scalar>();

        // TODO: recover wire constraints when they're available.

        let quotient: Scalar = {
            let offset = self.gates.len() + self.num_columns;
            (0..num_quotient_chunks)
                .map(|i| points[&xi][offset + i] * xi.pow_small(i * self.degree_bound))
                .sum()
        };
        let zero = xi.pow_small(self.degree_bound) - Scalar::ONE;
        if gate_constraint != quotient * zero {
            return Err(anyhow!("constraint violation"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starkom_bluesky::from_const;
    use starkom_pcs::hash::{Poseidon2Hash, Sha2Hash};

    #[inline]
    fn wire(row: usize, column: usize) -> Wire {
        Wire::new(row, column)
    }

    // This function tests the circuit from Vitalik's PLONK tutorial,
    // https://vitalik.eth.limo/general/2019/09/22/plonk.html#how-plonk-works.
    fn test_vitalik_circuit_impl<H: Hash<Scalar>>(blowup_log2: usize) -> Result<()> {
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
        let circuit = builder.build(CompilationOptions {
            normalize_constraints: false,
        })?;
        assert_eq!(circuit.num_rows(), 3);
        assert_eq!(circuit.degree_bound(), 8);
        assert_eq!(circuit.num_columns(), 3);
        let mut witness = circuit.make_witness();
        let x = from_const(3);
        witness.set(wire(square, 0), x);
        witness.set(wire(square, 1), x.square());
        witness.copy(wire(square, 0), wire(result, 0));
        witness.copy(wire(square, 1), wire(result, 1));
        witness.set(wire(result, 2), x.cube() + x + from_const(5));
        witness.copy(wire(result, 2), wire(nop, 0));
        let proof = circuit.prove::<H>(witness, ProvingOptions { blowup_log2 })?;
        circuit.verify::<H>(&proof, ProvingOptions { blowup_log2 })?;
        Ok(())
    }

    #[test]
    fn test_vitalik_circuit_sha2_blowup_2() {
        test_vitalik_circuit_impl::<Sha2Hash<Scalar>>(1).unwrap();
        assert!(test_vitalik_circuit_impl::<Sha2Hash<Scalar>>(1).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_poseidon2_blowup_2() {
        assert!(test_vitalik_circuit_impl::<Poseidon2Hash<Scalar>>(1).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_sha2_blowup_4() {
        assert!(test_vitalik_circuit_impl::<Sha2Hash<Scalar>>(2).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_poseidon2_blowup_4() {
        assert!(test_vitalik_circuit_impl::<Poseidon2Hash<Scalar>>(2).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_sha2_blowup_8() {
        assert!(test_vitalik_circuit_impl::<Sha2Hash<Scalar>>(3).is_ok());
    }

    #[test]
    fn test_vitalik_circuit_poseidon2_blowup_8() {
        assert!(test_vitalik_circuit_impl::<Poseidon2Hash<Scalar>>(3).is_ok());
    }

    // TODO
}
