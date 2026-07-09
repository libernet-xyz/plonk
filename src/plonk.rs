use crate::Constraint;
use crate::utils;
use crate::wires::{Wire, WirePartitioner};
use anyhow::{Result, anyhow};
use starkom_bluesky::Scalar;
use starkom_ff::Field;
use starkom_ff::PrimeField;
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
const COMMIT_INDEX_QUOTIENT: usize = 3;
const NUM_COMMIT_INDICES: usize = 4;

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
        Constraint::make_var(column_index)
    }

    /// Adds a gate to the circuit.
    pub fn add_gate(&mut self, constraint: Constraint) -> usize {
        self.num_columns = std::cmp::max(
            self.num_columns,
            1 + constraint
                .get_free_variables()
                .into_iter()
                .max()
                .unwrap_or(0),
        );
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

    /// Adds a gate from a parsed constraint expression, panicking if parsing fails.
    ///
    /// Equivalent to `builder.add_gate(expr.parse().unwrap())`.
    pub fn parse_and_add_gate(&mut self, expr: &'static str) -> usize {
        self.add_gate(expr.parse().unwrap())
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
            for node in self.wires.iter_nodes() {
                let indices: Vec<usize> = node
                    .iter()
                    .map(|wire| wire.column() * degree_bound + wire.row())
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
            degree_bound,
            num_columns: self.num_columns,
            gates,
            sigma,
            sigma_values,
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

    /// The [sigma polynomials](`Self::sigma`) expressed on the value domain.
    ///
    /// The layout is analogous to [`Self::sigma`] itself: the values are indexed column-first.
    sigma_values: Vec<Vec<Scalar>>,

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
                    accumulator[i + 1] *=
                        witness.data[j][i] + beta * generator_pow * omega_pow + gamma;
                    accumulator[i + 1] *=
                        (witness.data[j][i] + beta * self.sigma_values[j][i] + gamma)
                            .invert_unwrap();
                    generator_pow *= Scalar::MULTIPLICATIVE_GENERATOR;
                }
                omega_pow *= omega;
            }

            if accumulator.pop().unwrap() != Scalar::ONE {
                return Err(anyhow!("permutation accumulator wraparound check failed"));
            }

            Polynomial::encode2(accumulator)
        };

        let shifted = {
            let mut coefficients = accumulator.clone().take();
            let mut x = Scalar::ONE;
            for coefficient in coefficients.iter_mut() {
                *coefficient *= x;
                x *= omega;
            }
            Polynomial::with_coefficients(coefficients)
        };

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

        let circuit_polynomials = self
            .gates
            .iter()
            .map(|(_, selector)| selector.clone())
            .chain(self.sigma.iter().cloned())
            .collect();

        let mut committer =
            pcs::Committer::<H>::new(self.degree_bound, options.blowup_log2, circuit_polynomials);

        let columns: Vec<Polynomial> = witness
            .data
            .iter()
            .map(|data| Polynomial::encode2(data.clone()))
            .collect();

        committer.add_batch(columns.clone());

        let gate_constraint = {
            let delta = H::hash_two(*DST, committer.transcript_hash(), FIAT_SHAMIR_INDEX_DELTA);
            let mut gate_constraint = Polynomial::default();
            let mut pow = Scalar::ONE;
            for (constraint, selector) in &self.gates {
                gate_constraint += selector.clone() * constraint.compose(columns.as_slice()) * pow;
                pow *= delta;
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
        let (gates, selectors): (Vec<Constraint>, Vec<Polynomial>) = self.gates.into_iter().unzip();
        let sigma = self.sigma;
        let committer = pcs::Committer::<H>::new(
            self.degree_bound,
            options.blowup_log2,
            selectors.into_iter().chain(sigma.into_iter()).collect(),
        );
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
        let (gates, selectors): (Vec<Constraint>, Vec<Polynomial>) = self
            .gates
            .iter()
            .map(|(constraint, selector)| (constraint.clone(), selector.clone()))
            .unzip();
        let sigma = self.sigma.clone();
        let committer = pcs::Committer::<H>::new(
            self.degree_bound,
            options.blowup_log2,
            selectors.into_iter().chain(sigma.into_iter()).collect(),
        );
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

    fn lagrange0(x: Scalar, n: usize) -> Scalar {
        (x.pow_small(n) - Scalar::ONE)
            * (Scalar::from(n as u64) * (x - Scalar::ONE)).invert_unwrap()
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

        let num_gate_selectors = self.gates.len();
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

        let xi = H::hash_two(
            *DST,
            commitment.transcript_hash(COMMIT_INDEX_QUOTIENT + 1),
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

        let sigma: Vec<Scalar> = {
            let offset = num_gate_selectors;
            (0..self.num_columns)
                .map(|i| points[&xi][offset + i])
                .collect()
        };
        let variables: Vec<Scalar> = {
            let offset = num_gate_selectors + num_sigma_polynomials;
            (0..self.num_columns)
                .map(|i| points[&xi][offset + i])
                .collect()
        };

        let gate_constraint: Scalar = {
            let selectors: Vec<Scalar> = self
                .gates
                .iter()
                .enumerate()
                .map(|(i, _)| points[&xi][i])
                .collect();
            let constraints: Vec<Scalar> = self
                .gates
                .iter()
                .map(|constraint| constraint.evaluate(variables.as_slice()))
                .collect();
            let delta = H::hash_two(
                *DST,
                commitment.transcript_hash(COMMIT_INDEX_WITNESS + 1),
                FIAT_SHAMIR_INDEX_DELTA,
            );
            let mut result = Scalar::ZERO;
            let mut pow = Scalar::ONE;
            for (selector, constraint) in selectors.into_iter().zip(constraints.into_iter()) {
                result += selector * constraint * pow;
                pow *= delta;
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
            for (&variable, &sigma) in variables.iter().zip(sigma.iter()) {
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
        assert_eq!(proof.degree_bound(), circuit.degree_bound());
        assert_eq!(proof.blowup_log2(), blowup_log2);
        assert_eq!(
            proof.extended_domain_size(),
            circuit.degree_bound() << blowup_log2
        );
        circuit.verify::<H>(&proof, ProvingOptions { blowup_log2 })?;
        Ok(())
    }

    #[test]
    fn test_vitalik_circuit_sha2_blowup_2() {
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

    const DEFAULT_BLOWUP_LOG2: usize = 1;

    #[test]
    fn test_vitalik_circuit_with_expressions() {
        let mut builder = CircuitBuilder::default();
        let square = builder.parse_and_add_gate("w1 == w0 ^ 2");
        let result = builder.parse_and_add_gate("w2 == w0 * w1 + w0 + 5");
        builder.connect(wire(square, 0), wire(result, 0));
        builder.connect(wire(square, 1), wire(result, 1));
        let nop = builder.add_gate(Constraint::nop());
        builder.connect(wire(result, 2), wire(nop, 0));
        builder.declare_public_gates([nop]);
        let circuit = builder
            .build(CompilationOptions {
                normalize_constraints: false,
            })
            .unwrap();
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
        let blowup_log2 = DEFAULT_BLOWUP_LOG2;
        let options = ProvingOptions { blowup_log2 };
        let proof = circuit.prove(witness, options.clone()).unwrap();
        assert_eq!(proof.degree_bound(), circuit.degree_bound());
        assert_eq!(proof.blowup_log2(), blowup_log2);
        assert_eq!(
            proof.extended_domain_size(),
            circuit.degree_bound() << blowup_log2
        );
        assert!(circuit.verify::<Sha2Hash<Scalar>>(&proof, options).is_ok());
    }
}
