use anyhow::Result;
use starkom_bluesky::Scalar;
use starkom_ff::Field;
use starkom_pcs::{self as pcs, hash::Hash};
use std::collections::{BTreeMap, BTreeSet, btree_map};
use std::marker::PhantomData;
use std::ops::{Add, BitXor, Div, Mul, Neg, Sub};

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Default blowup factor in logarithmic form.
///
/// Used with the underlying PCS to compute low-degree extensions.
pub const OPTIONS_DEFAULT_BLOWUP_LOG2: usize = 3;

/// Represents a PLONK constraint as a sum of monomials.
///
/// Each monomial is in the form `coeff * var0^exp0 * var1^exp1 * ...`, where `coeff` is a constant
/// scalar, the `var` variables are witness columns, and the `exp` variables are constant exponents.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Constraint {
    /// The outer map represents the monomials in this constraint, while the inner maps represent
    /// the variables (ie. witness columns) in each monomial.
    ///
    /// The keys of the inner map are column indices and the values are exponents to which the
    /// corresponding variable is raised.
    ///
    /// The values of the outer map are the constant coefficients of each monomial.
    monomials: BTreeMap<BTreeMap<usize, Scalar>, Scalar>,
}

impl Constraint {
    /// Multiplies two monomials.
    ///
    /// The two monomials have the same layout as the inner maps of [`Self::monomials`]. Note that
    /// the coefficients are missing, they must be handled by the caller.
    fn multiply_variables(
        lhs: BTreeMap<usize, Scalar>,
        rhs: BTreeMap<usize, Scalar>,
    ) -> BTreeMap<usize, Scalar> {
        let mut result = lhs;
        for (column_index, exponent) in rhs {
            match result.get_mut(&column_index) {
                Some(preexisting_exponent) => {
                    *preexisting_exponent += exponent;
                }
                None => {
                    result.insert(column_index, exponent);
                }
            }
        }
        result
            .into_iter()
            .filter(|(_, exponent)| *exponent != Scalar::ZERO)
            .collect()
    }

    /// Returns a textual representation of the constraint formula.
    pub fn to_str(&self) -> String {
        self.monomials
            .iter()
            .map(|(variables, &coefficient)| {
                (coefficient != Scalar::ZERO)
                    .then(|| coefficient.to_str_radix(10, 0, false))
                    .into_iter()
                    .chain(
                        variables
                            .iter()
                            .map(|(&column_index, &exponent)| match exponent {
                                Scalar::ONE => format!("w[{}]", column_index),
                                exponent => format!(
                                    "w[{}] ^ {}",
                                    column_index,
                                    exponent.to_str_radix(10, 0, false)
                                ),
                            }),
                    )
                    .collect::<Vec<String>>()
                    .join(" * ")
            })
            .collect::<Vec<String>>()
            .join(" + ")
    }

    /// Evaluates the constraint using the provided variable substitution.
    ///
    /// NOTE: this function panics if one or more variables are missing from the substitution.
    ///
    /// NOTE: this algorithm is intentionally not constant-time because all constraint shapes are
    /// publicly known, so our timing doesn't reveal anything sensitive. Besides, this function is
    /// used by the verifier code, where we don't have anything to leak and we want to maximize
    /// performance.
    pub fn evaluate(&self, substitution: BTreeMap<usize, Scalar>) -> Scalar {
        let mut result = Scalar::ZERO;
        for (variables, coefficient) in &self.monomials {
            let mut value = *coefficient;
            if value == Scalar::ZERO {
                continue;
            }
            for (column_index, &exponent) in variables {
                let variable = substitution[column_index];
                match exponent {
                    Scalar::ZERO => {}
                    Scalar::ONE => {
                        value *= variable;
                    }
                    exponent => {
                        value *= variable.pow_vartime(exponent);
                    }
                }
            }
            result += value;
        }
        result
    }
}

impl Add for Constraint {
    type Output = Constraint;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (variables, coefficient) in rhs.monomials {
            match self.monomials.get_mut(&variables) {
                Some(preexisting_coefficient) => {
                    *preexisting_coefficient += coefficient;
                }
                None => {
                    self.monomials.insert(variables, coefficient);
                }
            }
        }
        self.monomials = self
            .monomials
            .into_iter()
            .filter(|(_, coefficient)| *coefficient != Scalar::ZERO)
            .collect();
        self
    }
}

impl Add<Scalar> for Constraint {
    type Output = Constraint;

    fn add(mut self, rhs: Scalar) -> Self::Output {
        self.monomials.insert(BTreeMap::default(), rhs);
        self
    }
}

impl Sub for Constraint {
    type Output = Constraint;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for (variables, coefficient) in rhs.monomials {
            match self.monomials.get_mut(&variables) {
                Some(preexisting_coefficient) => {
                    *preexisting_coefficient -= coefficient;
                }
                None => {
                    self.monomials.insert(variables, coefficient);
                }
            }
        }
        self.monomials = self
            .monomials
            .into_iter()
            .filter(|(_, coefficient)| *coefficient != Scalar::ZERO)
            .collect();
        self
    }
}

impl Sub<Scalar> for Constraint {
    type Output = Constraint;

    fn sub(mut self, rhs: Scalar) -> Self::Output {
        self.monomials.insert(BTreeMap::default(), -rhs);
        self
    }
}

impl Neg for Constraint {
    type Output = Constraint;

    fn neg(mut self) -> Self::Output {
        for (_, coefficient) in &mut self.monomials {
            *coefficient = coefficient.neg();
        }
        self
    }
}

impl Mul for Constraint {
    type Output = Constraint;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut monomials = BTreeMap::default();
        for (lhs_variables, lhs_coefficient) in self.monomials {
            if lhs_coefficient != Scalar::ZERO {
                for (rhs_variables, &rhs_coefficient) in &rhs.monomials {
                    if rhs_coefficient != Scalar::ZERO {
                        let variables =
                            Self::multiply_variables(lhs_variables.clone(), rhs_variables.clone());
                        let coefficient = lhs_coefficient * rhs_coefficient;
                        match monomials.get_mut(&variables) {
                            Some(preexisting_coefficient) => {
                                *preexisting_coefficient += coefficient
                            }
                            None => {
                                monomials.insert(variables, coefficient);
                            }
                        }
                    }
                }
            }
        }
        Constraint { monomials }
    }
}

impl Mul<Scalar> for Constraint {
    type Output = Constraint;

    fn mul(mut self, rhs: Scalar) -> Self::Output {
        if rhs == Scalar::ZERO {
            return Constraint {
                monomials: BTreeMap::default(),
            };
        }
        for (_, coefficient) in &mut self.monomials {
            *coefficient *= rhs;
        }
        self
    }
}

impl BitXor<Scalar> for Constraint {
    type Output = Constraint;

    /// We use the XOR operator to actually implement exponentiation. For example, if `x` is a
    /// `Constraint` instance (representing a single variable) then `x ^ 5` means x raised to 5.
    fn bitxor(self, rhs: Scalar) -> Self::Output {
        if rhs == Scalar::ZERO {
            return Constraint {
                monomials: BTreeMap::from([(BTreeMap::default(), Scalar::ONE)]),
            };
        }
        if rhs == Scalar::ONE {
            return self;
        }
        match self.monomials.len() {
            0 => Constraint {
                monomials: BTreeMap::default(),
            },
            1 => Constraint {
                monomials: self
                    .monomials
                    .into_iter()
                    .map(|(variables, coefficient)| {
                        (
                            variables
                                .into_iter()
                                .map(|(column_index, exponent)| (column_index, exponent * rhs))
                                .collect(),
                            coefficient.pow_vartime(rhs),
                        )
                    })
                    .collect(),
            },
            _ => panic!("raising a sum to a power is forbidden, try to simplify your constraint"),
        }
    }
}

impl BitXor<isize> for Constraint {
    type Output = Constraint;

    /// We use the XOR operator to actually implement exponentiation. For example, if `x` is a
    /// `Constraint` instance (representing a single variable) then `x ^ 5` means x raised to 5.
    ///
    /// This implementation allows negative exponents too, resulting in modular inversion. For
    /// example, `x ^ -1` correctly yields the modular inverse of `x` (internally it raises to
    /// `p - 2` = [`Scalar::MAX`] - 1), `x ^ -2` correctly yields the square of the modular inverse
    /// (internally raises to `(p - 2) * 2`), and so on.
    ///
    /// WARNING: inverting more than once will yield unexpected results due to the fact that the
    /// exponent -1 (which conventionally indicates modular inversion) actually maps to the field
    /// element that's congruent to -2. So for instance `(x ^ -1) ^ -1` will actually yield `x ^ 4`
    /// rather than `x ^ 1`.
    fn bitxor(self, rhs: isize) -> Self::Output {
        match rhs {
            0 => Constraint {
                monomials: BTreeMap::from([(BTreeMap::default(), Scalar::ONE)]),
            },
            1 => self,
            rhs => {
                let abs = rhs.unsigned_abs() as u64;
                let scalar = if rhs < 0 {
                    (Scalar::MAX - Scalar::ONE) * Scalar::from(abs)
                } else {
                    Scalar::from(abs)
                };
                self.bitxor(scalar)
            }
        }
    }
}

impl Div for Constraint {
    type Output = Constraint;

    /// Multiplies the LHS by the inverse of the RHS, which must have exactly one monomial.
    ///
    /// WARNING: if the monomial of the RHS contains inverted variables (e.g. `x ^ -1`) this
    /// division will yield unexpected results due to the fact that the exponent -1 (which
    /// conventionally indicates modular inversion) actually maps to the field element that's
    /// congruent to -2. So for instance `x / (y ^ -1)` will actually yield `x / (y ^ 4)` rather
    /// than `x * y`.
    fn div(self, rhs: Self) -> Self::Output {
        match rhs.monomials.len() {
            0 => panic!("division by zero"),
            1 => Constraint {
                monomials: self
                    .monomials
                    .into_iter()
                    .map(|(variables, coefficient)| {
                        (
                            variables
                                .into_iter()
                                .map(|(column_index, exponent)| {
                                    (column_index, exponent * (Scalar::MAX - Scalar::ONE))
                                })
                                .collect(),
                            coefficient.invert_vartime().unwrap(),
                        )
                    })
                    .collect(),
            },
            _ => panic!("dividing by a polynomial is forbidden, try to simplify your constraint"),
        }
    }
}

impl Div<Scalar> for Constraint {
    type Output = Constraint;

    fn div(self, rhs: Scalar) -> Self::Output {
        self.mul(rhs.invert_vartime().unwrap())
    }
}

/// A "wire" is a termination of a gate, identified by a row index and a column index.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Wire {
    row: usize,
    column: usize,
}

#[derive(Debug, Clone)]
struct NodeIterator<'a> {
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
struct WirePartitioning {
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

impl WirePartitioning {
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
    wires: WirePartitioning,

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
        Constraint {
            monomials: BTreeMap::from([(
                BTreeMap::from([(column_index, Scalar::ONE)]),
                Scalar::ONE,
            )]),
        }
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
    pub fn get(&self, row: usize, column: usize) -> Scalar {
        self.data[column][row]
    }

    /// Updates a witness cell.
    pub fn set(&mut self, value: Scalar, row: usize, column: usize) {
        self.data[column][row] = value;
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
