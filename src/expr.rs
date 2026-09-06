use crate::lexer;
use crate::parser;
use crate::utils::{is_pseudo_negative, isize_to_scalar};
use crate::witness::Cell;
use starkom_bluesky::Scalar as BS;
use starkom_ff::Field;
use starkom_goldilocks::GL;
use starkom_poly::Polynomial;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, BitXor, BitXorAssign, Div, DivAssign, Index, Mul, MulAssign, Neg, Sub,
    SubAssign,
};
use std::str::FromStr;

/// Short-hand for [`Constraint::make_var`].
#[inline]
pub fn var<F: Field>(column_index: usize) -> Constraint<F> {
    Constraint::make_var(column_index, 0)
}

/// Short-hand for [`Constraint::make_var`] with a rotation.
#[inline]
pub fn rvar<F: Field>(column_index: usize, rotation: isize) -> Constraint<F> {
    Constraint::make_var(column_index, rotation)
}

/// Short-hand for [`Constraint::make_const`].
#[inline]
pub fn make_const<F: Field>(value: F) -> Constraint<F> {
    Constraint::make_const(value)
}

/// Represents a variable in a [`Constraint`] expression.
///
/// The variable is identified by its witness column index and a "rotation", which is a row offset
/// relative to where the constraint applies.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Variable {
    column_index: usize,
    rotation: isize,
}

impl Variable {
    /// Creates a new `Variable`.
    pub const fn new(column_index: usize, rotation: isize) -> Self {
        Self {
            column_index,
            rotation,
        }
    }

    /// Witness column index the variable corresponds to.
    pub const fn column_index(&self) -> usize {
        self.column_index
    }

    /// Row offset relative to where the [`Constraint`] applies.
    pub const fn rotation(&self) -> isize {
        self.rotation
    }

    /// Maps the variable to a witness cell given the root cell where the constraint applies.
    ///
    /// For example, if the root cell is at row 12 and column 34 then `var(3, +2)` maps to row 14
    /// and column 37.
    pub const fn map_to_cell(&self, root_cell: Cell) -> Cell {
        Cell::new(
            if self.rotation < 0 {
                let row = root_cell.row();
                let rotation_abs = self.rotation.unsigned_abs();
                assert!(rotation_abs <= row);
                row - rotation_abs
            } else {
                root_cell.row() + self.rotation.unsigned_abs()
            },
            root_cell.column() + self.column_index,
        )
    }

    /// Remaps the variable to a different column index, as per [`Constraint::remap_variables`].
    pub const fn remap(self, column_offset: isize) -> Self {
        assert!(self.column_index <= isize::MAX as usize);
        let column_index = column_offset + (self.column_index as isize);
        assert!(column_index >= 0);
        Self {
            column_index: column_index as usize,
            rotation: self.rotation,
        }
    }

    /// Used by [`Constraint::compose`] when replacing variables with column polynomials.
    pub(crate) fn rotate_column<F: Field>(
        &self,
        column: Polynomial<F>,
        omega: F,
        omega_inv: F,
    ) -> Polynomial<F> {
        // NOTE: `pow_small_vartime` is okay here because these calls depend on the circuit
        // structure, not on the witness.
        match self.rotation.cmp(&0) {
            Ordering::Less => {
                column.shift_domain_by(omega_inv.pow_small_vartime(self.rotation.unsigned_abs()))
            }
            Ordering::Greater => {
                column.shift_domain_by(omega.pow_small_vartime(self.rotation.unsigned_abs()))
            }
            Ordering::Equal => column,
        }
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.rotation < 0 {
            write!(
                f,
                "var({},-{})",
                self.column_index,
                self.rotation.unsigned_abs()
            )
        } else if self.rotation > 0 {
            write!(
                f,
                "var({},+{})",
                self.column_index,
                self.rotation.unsigned_abs()
            )
        } else {
            write!(f, "var({})", self.column_index)
        }
    }
}

impl Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// Represents a PLONK constraint as a sum of monomials (implicitly constrained to equal 0).
///
/// Each monomial is in the form `coeff * var0^exp0 * var1^exp1 * ...`, where `coeff` is a constant
/// scalar, the `var` variables are witness columns, and the `exp` variables are constant exponents.
#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Constraint<F: Field> {
    /// The outer map represents the monomials in this constraint, while the inner maps represent
    /// the variables in each monomial.
    ///
    /// The values of the inner map are (possibly negative) exponents to which the corresponding
    /// variable is raised.
    ///
    /// The values of the outer map are the constant coefficients of each monomial.
    monomials: BTreeMap<BTreeMap<Variable, isize>, F>,
}

impl<F: Field> Constraint<F> {
    /// Makes a [`Constraint`] whose expression is a constant value.
    pub fn make_const(value: F) -> Self {
        Self {
            monomials: if value != F::ZERO {
                BTreeMap::from([(BTreeMap::default(), value)])
            } else {
                BTreeMap::default()
            },
        }
    }

    /// Makes a `Constraint` whose expression is a single variable reference.
    ///
    /// `column_index` is the index of the witness column the variable refers to. `rotation` is the
    /// (possibly negative) row offset relative to where the [`Constraint`] applies. Non-zero
    /// rotations allow gates to refer to witness cells from other rows.
    pub fn make_var(column_index: usize, rotation: isize) -> Self {
        Self {
            monomials: BTreeMap::from([(
                BTreeMap::from([(
                    Variable {
                        column_index,
                        rotation,
                    },
                    1,
                )]),
                F::ONE,
            )]),
        }
    }

    /// Makes a NOP constraint, ie. one that simply constrains `0 == 0`.
    ///
    /// This type of gate can be used for revealing specific wires. Our engine also uses it
    /// internally to add blinding rows.
    ///
    /// The returned constraint is exactly the same as `Constraint::default()`.
    pub fn nop() -> Self {
        Self::default()
    }

    pub(crate) fn get_min_column_index(&self) -> usize {
        self.monomials
            .iter()
            .flat_map(|(variables, _)| {
                variables
                    .iter()
                    .map(|(variable, _)| variable.column_index())
            })
            .min()
            .unwrap_or(0)
    }

    pub fn remap_variables(self, column_offset: isize) -> Self {
        Self {
            monomials: self
                .monomials
                .into_iter()
                .map(|(variables, coefficient)| {
                    (
                        variables
                            .into_iter()
                            .map(|(variable, exponent)| (variable.remap(column_offset), exponent))
                            .collect(),
                        coefficient,
                    )
                })
                .collect(),
        }
    }

    /// Removes variables with zero exponents from every monomial and monomials with zero
    /// coefficients from the overall Constraint.
    ///
    /// Invoked after every operation so that the invariants are always held.
    fn normalize(&mut self) {
        *self = std::mem::take(self).to_normalized();
    }

    /// Chainable version of [`Self::normalize`].
    fn to_normalized(self) -> Self {
        let mut monomials = BTreeMap::default();
        self.monomials
            .into_iter()
            .map(|(variables, coefficient)| {
                (
                    variables
                        .into_iter()
                        .filter(|(_, exponent)| *exponent != 0)
                        .collect::<BTreeMap<Variable, isize>>(),
                    coefficient,
                )
            })
            .filter(|(_, coefficient)| *coefficient != F::ZERO)
            .for_each(|(variables, coefficient)| {
                *monomials.entry(variables).or_default() += coefficient;
            });
        Constraint { monomials }
    }

    /// Multiplies two monomials.
    ///
    /// The two monomials have the same layout as the inner maps of [`Self::monomials`]. Note that
    /// the coefficients are missing, they must be handled by the caller.
    fn multiply_variables<I: IntoIterator<Item = (Variable, isize)>>(
        lhs: BTreeMap<Variable, isize>,
        rhs: I,
    ) -> BTreeMap<Variable, isize> {
        let mut result = lhs;
        for (variable, exponent) in rhs {
            *result.entry(variable).or_default() += exponent;
        }
        result
    }

    fn print_coefficient(coefficient: &F) -> String {
        if is_pseudo_negative(coefficient) {
            format!(
                "-{}",
                (F::MAX - coefficient + F::ONE).to_str_radix(10, 0, false)
            )
        } else {
            coefficient.to_str_radix(10, 0, false)
        }
    }

    /// Returns a textual representation of the constraint formula.
    ///
    /// This is the internal implementation of [`Self::to_string`].
    fn format_expression(&self) -> String {
        if self.monomials.is_empty() {
            return "0".into();
        }
        self.monomials
            .iter()
            .map(|(variables, coefficient)| {
                if variables.is_empty() {
                    return Self::print_coefficient(coefficient);
                }
                (*coefficient != F::ONE)
                    .then(|| Self::print_coefficient(coefficient))
                    .into_iter()
                    .chain(
                        variables
                            .iter()
                            .map(|(variable, &exponent)| match exponent {
                                1 => variable.to_string(),
                                exponent => {
                                    format!("{} ^ {}", variable.to_string(), exponent)
                                }
                            }),
                    )
                    .collect::<Vec<String>>()
                    .join(" * ")
            })
            .collect::<Vec<String>>()
            .join(" + ")
    }

    /// Indicates whether this constraint expression can be raised to the given power.
    ///
    /// Any expression can be raised to a non-negative power (sums use a square-and-multiply
    /// algorithm), but negative powers are only supported when the expression is not a sum.
    pub fn can_raise_to(&self, exponent: isize) -> bool {
        exponent >= 0 || self.monomials.len() < 2
    }

    /// Returns the list of variables referenced by this constraint expression, represented as a set
    /// of column indices (each variable corresponds to a column index).
    pub fn get_free_variables(&self) -> BTreeSet<Variable> {
        let mut set = BTreeSet::default();
        for (variables, _) in &self.monomials {
            for (&variable, _) in variables {
                set.insert(variable);
            }
        }
        set
    }

    /// Indicates whether this constraint expression is constant, which happens when the
    /// [free variable set](`Self::get_free_variables`) is empty.
    ///
    /// Since `Constraint` instances are expressions that are implicitly equalled to 0, it follows
    /// that a non-zero constant `Constraint` is invalid because it will always fail in all
    /// circuits, regardless of the witness values. For example, `42 == 0` will always block
    /// proving. Because of that, constant-testing is not very useful as a public API. It's mostly
    /// used internally by the expression parser and doesn't have many other use cases.
    pub fn is_constant(&self) -> bool {
        self.monomials
            .iter()
            .all(|(variables, _)| variables.is_empty())
    }

    /// If this constraint expression [is constant](`Self::is_constant`) this function returns its
    /// constant value, otherwise it returns `None`.
    ///
    /// Since `Constraint` instances are expressions that are implicitly equalled to 0, it follows
    /// that a non-zero constant `Constraint` is invalid because it will always fail in all
    /// circuits, regardless of the witness values. For example, `42 == 0` will always block
    /// proving. Because of that, constant-testing is not very useful as a public API. It's mostly
    /// used internally by the expression parser and doesn't have many other use cases.
    pub fn get_constant_value(&self) -> Option<F> {
        let mut value = F::ZERO;
        for (variables, coefficient) in &self.monomials {
            if !variables.is_empty() {
                return None;
            }
            value += coefficient;
        }
        Some(value)
    }

    /// If this constraint is a single monomial with a single [`Variable`], this function returns
    /// that variable; otherwise it returns `None`. The monomial must have unit coefficient and the
    /// variable must have unit exponent, otherwise `None` is also returned.
    pub fn get_variable(&self) -> Option<Variable> {
        let mut maybe_variable = None;
        for (variables, &coefficient) in &self.monomials {
            if coefficient != F::ONE {
                return None;
            }
            for (variable, &exponent) in variables {
                if exponent != 1 {
                    return None;
                }
                if maybe_variable.is_some() {
                    return None;
                } else {
                    maybe_variable = Some(*variable);
                }
            }
        }
        maybe_variable
    }

    /// Returns the first variable with negative exponent, or `None` if there isn't one.
    ///
    /// Used by [`Self::canonicalize`] to find variables to multiply.
    fn get_next_inverted_variable(&self) -> Option<(Variable, isize)> {
        for (variables, _) in &self.monomials {
            for (&variable, &exponent) in variables {
                if exponent < 0 {
                    return Some((variable, exponent));
                }
            }
        }
        None
    }

    /// Converts the constraint to a form where all exponents are positive.
    ///
    /// For example, `x * y^-1 + y == 0` becomes `x + y^2 == 0`.
    ///
    /// This canonical form is suitable for use with [`Self::compose`], which doesn't work with
    /// negative exponents.
    ///
    /// WARNING: the canonical form is always more permissive than the original form because the
    /// latter disallows 0 for any variables with negative exponents. Make sure your circuit is not
    /// underconstrained because of that.
    pub fn canonicalize(mut self) -> Self {
        while let Some((variable, exponent)) = self.get_next_inverted_variable() {
            self.monomials = self
                .monomials
                .into_iter()
                .map(|(variables, coefficient)| {
                    (
                        Self::multiply_variables(variables, [(variable, -exponent)]),
                        coefficient,
                    )
                })
                .collect();
        }
        self.to_normalized()
    }

    /// Indicates whether this constraint is in canonical form as per [`Self::canonicalize`].
    ///
    /// Returns true iff all variables have positive exponents, false if there are negative
    /// exponents.
    pub fn is_canonical(&self) -> bool {
        self.get_next_inverted_variable().is_none()
    }

    /// Calculates the degree of the constraint.
    ///
    /// REQUIRES: the constraint must be in [canonical form](`Self::canonicalize`).
    pub fn get_degree(&self) -> usize {
        let mut degree = 0;
        for (variables, &coefficient) in &self.monomials {
            assert_ne!(coefficient, F::ZERO, "the constraint is not in normal form");
            degree = std::cmp::max(
                degree,
                variables
                    .iter()
                    .map(|(_, &exponent)| {
                        assert!(exponent > 0, "the constraint is not in canonical form");
                        exponent as usize
                    })
                    .sum(),
            );
        }
        degree
    }

    /// Evaluates the constraint using the provided variable substitution.
    ///
    /// The `substitution` map associates a value to each variable.
    ///
    /// This function panics if one or more variables are missing from the substitution.
    ///
    /// NOTE: this function also panics if the constraint expression attempts to compute the modular
    /// inverse of a zeroed variable.
    ///
    /// NOTE: this algorithm is intentionally not constant-time because all constraint shapes are
    /// publicly known, so our timing doesn't reveal anything sensitive. Besides, this function is
    /// used by the verifier code, where we don't have anything to leak and we want to maximize
    /// performance.
    pub fn evaluate<'a, G: Field + From<F>>(
        &'a self,
        substitution: &impl Index<&'a Variable, Output = G>,
    ) -> G {
        let mut result = G::ZERO;
        for (variables, &coefficient) in &self.monomials {
            if coefficient == F::ZERO {
                continue;
            }
            let mut monomial_value: G = coefficient.into();
            for (variable, &exponent) in variables {
                let variable_value = substitution[variable];
                match exponent {
                    0 => {}
                    1 => {
                        monomial_value *= variable_value;
                    }
                    exponent => {
                        if exponent < 0 {
                            monomial_value *= variable_value
                                .invert_unwrap()
                                .pow_small_vartime(exponent.unsigned_abs());
                        } else {
                            monomial_value *= variable_value.pow_small_vartime(exponent as usize);
                        }
                    }
                }
            }
            result += monomial_value;
        }
        result
    }

    /// Creates a [`Polynomial`] by replacing the [`Variable`]s in this constraint expression with
    /// the corresponding witness column polynomials via polynomial composition.
    ///
    /// The `substitution` array contains one polynomial for every witness column and must provide
    /// polynomials for all the columns referred to by the `Constraint`.
    ///
    /// `omega` is the root of unity used in the proof and is used to shift the column polynomials
    /// forward or backward in case one or more variables in the constraint have a non-zero
    /// rotation. Given a rotation offset `r`, shifting works by multiplying all coefficients of a
    /// column by powers of `omega^r`, with negative values of `r` performing modular inversion.
    pub fn compose(&self, omega: F, substitution: &[Polynomial<F>]) -> Polynomial<F> {
        let columns_by_variable = {
            let mut columns_by_variable = BTreeMap::default();
            let omega_inv = omega.invert_vartime().unwrap();
            for (variables, &coefficient) in &self.monomials {
                if coefficient != F::ZERO {
                    for (&variable, _) in variables {
                        if !columns_by_variable.contains_key(&variable) {
                            columns_by_variable.insert(
                                variable,
                                variable.rotate_column(
                                    substitution[variable.column_index()].clone(),
                                    omega,
                                    omega_inv,
                                ),
                            );
                        }
                    }
                }
            }
            columns_by_variable
        };
        self.compose2(&columns_by_variable)
    }

    pub(crate) fn compose2(
        &self,
        substitution: &BTreeMap<Variable, Polynomial<F>>,
    ) -> Polynomial<F> {
        self.monomials
            .iter()
            .map(|(variables, &coefficient)| match variables.len() {
                0 => Polynomial::constant(coefficient),
                1 => {
                    let (variable, &exponent) = variables.iter().next().unwrap();
                    assert!(
                        exponent >= 0,
                        "the constraint must be canonicalized before composition"
                    );
                    let exponent = exponent as usize;
                    match exponent {
                        1 => substitution[variable].clone() * coefficient,
                        _ => {
                            Polynomial::multiply_batch(std::iter::repeat_n(
                                &substitution[variable],
                                exponent,
                            )) * coefficient
                        }
                    }
                }
                _ => {
                    Polynomial::multiply_batch(variables.iter().flat_map(
                        |(variable, &exponent)| {
                            assert!(
                                exponent >= 0,
                                "the constraint must be canonicalized before composition"
                            );
                            std::iter::repeat_n(&substitution[variable], exponent as usize)
                        },
                    )) * coefficient
                }
            })
            .sum()
    }
}

impl<F: Field> Debug for Constraint<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constraint({})", self.format_expression())
    }
}

impl<F: Field> Display for Constraint<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_expression())
    }
}

impl<F: Field> From<Variable> for Constraint<F> {
    fn from(value: Variable) -> Self {
        Constraint::make_var(value.column_index(), value.rotation())
    }
}

impl From<BS> for Constraint<BS> {
    fn from(value: BS) -> Self {
        Constraint::make_const(value)
    }
}

impl From<GL> for Constraint<GL> {
    fn from(value: GL) -> Self {
        Constraint::make_const(value)
    }
}

impl<F: Field> From<isize> for Constraint<F> {
    fn from(value: isize) -> Self {
        Constraint::make_const(isize_to_scalar(value))
    }
}

impl<F: Field> FromStr for Constraint<F> {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let tokens = lexer::tokenize(s)?;
        parser::parse(tokens.as_slice())
    }
}

impl<F: Field, V: Into<Self>> AddAssign<V> for Constraint<F> {
    fn add_assign(&mut self, rhs: V) {
        let rhs = rhs.into();
        for (variables, coefficient) in rhs.monomials {
            *self.monomials.entry(variables).or_default() += coefficient;
        }
        self.normalize();
    }
}

impl<F: Field, V: Into<Self>> Add<V> for Constraint<F> {
    type Output = Self;

    fn add(mut self, rhs: V) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F: Field> Neg for Constraint<F> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for (_, coefficient) in &mut self.monomials {
            *coefficient = coefficient.neg();
        }
        self
    }
}

impl<F: Field, V: Into<Self>> SubAssign<V> for Constraint<F> {
    fn sub_assign(&mut self, rhs: V) {
        let rhs = rhs.into();
        for (variables, coefficient) in rhs.monomials {
            *self.monomials.entry(variables).or_default() -= coefficient;
        }
        self.normalize();
    }
}

impl<F: Field, V: Into<Self>> Sub<V> for Constraint<F> {
    type Output = Self;

    fn sub(mut self, rhs: V) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<F: Field, V: Into<Self>> MulAssign<V> for Constraint<F> {
    fn mul_assign(&mut self, rhs: V) {
        let rhs = rhs.into();
        let mut monomials = BTreeMap::default();
        for (lhs_variables, lhs_coefficient) in std::mem::take(&mut self.monomials) {
            if lhs_coefficient != F::ZERO {
                for (rhs_variables, &rhs_coefficient) in &rhs.monomials {
                    if rhs_coefficient != F::ZERO {
                        let variables = Self::multiply_variables(
                            lhs_variables.clone(),
                            rhs_variables
                                .iter()
                                .map(|(&column_index, &exponent)| (column_index, exponent)),
                        );
                        let coefficient = lhs_coefficient * rhs_coefficient;
                        *monomials.entry(variables).or_default() += coefficient;
                    }
                }
            }
        }
        self.monomials = monomials;
        self.normalize();
    }
}

impl<F: Field, V: Into<Self>> Mul<V> for Constraint<F> {
    type Output = Self;

    fn mul(mut self, rhs: V) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<F: Field> BitXorAssign<isize> for Constraint<F> {
    /// We use the XOR operator to actually implement exponentiation. For example, if `x` is a
    /// `Constraint` instance (representing a single variable) then `x ^ 5` means x raised to 5.
    ///
    /// Negative exponents are supported and they actually perform modular inversion.
    ///
    /// WARNING: in Rust the circumflex operator `^` has lower precedence than the arithmetic
    /// operators `+`, `-`, `*`, and `/`, so for example `x + y ^ 2` actually means `(x + y) ^ 2`.
    /// That's counterintuitive but unfortunately Rust doesn't provide a proper power operation, and
    /// exponentiation is often necessary when defining PLONK constraints. Make sure to always
    /// parenthesize accordingly, eg. `x + (y ^ 2)`.
    fn bitxor_assign(&mut self, mut rhs: isize) {
        match rhs {
            0 => {
                self.monomials = BTreeMap::from([(BTreeMap::default(), F::ONE)]);
            }
            1 => {}
            _ => match self.monomials.len() {
                0 => {
                    assert!(rhs >= 0, "cannot raise 0 to a negative power");
                }
                1 => {
                    self.monomials = std::mem::take(&mut self.monomials)
                        .into_iter()
                        .map(|(variables, coefficient)| {
                            (
                                variables
                                    .into_iter()
                                    .map(|(column_index, exponent)| (column_index, exponent * rhs))
                                    .collect(),
                                if rhs < 0 {
                                    coefficient.invert_unwrap()
                                } else {
                                    coefficient
                                }
                                .pow_small_vartime(rhs.unsigned_abs()),
                            )
                        })
                        .collect();
                }
                _ => {
                    assert!(rhs >= 0, "cannot raise a sum to a negative power");
                    let mut result = Self::make_const(F::ONE);
                    while rhs != 0 {
                        if (rhs & 1) != 0 {
                            result.mul_assign(self.clone());
                        }
                        rhs >>= 1;
                        self.mul_assign(self.clone());
                    }
                    *self = result;
                }
            },
        }
    }
}

impl<F: Field> BitXor<isize> for Constraint<F> {
    type Output = Self;

    fn bitxor(mut self, rhs: isize) -> Self::Output {
        self ^= rhs;
        self
    }
}

impl<F: Field, V: Into<Self>> DivAssign<V> for Constraint<F> {
    /// Multiplies the LHS by the inverse of the RHS, which must have exactly one monomial and must
    /// not be zero.
    fn div_assign(&mut self, rhs: V) {
        let rhs = rhs.into();
        match rhs.monomials.len() {
            0 => panic!("division by zero"),
            1 => *self *= rhs.bitxor(-1),
            _ => panic!("dividing by a polynomial is forbidden, try to simplify your constraint"),
        }
    }
}

impl<F: Field, V: Into<Self>> Div<V> for Constraint<F> {
    type Output = Self;

    fn div(mut self, rhs: V) -> Self::Output {
        self /= rhs;
        self
    }
}

impl<F: Field> Sum for Constraint<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Constraint::default(), |a, b| a + b)
    }
}

impl<F: Field> Product for Constraint<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Constraint::make_const(F::ONE), |a, b| a * b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starkom_bluesky::{Scalar, from_const};

    #[inline]
    fn cell(row: usize, column: usize) -> Cell {
        Cell::new(row, column)
    }

    #[test]
    fn test_raw_variable_1() {
        let variable = Variable::new(12, 34);
        assert_eq!(variable.column_index(), 12);
        assert_eq!(variable.rotation(), 34);
        assert_eq!(variable.map_to_cell(cell(42, 43)), cell(76, 55));
    }

    #[test]
    fn test_raw_variable_2() {
        let variable = Variable::new(56, -44);
        assert_eq!(variable.column_index(), 56);
        assert_eq!(variable.rotation(), -44);
        assert_eq!(variable.map_to_cell(cell(78, 45)), cell(34, 101));
    }

    #[test]
    fn test_compare_variables() {
        let v1 = Variable::new(34, 56);
        let v2 = Variable::new(34, 12);
        let v3 = Variable::new(34, 78);
        let v4 = Variable::new(56, 78);
        assert!(v1 == v1);
        assert!(v1 > v2);
        assert!(v1 < v3);
        assert!(v1 < v4);
        assert!(v2 == v2);
        assert!(v2 < v3);
        assert!(v2 < v4);
        assert!(v3 == v3);
        assert!(v3 < v4);
        assert!(v4 == v4);
    }

    #[test]
    fn test_remap_raw_variable() {
        let variable = Variable::new(12, 34);
        assert_eq!(variable.remap(56), Variable::new(68, 34));
    }

    #[test]
    #[should_panic]
    fn test_remap_raw_variable_negative_overflow_panics() {
        let _ = Variable::new(0, 34).remap(-1);
    }

    #[test]
    fn test_remap_variables_single() {
        let constraint: Constraint<Scalar> = var(3).remap_variables(5);
        assert_eq!(constraint, var(8));
    }

    #[test]
    fn test_remap_variables_negative_offset() {
        let constraint: Constraint<Scalar> = var(5).remap_variables(-5);
        assert_eq!(constraint, var(0));
    }

    #[test]
    fn test_remap_variables_preserves_rotation() {
        let constraint: Constraint<Scalar> = rvar(3, -2).remap_variables(4);
        assert_eq!(constraint, rvar(7, -2));
    }

    #[test]
    fn test_remap_variables_sum() {
        let constraint: Constraint<Scalar> = (var(0) + var(2)).remap_variables(3);
        assert_eq!(constraint, var(3) + var(5));
    }

    #[test]
    fn test_remap_variables_preserves_exponents() {
        let constraint: Constraint<Scalar> = (var(2) ^ 3).remap_variables(1);
        assert_eq!(constraint, var(3) ^ 3);
    }

    #[test]
    fn test_remap_variables_preserves_constant_terms() {
        let constraint: Constraint<Scalar> = (var(0) + 7).remap_variables(2);
        assert_eq!(constraint, var(2) + 7);
    }

    #[test]
    fn test_remap_variables_zero_offset_is_identity() {
        let constraint: Constraint<Scalar> = var(0) * var(1) + 9;
        assert_eq!(constraint.clone().remap_variables(0), constraint);
    }

    #[test]
    fn test_remap_variables_round_trip() {
        let constraint: Constraint<Scalar> = var(2) + var(4) * var(6);
        let remapped = constraint.clone().remap_variables(10).remap_variables(-10);
        assert_eq!(remapped, constraint);
    }

    #[test]
    #[should_panic]
    fn test_remap_variables_negative_overflow_panics() {
        let _: Constraint<Scalar> = var(0).remap_variables(-1);
    }

    #[test]
    fn test_min_column_index_empty() {
        let constraint: Constraint<Scalar> = Constraint::nop();
        assert_eq!(constraint.get_min_column_index(), 0);
    }

    #[test]
    fn test_min_column_index_constant() {
        let constraint: Constraint<Scalar> = make_const(from_const(5));
        assert_eq!(constraint.get_min_column_index(), 0);
    }

    #[test]
    fn test_min_column_index_single_variable_at_zero() {
        let constraint: Constraint<Scalar> = var(0);
        assert_eq!(constraint.get_min_column_index(), 0);
    }

    #[test]
    fn test_min_column_index_single_variable_nonzero() {
        let constraint: Constraint<Scalar> = var(3);
        assert_eq!(constraint.get_min_column_index(), 3);
    }

    #[test]
    fn test_min_column_index_ignores_rotation() {
        let constraint: Constraint<Scalar> = rvar(4, -3) + rvar(4, 7);
        assert_eq!(constraint.get_min_column_index(), 4);
    }

    #[test]
    fn test_min_column_index_multiple_variables_in_one_monomial() {
        let constraint: Constraint<Scalar> = var(6) * var(3);
        assert_eq!(constraint.get_min_column_index(), 3);
    }

    #[test]
    fn test_min_column_index_across_monomials() {
        let constraint: Constraint<Scalar> = (var(7) * var(9)) + var(2);
        assert_eq!(constraint.get_min_column_index(), 2);
    }

    #[test]
    fn test_min_column_index_unordered_sum() {
        let constraint: Constraint<Scalar> = var(5) + var(2) + var(8);
        assert_eq!(constraint.get_min_column_index(), 2);
    }

    #[test]
    fn test_min_column_index_constant_term_does_not_mask_variable_minimum() {
        let constraint: Constraint<Scalar> = var(5) + 3;
        assert_eq!(constraint.get_min_column_index(), 5);
    }

    fn evaluate<const N: usize>(
        constraint: &Constraint<Scalar>,
        substitution: [Scalar; N],
    ) -> Scalar {
        let variables: Vec<Variable> = constraint.get_free_variables().into_iter().collect();
        assert_eq!(variables.len(), N);
        let substitution: BTreeMap<Variable, Scalar> = variables
            .into_iter()
            .zip(substitution.into_iter())
            .collect();
        constraint.evaluate(&substitution)
    }

    #[test]
    fn test_empty() {
        let constraint: Constraint<Scalar> = Constraint::nop();
        assert_eq!(constraint, Constraint::default());
        assert_eq!(evaluate(&constraint, []), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    fn test_constant_impl(value: Scalar) {
        let constraint: Constraint<Scalar> = Constraint {
            monomials: BTreeMap::from([(BTreeMap::default(), value)]),
        };
        assert_eq!(constraint.get_free_variables(), BTreeSet::default());
        assert!(constraint.is_constant());
        assert_eq!(constraint.get_constant_value(), Some(value));
        assert_eq!(evaluate(&constraint, []), value);
        assert_eq!(constraint.to_string(), value.to_str_radix(10, 0, false));
    }

    #[test]
    fn test_constant() {
        test_constant_impl(from_const(0));
        test_constant_impl(from_const(12));
        test_constant_impl(from_const(34));
    }

    #[test]
    fn test_variable_0() {
        let constraint: Constraint<Scalar> = var(0);
        assert_eq!(
            constraint.get_free_variables(),
            BTreeSet::from([Variable::new(0, 0)])
        );
        assert!(!constraint.is_constant());
        assert!(constraint.get_constant_value().is_none());
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(0)");
    }

    #[test]
    fn test_variable_1() {
        let constraint: Constraint<Scalar> = var(1);
        assert_eq!(
            constraint.get_free_variables(),
            BTreeSet::from([Variable::new(1, 0)])
        );
        assert!(!constraint.is_constant());
        assert!(constraint.get_constant_value().is_none());
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(1)");
    }

    #[test]
    fn test_rotated_variable_1() {
        let constraint: Constraint<Scalar> = rvar(2, 1);
        assert_eq!(
            constraint.get_free_variables(),
            BTreeSet::from([Variable::new(2, 1)])
        );
        assert!(!constraint.is_constant());
        assert!(constraint.get_constant_value().is_none());
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(2,+1)");
    }

    #[test]
    fn test_rotated_variable_2() {
        let constraint: Constraint<Scalar> = rvar(2, -1);
        assert_eq!(
            constraint.get_free_variables(),
            BTreeSet::from([Variable::new(2, -1)])
        );
        assert!(!constraint.is_constant());
        assert!(constraint.get_constant_value().is_none());
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(2,-1)");
    }

    #[test]
    fn test_sum_1() {
        let constraint: Constraint<Scalar> = var(0) + var(1);
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            from_const(46)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(12)]),
            from_const(46)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(56), from_const(78)]),
            from_const(134)
        );
        assert_eq!(constraint.to_string(), "var(0) + var(1)");
    }

    #[test]
    fn test_sum_2() {
        let constraint: Constraint<Scalar> = var(1) + var(0);
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            from_const(46)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(12)]),
            from_const(46)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(56), from_const(78)]),
            from_const(134)
        );
        assert_eq!(constraint.to_string(), "var(0) + var(1)");
    }

    #[test]
    fn test_sum_3() {
        let constraint: Constraint<Scalar> = rvar(2, -1) + rvar(2, 1);
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            from_const(46)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(12)]),
            from_const(46)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(56), from_const(78)]),
            from_const(134)
        );
        assert_eq!(constraint.to_string(), "var(2,-1) + var(2,+1)");
    }

    #[test]
    fn test_another_sum() {
        let constraint: Constraint<Scalar> = var(0) + var(1) + var(2);
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(12), from_const(34), from_const(56)]
            ),
            from_const(102)
        );
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(12), from_const(56), from_const(34)]
            ),
            from_const(102)
        );
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(34), from_const(56), from_const(78)]
            ),
            from_const(168)
        );
        assert_eq!(constraint.to_string(), "var(0) + var(1) + var(2)");
    }

    #[test]
    fn test_add_scalar_1() {
        let constraint: Constraint<Scalar> = var(0) + from_const(12);
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(46));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(68));
        assert_eq!(constraint.to_string(), "12 + var(0)");
    }

    #[test]
    fn test_add_scalar_2() {
        let constraint: Constraint<Scalar> = var(0) + from_const(34);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(46));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(90));
        assert_eq!(constraint.to_string(), "34 + var(0)");
    }

    #[test]
    fn test_add_another_scalar() {
        let constraint: Constraint<Scalar> = var(0) + from_const(34) + from_const(56);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(102));
        assert_eq!(evaluate(&constraint, [from_const(78)]), from_const(168));
        assert_eq!(constraint.to_string(), "90 + var(0)");
    }

    #[test]
    fn test_optimize_sum_1() {
        let constraint: Constraint<Scalar> = var(0) + var(0) * -from_const(1);
        assert_eq!(evaluate(&constraint, []), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    #[test]
    fn test_optimize_sum_2() {
        let constraint: Constraint<Scalar> = var(0) + var(1) * -from_const(1);
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            -from_const(22)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(12)]),
            from_const(22)
        );
        assert_eq!(constraint.to_string(), "var(0) + -1 * var(1)");
    }

    #[test]
    fn test_optimize_sum_3() {
        let constraint: Constraint<Scalar> =
            var(0) + var(1) * -from_const(1) + var(0) * -from_const(1);
        assert_eq!(evaluate(&constraint, [from_const(12)]), -from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), -from_const(34));
        assert_eq!(constraint.to_string(), "-1 * var(1)");
    }

    #[test]
    fn test_negate_variable() {
        let constraint: Constraint<Scalar> = -var(0);
        assert_eq!(evaluate(&constraint, [from_const(12)]), -from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), -from_const(34));
        assert_eq!(constraint.to_string(), "-1 * var(0)");
    }

    #[test]
    fn test_negate_sum() {
        let constraint: Constraint<Scalar> = -(var(0) + var(1) + from_const(12));
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(56)]),
            -from_const(102)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(56), from_const(78)]),
            -from_const(146)
        );
        assert_eq!(constraint.to_string(), "-12 + -1 * var(0) + -1 * var(1)");
    }

    #[test]
    fn test_diff_1() {
        let constraint: Constraint<Scalar> = var(0) - var(1);
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            -from_const(22)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(12)]),
            from_const(22)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(56), from_const(78)]),
            -from_const(22)
        );
        assert_eq!(constraint.to_string(), "var(0) + -1 * var(1)");
    }

    #[test]
    fn test_diff_2() {
        let constraint: Constraint<Scalar> = var(1) - var(0);
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            from_const(22)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(12)]),
            -from_const(22)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(56), from_const(78)]),
            from_const(22)
        );
        assert_eq!(constraint.to_string(), "-1 * var(0) + var(1)");
    }

    #[test]
    fn test_diff_3() {
        let constraint: Constraint<Scalar> = rvar(2, -1) - rvar(2, 1);
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            -from_const(22)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(12)]),
            from_const(22)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(56), from_const(78)]),
            -from_const(22)
        );
        assert_eq!(constraint.to_string(), "var(2,-1) + -1 * var(2,+1)");
    }

    #[test]
    fn test_another_diff() {
        let constraint: Constraint<Scalar> = var(0) - var(1) - var(2);
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(12), from_const(34), from_const(56)]
            ),
            -from_const(78)
        );
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(12), from_const(56), from_const(34)]
            ),
            -from_const(78)
        );
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(34), from_const(56), from_const(78)]
            ),
            -from_const(100)
        );
        assert_eq!(constraint.to_string(), "var(0) + -1 * var(1) + -1 * var(2)");
    }

    #[test]
    fn test_sub_scalar_1() {
        let constraint: Constraint<Scalar> = var(0) - from_const(12);
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(22));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(44));
        assert_eq!(constraint.to_string(), "-12 + var(0)");
    }

    #[test]
    fn test_sub_scalar_2() {
        let constraint: Constraint<Scalar> = var(0) - from_const(34);
        assert_eq!(evaluate(&constraint, [from_const(12)]), -from_const(22));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(22));
        assert_eq!(constraint.to_string(), "-34 + var(0)");
    }

    #[test]
    fn test_sub_another_scalar() {
        let constraint: Constraint<Scalar> = var(0) - from_const(34) - from_const(56);
        assert_eq!(evaluate(&constraint, [from_const(12)]), -from_const(78));
        assert_eq!(evaluate(&constraint, [from_const(78)]), -from_const(12));
        assert_eq!(constraint.to_string(), "-90 + var(0)");
    }

    #[test]
    fn test_optimize_diff_1() {
        let constraint: Constraint<Scalar> = var(0) - var(0);
        assert_eq!(evaluate(&constraint, []), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    #[test]
    fn test_optimize_diff_2() {
        let constraint: Constraint<Scalar> = var(0) - var(1) * -from_const(1);
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            from_const(46)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(12)]),
            from_const(46)
        );
        assert_eq!(constraint.to_string(), "var(0) + var(1)");
    }

    #[test]
    fn test_optimize_diff_3() {
        let constraint: Constraint<Scalar> = var(0) - var(1) - var(0);
        assert_eq!(evaluate(&constraint, [from_const(12)]), -from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), -from_const(34));
        assert_eq!(constraint.to_string(), "-1 * var(1)");
    }

    #[test]
    fn test_product_1() {
        let constraint: Constraint<Scalar> = var(0) * var(1);
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            from_const(408)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(12)]),
            from_const(408)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(56), from_const(78)]),
            from_const(4368)
        );
        assert_eq!(constraint.to_string(), "var(0) * var(1)");
    }

    #[test]
    fn test_product_2() {
        let constraint: Constraint<Scalar> = var(1) * var(0);
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            from_const(408)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(12)]),
            from_const(408)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(56), from_const(78)]),
            from_const(4368)
        );
        assert_eq!(constraint.to_string(), "var(0) * var(1)");
    }

    #[test]
    fn test_product_3() {
        let constraint: Constraint<Scalar> = rvar(2, -1) * rvar(2, 1);
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            from_const(408)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34), from_const(12)]),
            from_const(408)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(56), from_const(78)]),
            from_const(4368)
        );
        assert_eq!(constraint.to_string(), "var(2,-1) * var(2,+1)");
    }

    #[test]
    fn test_another_product() {
        let constraint: Constraint<Scalar> = var(0) * var(1) * var(2);
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(12), from_const(34), from_const(56)]
            ),
            from_const(22848)
        );
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(12), from_const(56), from_const(34)]
            ),
            from_const(22848)
        );
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(34), from_const(56), from_const(78)]
            ),
            from_const(148512)
        );
        assert_eq!(constraint.to_string(), "var(0) * var(1) * var(2)");
    }

    #[test]
    fn test_product_same_variable() {
        let constraint: Constraint<Scalar> = var(0) * var(0);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(144));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(1156));
        assert_eq!(constraint.to_string(), "var(0) ^ 2");
    }

    #[test]
    fn test_mul_scalar_1() {
        let constraint: Constraint<Scalar> = var(0) * from_const(12);
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(408));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(672));
        assert_eq!(constraint.to_string(), "12 * var(0)");
    }

    #[test]
    fn test_mul_scalar_2() {
        let constraint: Constraint<Scalar> = var(0) * from_const(34);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(408));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(1904));
        assert_eq!(constraint.to_string(), "34 * var(0)");
    }

    #[test]
    fn test_mul_another_scalar() {
        let constraint: Constraint<Scalar> = var(0) * from_const(34) * from_const(56);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(22848));
        assert_eq!(evaluate(&constraint, [from_const(78)]), from_const(148512));
        assert_eq!(constraint.to_string(), "1904 * var(0)");
    }

    #[test]
    fn test_mul_by_zero() {
        let constraint: Constraint<Scalar> = var(0) * from_const(0);
        assert_eq!(evaluate(&constraint, []), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    #[test]
    fn test_optimize_product_1() {
        let constraint: Constraint<Scalar> = var(0) * (var(0) ^ -1);
        assert_eq!(evaluate(&constraint, []), from_const(1));
        assert_eq!(constraint.to_string(), "1");
    }

    #[test]
    fn test_optimize_product_2() {
        let constraint: Constraint<Scalar> = var(0) * (var(0) ^ -1) * var(1);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(1)");
    }

    #[test]
    fn test_optimize_product_3() {
        let constraint: Constraint<Scalar> = (var(0) ^ 2) * (var(0) ^ -1);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(0)");
    }

    #[test]
    fn test_pow_zero_exponent() {
        let constraint: Constraint<Scalar> = var(0) ^ 0;
        assert_eq!(evaluate(&constraint, []), from_const(1));
        assert_eq!(constraint.to_string(), "1");
    }

    #[test]
    fn test_pow_zero_exponent_on_sum() {
        let constraint: Constraint<Scalar> = (var(0) + var(1)) ^ 0;
        assert_eq!(evaluate(&constraint, []), from_const(1));
        assert_eq!(constraint.to_string(), "1");
    }

    #[test]
    fn test_pow_zero_exponent_of_zero() {
        let constraint: Constraint<Scalar> = make_const(from_const(0)) ^ 0;
        assert_eq!(evaluate(&constraint, []), from_const(1));
        assert_eq!(constraint.to_string(), "1");
    }

    #[test]
    fn test_pow_one_exponent() {
        let constraint: Constraint<Scalar> = var(0) ^ 1;
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(0)");
    }

    #[test]
    fn test_pow_one_exponent_on_sum() {
        let constraint: Constraint<Scalar> = (var(0) + var(1)) ^ 1;
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            from_const(46)
        );
        assert_eq!(constraint.to_string(), "var(0) + var(1)");
    }

    #[test]
    fn test_pow_positive_exponent() {
        let constraint: Constraint<Scalar> = var(0) ^ 3;
        assert_eq!(
            evaluate(&constraint, [from_const(12)]),
            from_const(12).pow_small_vartime(3)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34)]),
            from_const(34).pow_small_vartime(3)
        );
        assert_eq!(constraint.to_string(), "var(0) ^ 3");
    }

    #[test]
    fn test_pow_negative_exponent() {
        let constraint: Constraint<Scalar> = var(0) ^ -2;
        assert_eq!(
            evaluate(&constraint, [from_const(12)]),
            from_const(12).invert_unwrap().pow_small_vartime(2)
        );
        assert_eq!(
            evaluate(&constraint, [from_const(34)]),
            from_const(34).invert_unwrap().pow_small_vartime(2)
        );
        assert_eq!(constraint.to_string(), "var(0) ^ -2");
    }

    #[test]
    fn test_pow_constant_positive_exponent() {
        let constraint: Constraint<Scalar> = make_const(from_const(2)) ^ 3;
        assert_eq!(evaluate(&constraint, []), from_const(8));
        assert_eq!(constraint.to_string(), "8");
    }

    #[test]
    fn test_pow_constant_negative_exponent() {
        let constraint: Constraint<Scalar> = make_const(from_const(2)) ^ -1;
        assert_eq!(evaluate(&constraint, []), from_const(2).invert_unwrap());
    }

    #[test]
    fn test_bitxor_assign() {
        let mut constraint: Constraint<Scalar> = var(0);
        constraint ^= 3;
        assert_eq!(
            evaluate(&constraint, [from_const(12)]),
            from_const(12).pow_small_vartime(3)
        );
        assert_eq!(constraint.to_string(), "var(0) ^ 3");
    }

    #[test]
    fn test_squared_sum() {
        let constraint: Constraint<Scalar> = (var(0) + var(1)) ^ 2;
        assert_eq!(
            constraint,
            (var(0) ^ 2) + (var(1) ^ 2) + var(0) * var(1) * 2
        );
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            (from_const(12) + from_const(34)).square()
        );
        assert_eq!(
            constraint.to_string(),
            "2 * var(0) * var(1) + var(0) ^ 2 + var(1) ^ 2"
        );
    }

    #[test]
    fn test_cubed_sum() {
        let constraint: Constraint<Scalar> = (var(0) + var(1)) ^ 3;
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            (from_const(12) + from_const(34)).pow_small_vartime(3)
        );
    }

    #[test]
    #[should_panic(expected = "cannot raise a sum to a negative power")]
    fn test_pow_sum_negative_exponent_panics() {
        let _: Constraint<Scalar> = (var(0) + var(1)) ^ -2;
    }

    #[test]
    #[should_panic(expected = "cannot raise 0 to a negative power")]
    fn test_pow_zero_to_negative_exponent_panics_1() {
        let _: Constraint<Scalar> = make_const(from_const(0)) ^ -1;
    }

    #[test]
    #[should_panic(expected = "cannot raise 0 to a negative power")]
    fn test_pow_zero_to_negative_exponent_panics_2() {
        let _: Constraint<Scalar> = make_const(from_const(0)) ^ -2;
    }

    #[test]
    fn test_parsing() {
        let c1: Constraint<Scalar> = "var(0) ^ 2 * var(1) + var(0) + 5 == 35".parse().unwrap();
        let c2: Constraint<Scalar> = (var(0) ^ 2) * var(1) + var(0) - 30;
        let c3: Constraint<Scalar> = format!("{} == 0", c2).parse().unwrap();
        assert_eq!(c1, c2);
        assert_eq!(c1, c3);
    }

    #[test]
    fn test_iter_sum_empty() {
        let constraint: Constraint<Scalar> = std::iter::empty::<Constraint<Scalar>>().sum();
        assert_eq!(constraint, Constraint::default());
        assert_eq!(constraint.to_string(), "0");
    }

    #[test]
    fn test_iter_sum_single() {
        let constraint: Constraint<Scalar> = vec![var(0)].into_iter().sum();
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(constraint.to_string(), "var(0)");
    }

    #[test]
    fn test_iter_sum() {
        let constraint: Constraint<Scalar> = vec![var(0), var(1), var(2)].into_iter().sum();
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(12), from_const(34), from_const(56)]
            ),
            from_const(102)
        );
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(34), from_const(56), from_const(78)]
            ),
            from_const(168)
        );
        assert_eq!(constraint.to_string(), "var(0) + var(1) + var(2)");
    }

    #[test]
    fn test_iter_product_empty() {
        let constraint: Constraint<Scalar> = std::iter::empty::<Constraint<Scalar>>().product();
        assert_eq!(constraint.get_constant_value(), Some(Scalar::ONE));
        assert_eq!(constraint.to_string(), "1");
    }

    #[test]
    fn test_iter_product_single() {
        let constraint: Constraint<Scalar> = vec![var(0)].into_iter().product();
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(constraint.to_string(), "var(0)");
    }

    #[test]
    fn test_iter_product() {
        let constraint: Constraint<Scalar> = vec![var(0), var(1), var(2)].into_iter().product();
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(12), from_const(34), from_const(56)]
            ),
            from_const(22848)
        );
        assert_eq!(
            evaluate(
                &constraint,
                [from_const(34), from_const(56), from_const(78)]
            ),
            from_const(148512)
        );
        assert_eq!(constraint.to_string(), "var(0) * var(1) * var(2)");
    }
}
