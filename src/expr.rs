use crate::lexer;
use crate::parser;
use anyhow::Result;
use starkom_bluesky::Scalar;
use starkom_ff::{Field, PrimeField};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display};
use std::ops::{
    Add, AddAssign, BitXor, BitXorAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};
use std::str::FromStr;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Represents a PLONK constraint as a sum of monomials (implicitly constrained to equal 0).
///
/// Each monomial is in the form `coeff * var0^exp0 * var1^exp1 * ...`, where `coeff` is a constant
/// scalar, the `var` variables are witness columns, and the `exp` variables are constant exponents.
#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Constraint {
    /// The outer map represents the monomials in this constraint, while the inner maps represent
    /// the variables (ie. witness columns) in each monomial.
    ///
    /// The keys of the inner map are column indices and the values are (possibly negative)
    /// exponents to which the corresponding variable is raised.
    ///
    /// The values of the outer map are the constant coefficients of each monomial.
    monomials: BTreeMap<BTreeMap<usize, isize>, Scalar>,
}

impl Constraint {
    /// Makes a [`Constraint`] whose expression is a constant value.
    pub fn make_const(value: Scalar) -> Self {
        if value != Scalar::ZERO {
            Constraint {
                monomials: BTreeMap::from([(BTreeMap::default(), value)]),
            }
        } else {
            Constraint::default()
        }
    }

    /// Makes a `Constraint` whose expression is a single variable reference.
    ///
    /// `column_index` is the index of the witness column the variable refers to.
    pub fn make_var(column_index: usize) -> Self {
        Constraint {
            monomials: BTreeMap::from([(BTreeMap::from([(column_index, 1)]), Scalar::ONE)]),
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
                        .collect::<BTreeMap<usize, isize>>(),
                    coefficient,
                )
            })
            .filter(|(_, coefficient)| *coefficient != Scalar::ZERO)
            .for_each(
                |(variables, coefficient)| match monomials.get_mut(&variables) {
                    Some(preexisting_coefficient) => {
                        *preexisting_coefficient += coefficient;
                    }
                    None => {
                        monomials.insert(variables, coefficient);
                    }
                },
            );
        Constraint { monomials }
    }

    /// Multiplies two monomials.
    ///
    /// The two monomials have the same layout as the inner maps of [`Self::monomials`]. Note that
    /// the coefficients are missing, they must be handled by the caller.
    fn multiply_variables<I: IntoIterator<Item = (usize, isize)>>(
        lhs: BTreeMap<usize, isize>,
        rhs: I,
    ) -> BTreeMap<usize, isize> {
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
    }

    fn isize_to_scalar(value: isize) -> Scalar {
        let abs = value.unsigned_abs();
        if value < 0 {
            -Scalar::try_from(abs).unwrap()
        } else {
            Scalar::try_from(abs).unwrap()
        }
    }

    fn is_pseudo_negative(&value: &Scalar) -> bool {
        let half_range = Scalar::MAX * Scalar::TWO_INV;
        value > half_range
    }

    fn print_coefficient(coefficient: &Scalar) -> String {
        if Self::is_pseudo_negative(coefficient) {
            format!(
                "-{}",
                (Scalar::MAX - coefficient + Scalar::ONE).to_str_radix(10, 0, false)
            )
        } else {
            coefficient.to_str_radix(10, 0, false)
        }
    }

    /// Returns a textual representation of the constraint formula.
    pub fn to_string(&self) -> String {
        if self.monomials.is_empty() {
            return "0".into();
        }
        self.monomials
            .iter()
            .map(|(variables, coefficient)| {
                if variables.is_empty() {
                    return Self::print_coefficient(coefficient);
                }
                (*coefficient != Scalar::ONE)
                    .then(|| Self::print_coefficient(coefficient))
                    .into_iter()
                    .chain(
                        variables
                            .iter()
                            .map(|(&column_index, &exponent)| match exponent {
                                1 => format!("w{}", column_index),
                                exponent => {
                                    format!("w{} ^ {}", column_index, exponent)
                                }
                            }),
                    )
                    .collect::<Vec<String>>()
                    .join(" * ")
            })
            .collect::<Vec<String>>()
            .join(" + ")
    }

    /// Indicates whether this constraint expression can be raised to a power.
    ///
    /// Raising can only be done when the expression is not a sum, otherwise our exponentiation
    /// algorithm panics.
    pub fn can_raise(&self) -> bool {
        self.monomials.len() < 2
    }

    /// Returns the list of variables referenced by this constraint expression, represented as a set
    /// of column indices (each variable corresponds to a column index).
    pub fn get_free_variables(&self) -> BTreeSet<usize> {
        let mut set = BTreeSet::default();
        for (variables, _) in &self.monomials {
            for (&column_index, _) in variables {
                set.insert(column_index);
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

    /// If this constraint expression [is constant](`Self::is_constant`) it returns its constant
    /// value, otherwise it returns `None`.
    ///
    /// Since `Constraint` instances are expressions that are implicitly equalled to 0, it follows
    /// that a non-zero constant `Constraint` is invalid because it will always fail in all
    /// circuits, regardless of the witness values. For example, `42 == 0` will always block
    /// proving. Because of that, constant-testing is not very useful as a public API. It's mostly
    /// used internally by the expression parser and doesn't have many other use cases.
    pub fn get_value_if_constant(&self) -> Option<Scalar> {
        let mut value = Scalar::ZERO;
        for (variables, coefficient) in &self.monomials {
            if !variables.is_empty() {
                return None;
            }
            value += coefficient;
        }
        Some(value)
    }

    /// Returns the first variable with negative exponent, or `None` if there isn't one.
    ///
    /// Used by [`Self::canonicalize`] to find variables to multiply.
    fn get_next_inverted_variable(&self) -> Option<(usize, isize)> {
        for (variables, _) in &self.monomials {
            for (&column_index, &exponent) in variables {
                if exponent < 0 {
                    return Some((column_index, exponent));
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
        while let Some((column_index, exponent)) = self.get_next_inverted_variable() {
            self.monomials = self
                .monomials
                .into_iter()
                .map(|(variables, coefficient)| {
                    (
                        Self::multiply_variables(variables, [(column_index, -exponent)]),
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
            assert_ne!(
                coefficient,
                Scalar::ZERO,
                "the constraint is not in normal form"
            );
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
    /// The elements of the `substitution` array correspond to the witness column; the array assigns
    /// a value to every column.
    ///
    /// NOTE: this function panics if one or more variables are missing from the substitution.
    ///
    /// NOTE: this function also panics if the constraint expression attempts to compute the modular
    /// inverse of a zeroed variable.
    ///
    /// NOTE: this algorithm is intentionally not constant-time because all constraint shapes are
    /// publicly known, so our timing doesn't reveal anything sensitive. Besides, this function is
    /// used by the verifier code, where we don't have anything to leak and we want to maximize
    /// performance.
    pub fn evaluate(&self, substitution: &[Scalar]) -> Scalar {
        let mut result = Scalar::ZERO;
        for (variables, &coefficient) in &self.monomials {
            let mut value = coefficient;
            if value == Scalar::ZERO {
                continue;
            }
            for (&column_index, &exponent) in variables {
                let variable = substitution[column_index];
                match exponent {
                    0 => {}
                    1 => {
                        value *= variable;
                    }
                    exponent => {
                        if exponent < 0 {
                            value *= variable
                                .invert_unwrap()
                                .pow_small_vartime(exponent.unsigned_abs());
                        } else {
                            value *= variable.pow_small_vartime(exponent as usize);
                        }
                    }
                }
            }
            result += value;
        }
        result
    }

    pub fn compose(&self, substitution: &[Polynomial]) -> Polynomial {
        let mut result = Polynomial::default();
        for (variables, &coefficient) in &self.monomials {
            if coefficient == Scalar::ZERO {
                continue;
            }
            let mut monomial = Polynomial::constant(coefficient);
            for (&column_index, &exponent) in variables {
                let variable = &substitution[column_index];
                match exponent {
                    0 => {}
                    1 => {
                        monomial *= variable.clone();
                    }
                    exponent => {
                        assert!(
                            exponent > 0,
                            "the constraint must be canonicalized before composition"
                        );
                        for _ in 0..exponent {
                            monomial *= variable.clone();
                        }
                    }
                }
            }
            result += monomial;
        }
        result
    }
}

impl Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constraint({})", self)
    }
}

impl Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl FromStr for Constraint {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let tokens = lexer::tokenize(s)?;
        parser::parse(tokens.as_slice())
    }
}

impl AddAssign for Constraint {
    fn add_assign(&mut self, rhs: Self) {
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
        self.normalize();
    }
}

impl AddAssign<Scalar> for Constraint {
    fn add_assign(&mut self, rhs: Scalar) {
        let variables = BTreeMap::default();
        match self.monomials.get_mut(&variables) {
            Some(coefficient) => {
                *coefficient += rhs;
            }
            None => {
                self.monomials.insert(variables, rhs);
            }
        }
        self.normalize();
    }
}

impl AddAssign<isize> for Constraint {
    fn add_assign(&mut self, rhs: isize) {
        *self += Self::isize_to_scalar(rhs);
    }
}

impl Add for Constraint {
    type Output = Constraint;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<Scalar> for Constraint {
    type Output = Constraint;

    fn add(mut self, rhs: Scalar) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<isize> for Constraint {
    type Output = Constraint;

    fn add(self, rhs: isize) -> Self::Output {
        self.add(Self::isize_to_scalar(rhs))
    }
}

impl SubAssign for Constraint {
    fn sub_assign(&mut self, rhs: Self) {
        for (variables, coefficient) in rhs.monomials {
            match self.monomials.get_mut(&variables) {
                Some(preexisting_coefficient) => {
                    *preexisting_coefficient -= coefficient;
                }
                None => {
                    self.monomials.insert(variables, -coefficient);
                }
            }
        }
        self.normalize();
    }
}

impl SubAssign<Scalar> for Constraint {
    fn sub_assign(&mut self, rhs: Scalar) {
        let variables = BTreeMap::default();
        match self.monomials.get_mut(&variables) {
            Some(coefficient) => {
                *coefficient -= rhs;
            }
            None => {
                self.monomials.insert(variables, -rhs);
            }
        }
        self.normalize();
    }
}

impl SubAssign<isize> for Constraint {
    fn sub_assign(&mut self, rhs: isize) {
        *self -= Self::isize_to_scalar(rhs);
    }
}

impl Sub for Constraint {
    type Output = Constraint;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl Sub<Scalar> for Constraint {
    type Output = Constraint;

    fn sub(mut self, rhs: Scalar) -> Self::Output {
        self -= rhs;
        self
    }
}

impl Sub<isize> for Constraint {
    type Output = Constraint;

    fn sub(mut self, rhs: isize) -> Self::Output {
        self -= rhs;
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

impl MulAssign for Constraint {
    fn mul_assign(&mut self, rhs: Self) {
        let mut monomials = BTreeMap::default();
        for (lhs_variables, lhs_coefficient) in std::mem::take(&mut self.monomials) {
            if lhs_coefficient != Scalar::ZERO {
                for (rhs_variables, &rhs_coefficient) in &rhs.monomials {
                    if rhs_coefficient != Scalar::ZERO {
                        let variables = Self::multiply_variables(
                            lhs_variables.clone(),
                            rhs_variables
                                .iter()
                                .map(|(&column_index, &exponent)| (column_index, exponent)),
                        );
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
        self.monomials = monomials;
        self.normalize();
    }
}

impl MulAssign<Scalar> for Constraint {
    fn mul_assign(&mut self, rhs: Scalar) {
        if rhs != Scalar::ZERO {
            for (_, coefficient) in &mut self.monomials {
                *coefficient *= rhs;
            }
        } else {
            self.monomials = BTreeMap::default();
        }
    }
}

impl MulAssign<isize> for Constraint {
    fn mul_assign(&mut self, rhs: isize) {
        *self *= Self::isize_to_scalar(rhs);
    }
}

impl Mul for Constraint {
    type Output = Constraint;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Mul<Scalar> for Constraint {
    type Output = Constraint;

    fn mul(mut self, rhs: Scalar) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Mul<isize> for Constraint {
    type Output = Constraint;

    fn mul(mut self, rhs: isize) -> Self::Output {
        self *= rhs;
        self
    }
}

impl BitXorAssign<isize> for Constraint {
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
    fn bitxor_assign(&mut self, rhs: isize) {
        match rhs {
            0 => {
                self.monomials = BTreeMap::from([(BTreeMap::default(), Scalar::ONE)]);
            }
            1 => {}
            _ => match self.monomials.len() {
                0 => {}
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
                    panic!("raising a sum to a power is forbidden, try to simplify your constraint")
                }
            },
        }
    }
}

impl BitXor<isize> for Constraint {
    type Output = Constraint;

    fn bitxor(mut self, rhs: isize) -> Self::Output {
        self ^= rhs;
        self
    }
}

impl DivAssign for Constraint {
    /// Multiplies the LHS by the inverse of the RHS, which must have exactly one monomial.
    fn div_assign(&mut self, rhs: Self) {
        match rhs.monomials.len() {
            0 => panic!("division by zero"),
            1 => *self *= rhs.bitxor(-1),
            _ => panic!("dividing by a polynomial is forbidden, try to simplify your constraint"),
        }
    }
}

impl DivAssign<Scalar> for Constraint {
    fn div_assign(&mut self, rhs: Scalar) {
        *self *= rhs.invert_vartime().unwrap();
    }
}

impl DivAssign<isize> for Constraint {
    fn div_assign(&mut self, rhs: isize) {
        *self *= Self::isize_to_scalar(rhs).invert_vartime().unwrap();
    }
}

impl Div for Constraint {
    type Output = Constraint;

    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}

impl Div<Scalar> for Constraint {
    type Output = Constraint;

    fn div(mut self, rhs: Scalar) -> Self::Output {
        self /= rhs;
        self
    }
}

impl Div<isize> for Constraint {
    type Output = Constraint;

    fn div(mut self, rhs: isize) -> Self::Output {
        self /= rhs;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starkom_bluesky::from_const;

    #[inline(always)]
    fn make_var(column_index: usize) -> Constraint {
        Constraint::make_var(column_index)
    }

    #[test]
    fn test_empty() {
        let constraint = Constraint::nop();
        assert_eq!(constraint, Constraint::default());
        assert_eq!(constraint.evaluate(&[]), from_const(0));
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(0));
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(0));
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(0)
        );
        assert_eq!(constraint.to_string(), "0");
    }

    fn test_constant_impl(value: Scalar) {
        let constraint = Constraint {
            monomials: BTreeMap::from([(BTreeMap::default(), value)]),
        };
        assert_eq!(constraint.get_free_variables(), BTreeSet::default());
        assert!(constraint.is_constant());
        assert_eq!(constraint.get_value_if_constant(), Some(value));
        assert_eq!(constraint.evaluate(&[]), value);
        assert_eq!(constraint.evaluate(&[from_const(12)]), value);
        assert_eq!(constraint.evaluate(&[from_const(34)]), value);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            value
        );
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
        let constraint = make_var(0);
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(12));
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(34));
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(56)
        );
        assert_eq!(constraint.to_string(), "w0");
    }

    #[test]
    fn test_variable_1() {
        let constraint = make_var(1);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(34)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(12)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(78)
        );
        assert_eq!(constraint.to_string(), "w1");
    }

    #[test]
    fn test_variable_2() {
        let constraint = make_var(2);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34), from_const(56)]),
            from_const(56)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78), from_const(90)]),
            from_const(90)
        );
        assert_eq!(constraint.to_string(), "w2");
    }

    #[test]
    fn test_sum_1() {
        let constraint = make_var(0) + make_var(1);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(46)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(46)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(134)
        );
        assert_eq!(constraint.to_string(), "w0 + w1");
    }

    #[test]
    fn test_sum_2() {
        let constraint = make_var(1) + make_var(0);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(46)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(46)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(134)
        );
        assert_eq!(constraint.to_string(), "w0 + w1");
    }

    #[test]
    fn test_sum_3() {
        let constraint = make_var(1) + make_var(2);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34), from_const(56)]),
            from_const(90)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(56), from_const(34)]),
            from_const(90)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(56), from_const(12)]),
            from_const(68)
        );
        assert_eq!(constraint.to_string(), "w1 + w2");
    }

    #[test]
    fn test_another_sum() {
        let constraint = make_var(0) + make_var(1) + make_var(2);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34), from_const(56)]),
            from_const(102)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(56), from_const(34)]),
            from_const(102)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(56), from_const(78)]),
            from_const(168)
        );
        assert_eq!(constraint.to_string(), "w0 + w1 + w2");
    }

    #[test]
    fn test_add_scalar_1() {
        let constraint = make_var(0) + from_const(12);
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(46));
        assert_eq!(constraint.evaluate(&[from_const(56)]), from_const(68));
        assert_eq!(constraint.to_string(), "12 + w0");
    }

    #[test]
    fn test_add_scalar_2() {
        let constraint = make_var(0) + from_const(34);
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(46));
        assert_eq!(constraint.evaluate(&[from_const(56)]), from_const(90));
        assert_eq!(constraint.to_string(), "34 + w0");
    }

    #[test]
    fn test_add_another_scalar() {
        let constraint = make_var(0) + from_const(34) + from_const(56);
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(102));
        assert_eq!(constraint.evaluate(&[from_const(78)]), from_const(168));
        assert_eq!(constraint.to_string(), "90 + w0");
    }

    #[test]
    fn test_optimize_sum_1() {
        let constraint = make_var(0) + make_var(0) * -from_const(1);
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(0));
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    #[test]
    fn test_optimize_sum_2() {
        let constraint = make_var(0) + make_var(1) * -from_const(1);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            -from_const(22)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(22)
        );
        assert_eq!(constraint.to_string(), "w0 + -1 * w1");
    }

    #[test]
    fn test_optimize_sum_3() {
        let w0 = make_var(0);
        let w1 = make_var(1);
        let constraint = w0.clone() + w1 * -from_const(1) + w0 * -from_const(1);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            -from_const(34)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            -from_const(12)
        );
        assert_eq!(constraint.to_string(), "-1 * w1");
    }

    #[test]
    fn test_compound_sum_1() {
        let mut constraint = make_var(0);
        constraint += make_var(1);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(46)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(46)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(134)
        );
        assert_eq!(constraint.to_string(), "w0 + w1");
    }

    #[test]
    fn test_compound_sum_2() {
        let mut constraint = make_var(1);
        constraint += make_var(0);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(46)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(46)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(134)
        );
        assert_eq!(constraint.to_string(), "w0 + w1");
    }

    #[test]
    fn test_sub_1() {
        let constraint = make_var(0) - make_var(1);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            -from_const(22)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(22)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            -from_const(22)
        );
        assert_eq!(constraint.to_string(), "w0 + -1 * w1");
    }

    #[test]
    fn test_sub_2() {
        let constraint = make_var(1) - make_var(0);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(22)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            -from_const(22)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(22)
        );
        assert_eq!(constraint.to_string(), "-1 * w0 + w1");
    }

    #[test]
    fn test_sub_3() {
        let constraint = make_var(1) - make_var(2);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34), from_const(56)]),
            -from_const(22)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(56), from_const(34)]),
            from_const(22)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(56), from_const(12)]),
            from_const(44)
        );
        assert_eq!(constraint.to_string(), "w1 + -1 * w2");
    }

    #[test]
    fn test_another_sub() {
        let constraint = make_var(0) - make_var(1) - make_var(2);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34), from_const(56)]),
            -from_const(78)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(56), from_const(34)]),
            -from_const(78)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(56), from_const(78)]),
            -from_const(100)
        );
        assert_eq!(constraint.to_string(), "w0 + -1 * w1 + -1 * w2");
    }

    #[test]
    fn test_sub_scalar_1() {
        let constraint = make_var(0) - from_const(12);
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(22));
        assert_eq!(constraint.evaluate(&[from_const(56)]), from_const(44));
        assert_eq!(constraint.to_string(), "-12 + w0");
    }

    #[test]
    fn test_sub_scalar_2() {
        let constraint = make_var(0) - from_const(34);
        assert_eq!(constraint.evaluate(&[from_const(12)]), -from_const(22));
        assert_eq!(constraint.evaluate(&[from_const(56)]), from_const(22));
        assert_eq!(constraint.to_string(), "-34 + w0");
    }

    #[test]
    fn test_sub_another_scalar() {
        let constraint = make_var(0) - from_const(34) - from_const(56);
        assert_eq!(constraint.evaluate(&[from_const(12)]), -from_const(78));
        assert_eq!(constraint.evaluate(&[from_const(78)]), -from_const(12));
        assert_eq!(constraint.to_string(), "-90 + w0");
    }

    #[test]
    fn test_optimize_sub_1() {
        let constraint = make_var(0) - make_var(0);
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(0));
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    #[test]
    fn test_optimize_sub_2() {
        let w0 = make_var(0);
        let w1 = make_var(1);
        let constraint = w0.clone() - w1 - w0;
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            -from_const(34)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            -from_const(12)
        );
        assert_eq!(constraint.to_string(), "-1 * w1");
    }

    #[test]
    fn test_compound_sub_1() {
        let mut constraint = make_var(0);
        constraint -= make_var(1);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            -from_const(22)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(22)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            -from_const(22)
        );
        assert_eq!(constraint.to_string(), "w0 + -1 * w1");
    }

    #[test]
    fn test_compound_sub_2() {
        let mut constraint = make_var(1);
        constraint -= make_var(0);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(22)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            -from_const(22)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(22)
        );
        assert_eq!(constraint.to_string(), "-1 * w0 + w1");
    }

    #[test]
    fn test_neg_1() {
        let constraint = -make_var(0);
        assert_eq!(constraint.evaluate(&[from_const(12)]), -from_const(12));
        assert_eq!(constraint.evaluate(&[from_const(34)]), -from_const(34));
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            -from_const(56)
        );
        assert_eq!(constraint.to_string(), "-1 * w0");
    }

    #[test]
    fn test_neg_double() {
        let constraint = -(-make_var(0));
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(12));
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(34));
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(56)
        );
        assert_eq!(constraint.to_string(), "w0");
    }

    #[test]
    fn test_neg_sum() {
        let constraint = -(make_var(0) + make_var(1));
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            -from_const(46)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            -from_const(46)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            -from_const(134)
        );
        assert_eq!(constraint.to_string(), "-1 * w0 + -1 * w1");
    }

    #[test]
    fn test_neg_scalar() {
        let constraint = -(make_var(0) + from_const(12));
        assert_eq!(constraint.evaluate(&[from_const(34)]), -from_const(46));
        assert_eq!(constraint.evaluate(&[from_const(56)]), -from_const(68));
        assert_eq!(constraint.to_string(), "-12 + -1 * w0");
    }

    #[test]
    fn test_mul_1() {
        let constraint = make_var(0) * make_var(1);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(408)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(408)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(4368)
        );
        assert_eq!(constraint.to_string(), "w0 * w1");
    }

    #[test]
    fn test_mul_2() {
        let constraint = make_var(1) * make_var(0);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(408)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(408)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(4368)
        );
        assert_eq!(constraint.to_string(), "w0 * w1");
    }

    #[test]
    fn test_mul_3() {
        let constraint = make_var(1) * make_var(2);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34), from_const(56)]),
            from_const(1904)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(56), from_const(34)]),
            from_const(1904)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(56), from_const(12)]),
            from_const(672)
        );
        assert_eq!(constraint.to_string(), "w1 * w2");
    }

    #[test]
    fn test_another_mul() {
        let constraint = make_var(0) * make_var(1) * make_var(2);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34), from_const(56)]),
            from_const(22848)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(56), from_const(34)]),
            from_const(22848)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(56), from_const(78)]),
            from_const(148512)
        );
        assert_eq!(constraint.to_string(), "w0 * w1 * w2");
    }

    #[test]
    fn test_mul_scalar_1() {
        let constraint = make_var(0) * from_const(12);
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(408));
        assert_eq!(constraint.evaluate(&[from_const(56)]), from_const(672));
        assert_eq!(constraint.to_string(), "12 * w0");
    }

    #[test]
    fn test_mul_scalar_2() {
        let constraint = make_var(0) * from_const(34);
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(408));
        assert_eq!(constraint.evaluate(&[from_const(56)]), from_const(1904));
        assert_eq!(constraint.to_string(), "34 * w0");
    }

    #[test]
    fn test_mul_another_scalar() {
        let constraint = make_var(0) * from_const(34) * from_const(56);
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(22848));
        assert_eq!(constraint.evaluate(&[from_const(78)]), from_const(148512));
        assert_eq!(constraint.to_string(), "1904 * w0");
    }

    #[test]
    fn test_mul_by_zero() {
        let constraint = make_var(0) * from_const(0);
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(0));
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    #[test]
    fn test_optimize_mul() {
        let w0 = make_var(0);
        let w1 = make_var(1);
        let constraint = (w0.clone() + w1.clone()) * (w0 - w1);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            -from_const(1012)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(1012)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            -from_const(2948)
        );
        assert_eq!(constraint.to_string(), "w0 ^ 2 + -1 * w1 ^ 2");
    }

    #[test]
    fn test_compound_mul_1() {
        let mut constraint = make_var(0);
        constraint *= make_var(1);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(408)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(408)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(4368)
        );
        assert_eq!(constraint.to_string(), "w0 * w1");
    }

    #[test]
    fn test_compound_mul_2() {
        let mut constraint = make_var(1);
        constraint *= make_var(0);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(408)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(12)]),
            from_const(408)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(78)]),
            from_const(4368)
        );
        assert_eq!(constraint.to_string(), "w0 * w1");
    }

    #[test]
    fn test_pow_0() {
        let constraint = make_var(0) ^ 0;
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(1));
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(1));
        assert_eq!(constraint.to_string(), "1");
    }

    #[test]
    fn test_pow_0_of_zero() {
        let constraint = Constraint::nop() ^ 0;
        assert_eq!(constraint.evaluate(&[]), from_const(1));
        assert_eq!(constraint.to_string(), "1");
    }

    #[test]
    fn test_pow_1() {
        let constraint = make_var(0) ^ 1;
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(12));
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "w0");
    }

    #[test]
    fn test_pow_1_of_sum() {
        let constraint = (make_var(0) + make_var(1)) ^ 1;
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(34)]),
            from_const(46)
        );
        assert_eq!(constraint.to_string(), "w0 + w1");
    }

    #[test]
    fn test_pow_2() {
        let constraint = make_var(0) ^ 2;
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(144));
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(1156));
        assert_eq!(constraint.to_string(), "w0 ^ 2");
    }

    #[test]
    fn test_pow_3() {
        let constraint = make_var(1) ^ 3;
        assert_eq!(
            constraint.evaluate(&[from_const(0), from_const(12)]),
            from_const(1728)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(0), from_const(34)]),
            from_const(39304)
        );
        assert_eq!(constraint.to_string(), "w1 ^ 3");
    }

    #[test]
    fn test_pow_negative_1() {
        let constraint = make_var(0) ^ -1;
        assert_eq!(
            constraint.evaluate(&[from_const(12)]) * from_const(12),
            from_const(1)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34)]) * from_const(34),
            from_const(1)
        );
        assert_eq!(constraint.to_string(), "w0 ^ -1");
    }

    #[test]
    fn test_pow_negative_2() {
        let constraint = make_var(0) ^ -2;
        let value = from_const(12);
        assert_eq!(constraint.evaluate(&[value]) * value * value, from_const(1));
        assert_eq!(constraint.to_string(), "w0 ^ -2");
    }

    #[test]
    fn test_pow_of_constant() {
        let constraint = Constraint::make_const(from_const(3)) ^ 4;
        assert_eq!(constraint.evaluate(&[]), from_const(81));
        assert_eq!(constraint.to_string(), "81");
    }

    #[test]
    fn test_pow_of_zero_constraint() {
        let constraint = Constraint::nop() ^ 5;
        assert_eq!(constraint.evaluate(&[]), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    #[test]
    #[should_panic(expected = "raising a sum to a power is forbidden")]
    fn test_pow_sum_panics() {
        let _ = (make_var(0) + make_var(1)) ^ 2;
    }

    #[test]
    fn test_compound_pow_1() {
        let mut constraint = make_var(0);
        constraint ^= 2;
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(144));
        assert_eq!(constraint.evaluate(&[from_const(34)]), from_const(1156));
        assert_eq!(constraint.to_string(), "w0 ^ 2");
    }

    #[test]
    fn test_compound_pow_2() {
        let mut constraint = make_var(1);
        constraint ^= 3;
        assert_eq!(
            constraint.evaluate(&[from_const(0), from_const(12)]),
            from_const(1728)
        );
        assert_eq!(constraint.to_string(), "w1 ^ 3");
    }

    #[test]
    fn test_div_1() {
        let constraint = make_var(0) / make_var(1);
        assert_eq!(
            constraint.evaluate(&[from_const(408), from_const(12)]),
            from_const(34)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(408), from_const(34)]),
            from_const(12)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(4368), from_const(56)]),
            from_const(78)
        );
        assert_eq!(constraint.to_string(), "w0 * w1 ^ -1");
    }

    #[test]
    fn test_div_2() {
        let constraint = make_var(1) / make_var(0);
        assert_eq!(
            constraint.evaluate(&[from_const(12), from_const(408)]),
            from_const(34)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(34), from_const(408)]),
            from_const(12)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(56), from_const(4368)]),
            from_const(78)
        );
        assert_eq!(constraint.to_string(), "w0 ^ -1 * w1");
    }

    #[test]
    fn test_div_3() {
        let constraint = make_var(1) / make_var(2);
        assert_eq!(
            constraint.evaluate(&[from_const(0), from_const(1904), from_const(56)]),
            from_const(34)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(0), from_const(1904), from_const(34)]),
            from_const(56)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(0), from_const(672), from_const(56)]),
            from_const(12)
        );
        assert_eq!(constraint.to_string(), "w1 * w2 ^ -1");
    }

    #[test]
    fn test_div_scalar_1() {
        let divisor = from_const(12);
        let constraint = make_var(0) / divisor;
        let expected_coefficient = divisor.invert_vartime().unwrap();
        assert_eq!(constraint.evaluate(&[from_const(408)]), from_const(34));
        assert_eq!(constraint.evaluate(&[from_const(672)]), from_const(56));
        assert_eq!(
            constraint.to_string(),
            format!(
                "{} * w0",
                Constraint::print_coefficient(&expected_coefficient)
            )
        );
    }

    #[test]
    fn test_div_scalar_2() {
        let divisor = from_const(34);
        let constraint = make_var(0) / divisor;
        let expected_coefficient = divisor.invert_vartime().unwrap();
        assert_eq!(constraint.evaluate(&[from_const(408)]), from_const(12));
        assert_eq!(constraint.evaluate(&[from_const(1904)]), from_const(56));
        assert_eq!(
            constraint.to_string(),
            format!(
                "{} * w0",
                Constraint::print_coefficient(&expected_coefficient)
            )
        );
    }

    #[test]
    fn test_div_another_scalar() {
        let constraint = make_var(0) / from_const(34) / from_const(56);
        let expected_coefficient =
            from_const(34).invert_vartime().unwrap() * from_const(56).invert_vartime().unwrap();
        assert_eq!(constraint.evaluate(&[from_const(22848)]), from_const(12));
        assert_eq!(constraint.evaluate(&[from_const(148512)]), from_const(78));
        assert_eq!(
            constraint.to_string(),
            format!(
                "{} * w0",
                Constraint::print_coefficient(&expected_coefficient)
            )
        );
    }

    #[test]
    fn test_div_isize() {
        let constraint = make_var(0) / 4isize;
        let expected_coefficient = Constraint::isize_to_scalar(4).invert_vartime().unwrap();
        assert_eq!(constraint.evaluate(&[from_const(12)]), from_const(3));
        assert_eq!(constraint.evaluate(&[from_const(40)]), from_const(10));
        assert_eq!(
            constraint.to_string(),
            format!(
                "{} * w0",
                Constraint::print_coefficient(&expected_coefficient)
            )
        );
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_div_by_zero_panics() {
        let _ = make_var(0) / Constraint::default();
    }

    #[test]
    #[should_panic(expected = "dividing by a polynomial is forbidden")]
    fn test_div_by_sum_panics() {
        let _ = make_var(0) / (make_var(1) + make_var(2));
    }

    #[test]
    fn test_compound_div_1() {
        let mut constraint = make_var(0);
        constraint /= make_var(1);
        assert_eq!(
            constraint.evaluate(&[from_const(408), from_const(12)]),
            from_const(34)
        );
        assert_eq!(
            constraint.evaluate(&[from_const(408), from_const(34)]),
            from_const(12)
        );
        assert_eq!(constraint.to_string(), "w0 * w1 ^ -1");
    }

    #[test]
    fn test_compound_div_2() {
        let mut constraint = make_var(0);
        constraint /= from_const(12);
        assert_eq!(constraint.evaluate(&[from_const(408)]), from_const(34));
        assert_eq!(constraint.evaluate(&[from_const(672)]), from_const(56));
    }
}
