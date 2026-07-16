use crate::lexer;
use crate::parser;
use crate::utils::{is_pseudo_negative, isize_to_scalar};
use crate::witness::Cell;
use starkom_bluesky::Scalar;
use starkom_ff::Field;
use starkom_poly;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display};
use std::ops::{
    Add, AddAssign, BitXor, BitXorAssign, Div, DivAssign, Index, Mul, MulAssign, Neg, Sub,
    SubAssign,
};
use std::str::FromStr;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Short-hand for [`Constraint::make_var`].
#[inline]
pub fn var(column_index: usize) -> Constraint {
    Constraint::make_var(column_index, 0)
}

/// Short-hand for [`Constraint::make_var`] with a rotation.
#[inline]
pub fn rvar(column_index: usize, rotation: isize) -> Constraint {
    Constraint::make_var(column_index, rotation)
}

/// Short-hand for [`Constraint::make_const`].
#[inline]
pub fn make_const(value: isize) -> Constraint {
    Constraint::make_const(isize_to_scalar(value))
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
                root_cell.row() - self.rotation.unsigned_abs()
            } else {
                root_cell.row() + self.rotation.unsigned_abs()
            },
            root_cell.column() + self.column_index,
        )
    }

    /// Remaps the variable to a different column index, as per [`Constraint::remap_variables`].
    pub const fn remap(self, column_offset: usize) -> Self {
        Self {
            column_index: column_offset + self.column_index,
            rotation: self.rotation,
        }
    }

    /// Used by [`Constraint::compose`] when replacing variables with column polynomials.
    fn rotate_column(&self, omega: Scalar, column: Polynomial) -> Polynomial {
        match self.rotation.cmp(&0) {
            Ordering::Less => column.shift_domain_by(
                omega
                    .invert_unwrap()
                    .pow_small(self.rotation.unsigned_abs()),
            ),
            Ordering::Greater => {
                column.shift_domain_by(omega.pow_small(self.rotation.unsigned_abs()))
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
pub struct Constraint {
    /// The outer map represents the monomials in this constraint, while the inner maps represent
    /// the variables in each monomial.
    ///
    /// The values of the inner map are (possibly negative) exponents to which the corresponding
    /// variable is raised.
    ///
    /// The values of the outer map are the constant coefficients of each monomial.
    monomials: BTreeMap<BTreeMap<Variable, isize>, Scalar>,
}

impl Constraint {
    /// Makes a [`Constraint`] whose expression is a constant value.
    pub fn make_const(value: Scalar) -> Self {
        Self {
            monomials: if value != Scalar::ZERO {
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
                Scalar::ONE,
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

    pub fn remap_variables(self, column_offset: usize) -> Self {
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
    fn multiply_variables<I: IntoIterator<Item = (Variable, isize)>>(
        lhs: BTreeMap<Variable, isize>,
        rhs: I,
    ) -> BTreeMap<Variable, isize> {
        let mut result = lhs;
        for (variable, exponent) in rhs {
            match result.get_mut(&variable) {
                Some(preexisting_exponent) => {
                    *preexisting_exponent += exponent;
                }
                None => {
                    result.insert(variable, exponent);
                }
            }
        }
        result
    }

    fn print_coefficient(coefficient: &Scalar) -> String {
        if is_pseudo_negative(coefficient) {
            format!(
                "-{}",
                (Scalar::MAX - coefficient + Scalar::ONE).to_str_radix(10, 0, false)
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
                (*coefficient != Scalar::ONE)
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

    /// Indicates whether this constraint expression can be raised to a power.
    ///
    /// Raising can only be done when the expression is not a sum, otherwise our exponentiation
    /// algorithm panics.
    pub fn can_raise(&self) -> bool {
        self.monomials.len() < 2
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

    pub fn get_variable(&self) -> Option<Variable> {
        let mut maybe_variable = None;
        for (variables, _) in &self.monomials {
            for (variable, _) in variables {
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
    pub fn evaluate<'a, S: Index<&'a Variable, Output = Scalar>>(
        &'a self,
        substitution: &S,
    ) -> Scalar {
        let mut result = Scalar::ZERO;
        for (variables, &coefficient) in &self.monomials {
            let mut monomial_value = coefficient;
            if monomial_value == Scalar::ZERO {
                continue;
            }
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
    pub fn compose(&self, omega: Scalar, substitution: &[Polynomial]) -> Polynomial {
        let columns_by_variable = {
            let mut columns_by_variable = BTreeMap::default();
            for (variables, &coefficient) in &self.monomials {
                if coefficient != Scalar::ZERO {
                    for (&variable, _) in variables {
                        if !columns_by_variable.contains_key(&variable) {
                            columns_by_variable.insert(
                                variable,
                                variable.rotate_column(
                                    omega,
                                    substitution[variable.column_index()].clone(),
                                ),
                            );
                        }
                    }
                }
            }
            columns_by_variable
        };
        let mut result = Polynomial::default();
        for (variables, &coefficient) in &self.monomials {
            if coefficient == Scalar::ZERO {
                continue;
            }
            let mut monomial = Polynomial::constant(coefficient);
            for (variable, &exponent) in variables {
                let column = &columns_by_variable[variable];
                match exponent {
                    0 => {}
                    1 => {
                        monomial *= column.clone();
                    }
                    exponent => {
                        assert!(
                            exponent > 0,
                            "the constraint must be canonicalized before composition"
                        );
                        for _ in 0..exponent {
                            monomial *= column.clone();
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
        write!(f, "Constraint({})", self.format_expression())
    }
}

impl Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_expression())
    }
}

impl From<Variable> for Constraint {
    fn from(value: Variable) -> Self {
        Constraint::make_var(value.column_index(), value.rotation())
    }
}

impl From<Scalar> for Constraint {
    fn from(value: Scalar) -> Self {
        Constraint::make_const(value)
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
        *self += isize_to_scalar(rhs);
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
        self.add(isize_to_scalar(rhs))
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
        *self -= isize_to_scalar(rhs);
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
        *self *= isize_to_scalar(rhs);
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
        *self *= isize_to_scalar(rhs).invert_vartime().unwrap();
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
    use crate::witness::cell;
    use starkom_bluesky::from_const;

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

    fn evaluate<const N: usize>(constraint: &Constraint, substitution: [Scalar; N]) -> Scalar {
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
        let constraint = Constraint::nop();
        assert_eq!(constraint, Constraint::default());
        assert_eq!(evaluate(&constraint, []), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    fn test_constant_impl(value: Scalar) {
        let constraint = Constraint {
            monomials: BTreeMap::from([(BTreeMap::default(), value)]),
        };
        assert_eq!(constraint.get_free_variables(), BTreeSet::default());
        assert!(constraint.is_constant());
        assert_eq!(constraint.get_value_if_constant(), Some(value));
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
        let constraint = var(0);
        assert_eq!(
            constraint.get_free_variables(),
            BTreeSet::from([Variable::new(0, 0)])
        );
        assert!(!constraint.is_constant());
        assert!(constraint.get_value_if_constant().is_none());
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(0)");
    }

    #[test]
    fn test_variable_1() {
        let constraint = var(1);
        assert_eq!(
            constraint.get_free_variables(),
            BTreeSet::from([Variable::new(1, 0)])
        );
        assert!(!constraint.is_constant());
        assert!(constraint.get_value_if_constant().is_none());
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(1)");
    }

    #[test]
    fn test_rotated_variable_1() {
        let constraint = rvar(2, 1);
        assert_eq!(
            constraint.get_free_variables(),
            BTreeSet::from([Variable::new(2, 1)])
        );
        assert!(!constraint.is_constant());
        assert!(constraint.get_value_if_constant().is_none());
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(2,+1)");
    }

    #[test]
    fn test_rotated_variable_2() {
        let constraint = rvar(2, -1);
        assert_eq!(
            constraint.get_free_variables(),
            BTreeSet::from([Variable::new(2, -1)])
        );
        assert!(!constraint.is_constant());
        assert!(constraint.get_value_if_constant().is_none());
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(2,-1)");
    }

    #[test]
    fn test_sum_1() {
        let constraint = var(0) + var(1);
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
        let constraint = var(1) + var(0);
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
        let constraint = rvar(2, -1) + rvar(2, 1);
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
        let constraint = var(0) + var(1) + var(2);
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
        let constraint = var(0) + from_const(12);
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(46));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(68));
        assert_eq!(constraint.to_string(), "12 + var(0)");
    }

    #[test]
    fn test_add_scalar_2() {
        let constraint = var(0) + from_const(34);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(46));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(90));
        assert_eq!(constraint.to_string(), "34 + var(0)");
    }

    #[test]
    fn test_add_another_scalar() {
        let constraint = var(0) + from_const(34) + from_const(56);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(102));
        assert_eq!(evaluate(&constraint, [from_const(78)]), from_const(168));
        assert_eq!(constraint.to_string(), "90 + var(0)");
    }

    #[test]
    fn test_optimize_sum_1() {
        let constraint = var(0) + var(0) * -from_const(1);
        assert_eq!(evaluate(&constraint, []), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    #[test]
    fn test_optimize_sum_2() {
        let constraint = var(0) + var(1) * -from_const(1);
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
        let constraint = var(0) + var(1) * -from_const(1) + var(0) * -from_const(1);
        assert_eq!(evaluate(&constraint, [from_const(12)]), -from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), -from_const(34));
        assert_eq!(constraint.to_string(), "-1 * var(1)");
    }

    #[test]
    fn test_negate_variable() {
        let constraint = -var(0);
        assert_eq!(evaluate(&constraint, [from_const(12)]), -from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), -from_const(34));
        assert_eq!(constraint.to_string(), "-1 * var(0)");
    }

    #[test]
    fn test_negate_sum() {
        let constraint = -(var(0) + var(1) + from_const(12));
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
        let constraint = var(0) - var(1);
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
        let constraint = var(1) - var(0);
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
        let constraint = rvar(2, -1) - rvar(2, 1);
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
        let constraint = var(0) - var(1) - var(2);
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
        let constraint = var(0) - from_const(12);
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(22));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(44));
        assert_eq!(constraint.to_string(), "-12 + var(0)");
    }

    #[test]
    fn test_sub_scalar_2() {
        let constraint = var(0) - from_const(34);
        assert_eq!(evaluate(&constraint, [from_const(12)]), -from_const(22));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(22));
        assert_eq!(constraint.to_string(), "-34 + var(0)");
    }

    #[test]
    fn test_sub_another_scalar() {
        let constraint = var(0) - from_const(34) - from_const(56);
        assert_eq!(evaluate(&constraint, [from_const(12)]), -from_const(78));
        assert_eq!(evaluate(&constraint, [from_const(78)]), -from_const(12));
        assert_eq!(constraint.to_string(), "-90 + var(0)");
    }

    #[test]
    fn test_optimize_diff_1() {
        let constraint = var(0) - var(0);
        assert_eq!(evaluate(&constraint, []), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    #[test]
    fn test_optimize_diff_2() {
        let constraint = var(0) - var(1) * -from_const(1);
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
        let constraint = var(0) - var(1) - var(0);
        assert_eq!(evaluate(&constraint, [from_const(12)]), -from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), -from_const(34));
        assert_eq!(constraint.to_string(), "-1 * var(1)");
    }

    #[test]
    fn test_product_1() {
        let constraint = var(0) * var(1);
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
        let constraint = var(1) * var(0);
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
        let constraint = rvar(2, -1) * rvar(2, 1);
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
        let constraint = var(0) * var(1) * var(2);
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
        let constraint = var(0) * var(0);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(144));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(1156));
        assert_eq!(constraint.to_string(), "var(0) ^ 2");
    }

    #[test]
    fn test_mul_scalar_1() {
        let constraint = var(0) * from_const(12);
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(408));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(672));
        assert_eq!(constraint.to_string(), "12 * var(0)");
    }

    #[test]
    fn test_mul_scalar_2() {
        let constraint = var(0) * from_const(34);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(408));
        assert_eq!(evaluate(&constraint, [from_const(56)]), from_const(1904));
        assert_eq!(constraint.to_string(), "34 * var(0)");
    }

    #[test]
    fn test_mul_another_scalar() {
        let constraint = var(0) * from_const(34) * from_const(56);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(22848));
        assert_eq!(evaluate(&constraint, [from_const(78)]), from_const(148512));
        assert_eq!(constraint.to_string(), "1904 * var(0)");
    }

    #[test]
    fn test_mul_by_zero() {
        let constraint = var(0) * from_const(0);
        assert_eq!(evaluate(&constraint, []), from_const(0));
        assert_eq!(constraint.to_string(), "0");
    }

    #[test]
    fn test_optimize_product_1() {
        let constraint = var(0) * (var(0) ^ -1);
        assert_eq!(evaluate(&constraint, []), from_const(1));
        assert_eq!(constraint.to_string(), "1");
    }

    #[test]
    fn test_optimize_product_2() {
        let constraint = var(0) * (var(0) ^ -1) * var(1);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(1)");
    }

    #[test]
    fn test_optimize_product_3() {
        let constraint = (var(0) ^ 2) * (var(0) ^ -1);
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(0)");
    }

    #[test]
    fn test_pow_zero_exponent() {
        let constraint = var(0) ^ 0;
        assert_eq!(evaluate(&constraint, []), from_const(1));
        assert_eq!(constraint.to_string(), "1");
    }

    #[test]
    fn test_pow_zero_exponent_on_sum() {
        let constraint = (var(0) + var(1)) ^ 0;
        assert_eq!(evaluate(&constraint, []), from_const(1));
        assert_eq!(constraint.to_string(), "1");
    }

    #[test]
    fn test_pow_zero_exponent_of_zero() {
        let constraint = make_const(0) ^ 0;
        assert_eq!(evaluate(&constraint, []), from_const(1));
        assert_eq!(constraint.to_string(), "1");
    }

    #[test]
    fn test_pow_one_exponent() {
        let constraint = var(0) ^ 1;
        assert_eq!(evaluate(&constraint, [from_const(12)]), from_const(12));
        assert_eq!(evaluate(&constraint, [from_const(34)]), from_const(34));
        assert_eq!(constraint.to_string(), "var(0)");
    }

    #[test]
    fn test_pow_one_exponent_on_sum() {
        let constraint = (var(0) + var(1)) ^ 1;
        assert_eq!(
            evaluate(&constraint, [from_const(12), from_const(34)]),
            from_const(46)
        );
        assert_eq!(constraint.to_string(), "var(0) + var(1)");
    }

    #[test]
    fn test_pow_positive_exponent() {
        let constraint = var(0) ^ 3;
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
        let constraint = var(0) ^ -2;
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
        let constraint = make_const(2) ^ 3;
        assert_eq!(evaluate(&constraint, []), from_const(8));
        assert_eq!(constraint.to_string(), "8");
    }

    #[test]
    fn test_pow_constant_negative_exponent() {
        let constraint = make_const(2) ^ -1;
        assert_eq!(evaluate(&constraint, []), from_const(2).invert_unwrap());
    }

    #[test]
    fn test_bitxor_assign() {
        let mut constraint = var(0);
        constraint ^= 3;
        assert_eq!(
            evaluate(&constraint, [from_const(12)]),
            from_const(12).pow_small_vartime(3)
        );
        assert_eq!(constraint.to_string(), "var(0) ^ 3");
    }

    #[test]
    #[should_panic(expected = "raising a sum to a power is forbidden")]
    fn test_pow_sum_panics() {
        let _ = (var(0) + var(1)) ^ 2;
    }

    #[test]
    #[should_panic(expected = "cannot raise 0 to a negative power")]
    fn test_pow_zero_to_negative_exponent_panics_1() {
        let _ = make_const(0) ^ -1;
    }

    #[test]
    #[should_panic(expected = "cannot raise 0 to a negative power")]
    fn test_pow_zero_to_negative_exponent_panics_2() {
        let _ = make_const(0) ^ -2;
    }

    #[test]
    fn test_parsing() {
        let c1: Constraint = "var(0) ^ 2 * var(1) + var(0) + 5 == 35".parse().unwrap();
        let c2 = (var(0) ^ 2) * var(1) + var(0) - 30;
        let c3: Constraint = format!("{} == 0", c2).parse().unwrap();
        assert_eq!(c1, c2);
        assert_eq!(c1, c3);
    }
}
