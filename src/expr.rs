use starkom_bluesky::Scalar;
use starkom_ff::{Field, PrimeField};
use std::collections::BTreeMap;
use std::ops::{Add, BitXor, Div, Mul, Neg, Sub};

/// Represents a PLONK constraint as a sum of monomials.
///
/// Each monomial is in the form `coeff * var0^exp0 * var1^exp1 * ...`, where `coeff` is a constant
/// scalar, the `var` variables are witness columns, and the `exp` variables are constant exponents.
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
    /// Makes a `Constraint` whose expression is a single variable reference.
    ///
    /// `column_index` is the index of the witness column the variable refers to.
    pub(crate) fn make_var(column_index: usize) -> Self {
        Constraint {
            monomials: BTreeMap::from([(BTreeMap::from([(column_index, 1)]), Scalar::ONE)]),
        }
    }

    /// Multiplies two monomials.
    ///
    /// The two monomials have the same layout as the inner maps of [`Self::monomials`]. Note that
    /// the coefficients are missing, they must be handled by the caller.
    fn multiply_variables(
        lhs: BTreeMap<usize, isize>,
        rhs: BTreeMap<usize, isize>,
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
            .into_iter()
            .filter(|(_, exponent)| *exponent != 0)
            .collect()
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
        for (variables, coefficient) in &self.monomials {
            let mut value = *coefficient;
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
        let variables = BTreeMap::default();
        match self.monomials.get_mut(&variables) {
            Some(coefficient) => {
                *coefficient += rhs;
            }
            None => {
                self.monomials.insert(variables, rhs);
            }
        }
        self
    }
}

impl Add<isize> for Constraint {
    type Output = Constraint;

    fn add(self, rhs: isize) -> Self::Output {
        self.add(Self::isize_to_scalar(rhs))
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
        let variables = BTreeMap::default();
        match self.monomials.get_mut(&variables) {
            Some(coefficient) => {
                *coefficient -= rhs;
            }
            None => {
                self.monomials.insert(variables, -rhs);
            }
        }
        self
    }
}

impl Sub<isize> for Constraint {
    type Output = Constraint;

    fn sub(self, rhs: isize) -> Self::Output {
        self.sub(Self::isize_to_scalar(rhs))
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

impl Mul<isize> for Constraint {
    type Output = Constraint;

    fn mul(self, rhs: isize) -> Self::Output {
        self.mul(Self::isize_to_scalar(rhs))
    }
}

impl BitXor<isize> for Constraint {
    type Output = Constraint;

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
    fn bitxor(self, rhs: isize) -> Self::Output {
        match rhs {
            0 => Constraint {
                monomials: BTreeMap::from([(BTreeMap::default(), Scalar::ONE)]),
            },
            1 => self,
            _ => match self.monomials.len() {
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
                                if rhs < 0 {
                                    coefficient
                                        .invert_unwrap()
                                        .pow_small_vartime(rhs.unsigned_abs())
                                } else {
                                    coefficient.pow_small_vartime(rhs as usize)
                                },
                            )
                        })
                        .collect(),
                },
                _ => {
                    panic!("raising a sum to a power is forbidden, try to simplify your constraint")
                }
            },
        }
    }
}

impl Div for Constraint {
    type Output = Constraint;

    /// Multiplies the LHS by the inverse of the RHS, which must have exactly one monomial.
    fn div(self, rhs: Self) -> Self::Output {
        match rhs.monomials.len() {
            0 => panic!("division by zero"),
            1 => self.mul(rhs.bitxor(-1)),
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
        let constraint = Constraint::default();
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

    // TODO
}
