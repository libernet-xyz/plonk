use crate::expr::Constraint;
use crate::lexer::Token;
use crate::utils::{is_pseudo_negative, scalar_to_isize};
use anyhow::{Result, anyhow};
use starkom_ff::Field;
use std::fmt::Debug;
use std::marker::PhantomData;

static WITNESS_FUNCTION_NAME: &'static str = "var";

/// A recursive descent parser for Starkom's expression syntax.
#[derive(Debug, Clone)]
struct Parser<'a, F: Field> {
    tokens: &'a [Token],
    _data: PhantomData<F>,
}

impl<'a, F: Field> Parser<'a, F> {
    fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens,
            _data: PhantomData,
        }
    }

    fn peek_token(&mut self) -> Result<Token> {
        if self.tokens.is_empty() {
            Err(anyhow!("unexpected end of input"))
        } else {
            Ok(self.tokens[0].clone())
        }
    }

    fn next_token(&mut self) {
        self.tokens = &self.tokens[1..];
    }

    fn consume_token(&mut self) -> Result<Token> {
        let token = self.peek_token()?;
        self.next_token();
        Ok(token)
    }

    fn skip_token(&mut self, token: Token) -> Result<()> {
        if self.peek_token()? != token {
            return Err(anyhow!("syntax error"));
        }
        self.next_token();
        Ok(())
    }

    fn parse_variable(&mut self) -> Result<Constraint<F>> {
        self.skip_token(Token::LeftBracket)?;
        let column_index = {
            let column_index_expression = self.parse_sum()?;
            match column_index_expression.get_constant_value() {
                Some(column_index) => {
                    let column_index = scalar_to_isize(column_index)?;
                    if column_index < 0 {
                        Err(anyhow!(
                            "invalid witness column index {}: must be positive",
                            column_index
                        ))
                    } else {
                        Ok(column_index.unsigned_abs())
                    }
                }
                None => Err(anyhow!(
                    "invalid column index `{}`: must be a constant",
                    column_index_expression
                )),
            }
        }?;
        let rotation = match self.peek_token()? {
            Token::Comma => {
                self.next_token();
                let negative = match self.consume_token()? {
                    Token::Plus => Ok(false),
                    Token::Minus => Ok(true),
                    _ => Err(anyhow!("syntax error")),
                }?;
                let rotation = match self.consume_token()? {
                    Token::Number10(value) => {
                        let value: F = F::from_str_radix(value.as_str(), 10).unwrap();
                        if is_pseudo_negative(&value) {
                            return Err(anyhow!(
                                "invalid rotation value {}: must be a small number!",
                                value
                            ));
                        }
                        match scalar_to_isize(value) {
                            Ok(value) => Ok(value),
                            Err(_) => Err(anyhow!(
                                "invalid rotation value {}: must be a small number!",
                                value
                            )),
                        }
                    }
                    _ => Err(anyhow!("syntax error")),
                }?;
                if negative { -rotation } else { rotation }
            }
            _ => 0,
        };
        self.skip_token(Token::RightBracket)?;
        Ok(Constraint::make_var(column_index, rotation))
    }

    fn parse_leaf(&mut self) -> Result<Constraint<F>> {
        match self.consume_token()? {
            Token::Number2(value) => Ok(Constraint::make_const(
                F::from_str_radix(value.as_str(), 2).unwrap(),
            )),
            Token::Number8(value) => Ok(Constraint::make_const(
                F::from_str_radix(value.as_str(), 8).unwrap(),
            )),
            Token::Number10(value) => Ok(Constraint::make_const(
                F::from_str_radix(value.as_str(), 10).unwrap(),
            )),
            Token::Number16(value) => Ok(Constraint::make_const(
                F::from_str_radix(value.as_str(), 16).unwrap(),
            )),
            Token::Identifier(label) => {
                if label != WITNESS_FUNCTION_NAME {
                    return Err(anyhow!("unknown identifier `{}`", label));
                }
                self.parse_variable()
            }
            Token::LeftBracket => {
                let inner = self.parse_sum()?;
                self.skip_token(Token::RightBracket)?;
                Ok(inner)
            }
            _ => Err(anyhow!("syntax error")),
        }
    }

    fn parse_unary_expression(&mut self) -> Result<Constraint<F>> {
        match self.peek_token()? {
            Token::Plus => {
                self.next_token();
                self.parse_unary_expression()
            }
            Token::Minus => {
                self.next_token();
                Ok(-self.parse_unary_expression()?)
            }
            _ => self.parse_leaf(),
        }
    }

    fn parse_exponent(&mut self) -> Result<isize> {
        match self.parse_unary_expression()?.get_constant_value() {
            Some(value) => scalar_to_isize(value),
            None => Err(anyhow!("exponents may not contain variables")),
        }
    }

    fn parse_power(&mut self) -> Result<Constraint<F>> {
        let mut base = self.parse_unary_expression()?;
        loop {
            match self.peek_token()? {
                Token::Power => {
                    self.next_token();
                    let exponent = self.parse_exponent()?;
                    if !base.can_raise_to(exponent) {
                        return Err(anyhow!(
                            "expression `{}` cannot be raised to a negative power",
                            base
                        ));
                    }
                    base ^= exponent;
                }
                _ => {
                    return Ok(base);
                }
            }
        }
    }

    fn parse_product(&mut self) -> Result<Constraint<F>> {
        let mut operand = self.parse_power()?;
        loop {
            match self.peek_token()? {
                Token::Multiply => {
                    self.next_token();
                    operand *= self.parse_power()?;
                }
                Token::Divide => {
                    self.next_token();
                    operand /= self.parse_power()?;
                }
                _ => {
                    return Ok(operand);
                }
            }
        }
    }

    fn parse_sum(&mut self) -> Result<Constraint<F>> {
        let mut operand = self.parse_product()?;
        loop {
            match self.peek_token()? {
                Token::Plus => {
                    self.next_token();
                    operand += self.parse_product()?;
                }
                Token::Minus => {
                    self.next_token();
                    operand -= self.parse_product()?;
                }
                _ => {
                    return Ok(operand);
                }
            }
        }
    }

    fn parse_equality(&mut self) -> Result<Constraint<F>> {
        let lhs = self.parse_sum()?;
        self.skip_token(Token::Equal)?;
        let rhs = self.parse_sum()?;
        Ok(lhs - rhs)
    }

    fn parse(mut self) -> Result<Constraint<F>> {
        let constraint = self.parse_equality()?;
        self.skip_token(Token::EndOfInput)?;
        Ok(constraint)
    }
}

/// Parses a (tokenized) expression in Starkom's expression syntax.
pub(crate) fn parse<F: Field>(tokens: &[Token]) -> Result<Constraint<F>> {
    Parser::new(tokens).parse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{make_const, rvar, var};
    use crate::lexer;
    use starkom_bluesky::Scalar as BS;

    fn parse<F: Field>(s: &'static str) -> Constraint<F> {
        let tokens = lexer::tokenize(s).unwrap();
        super::parse(tokens.as_slice()).unwrap()
    }

    #[inline]
    fn nop<F: Field>() -> Constraint<F> {
        Constraint::nop()
    }

    #[test]
    fn test_constants() {
        assert_eq!(parse::<BS>("0 == 0"), make_const(0));
        assert_eq!(parse::<BS>("12 == 0"), make_const(12));
        assert_eq!(parse::<BS>("56 == 34"), make_const(22));
    }

    #[test]
    fn test_variables() {
        assert_eq!(parse::<BS>("var(0) == 0"), var(0));
        assert_eq!(parse::<BS>("var(1) == 0"), var(1));
        assert_eq!(parse::<BS>("var(2) == 0"), var(2));
        assert_eq!(parse::<BS>("var(12, +0) == 0"), var(12));
        assert_eq!(parse::<BS>("var(34, +1) == 0"), rvar(34, 1));
        assert_eq!(parse::<BS>("var(56, -1) == 0"), rvar(56, -1));
        assert_eq!(parse::<BS>("var(78, +2) == 0"), rvar(78, 2));
        assert_eq!(parse::<BS>("var(90, -2) == 0"), rvar(90, -2));
    }

    #[test]
    fn test_unary() {
        assert_eq!(parse::<BS>("-var(0) == 0"), -var(0));
        assert_eq!(parse::<BS>("--var(0) == 0"), var(0));
        assert_eq!(parse::<BS>("---var(0) == 0"), -var(0));
        assert_eq!(parse::<BS>("+var(0) == 0"), var(0));
        assert_eq!(parse::<BS>("++var(0) == 0"), var(0));
        assert_eq!(parse::<BS>("+-var(0) == 0"), -var(0));
        assert_eq!(parse::<BS>("-+var(0) == 0"), -var(0));
        assert_eq!(parse::<BS>("-++var(0) == 0"), -var(0));
        assert_eq!(parse::<BS>("+-+var(0) == 0"), -var(0));
        assert_eq!(parse::<BS>("++-var(0) == 0"), -var(0));
        assert_eq!(parse::<BS>("+-+-+var(0) == 0"), var(0));
        assert_eq!(parse::<BS>("-+-+-var(0) == 0"), -var(0));
    }

    #[test]
    fn test_sum() {
        assert_eq!(parse::<BS>("var(0) + var(0) == 0"), var(0) * 2);
        assert_eq!(parse::<BS>("var(0) + var(1) == 0"), var(0) + var(1));
        assert_eq!(parse::<BS>("var(1) + var(0) == 0"), var(0) + var(1));
        assert_eq!(parse::<BS>("var(0) + 42 == 0"), var(0) + make_const(42));
        assert_eq!(parse::<BS>("42 + var(0) == 0"), var(0) + make_const(42));
        assert_eq!(
            parse::<BS>("var(0) + var(1) + var(2) == 0"),
            var(0) + var(1) + var(2)
        );
        assert_eq!(
            parse::<BS>("var(2) + var(2) + var(3) == 0"),
            var(2) * 2 + var(3)
        );
        assert_eq!(
            parse::<BS>("var(3) + var(2) + var(3) == 0"),
            var(3) * 2 + var(2)
        );
        assert_eq!(parse::<BS>("var(4) + var(4) + var(4) == 0"), var(4) * 3);
    }

    #[test]
    fn test_subtraction() {
        assert_eq!(parse::<BS>("var(0) - var(0) == 0"), nop());
        assert_eq!(parse::<BS>("var(0) - var(1) == 0"), var(0) - var(1));
        assert_eq!(parse::<BS>("var(1) - var(0) == 0"), var(1) - var(0));
        assert_eq!(parse::<BS>("var(0) - 42 == 0"), var(0) - make_const(42));
        assert_eq!(parse::<BS>("42 - var(0) == 0"), make_const(42) - var(0));
        assert_eq!(
            parse::<BS>("var(0) - var(1) - var(2) == 0"),
            var(0) - var(1) - var(2)
        );
        assert_eq!(parse::<BS>("var(2) - var(2) - var(3) == 0"), -var(3));
        assert_eq!(parse::<BS>("var(3) - var(2) - var(3) == 0"), -var(2));
        assert_eq!(parse::<BS>("var(4) - var(4) - var(4) == 0"), -var(4));
    }

    #[test]
    fn test_product() {
        assert_eq!(parse::<BS>("var(0) * var(0) == 0"), var(0) ^ 2);
        assert_eq!(parse::<BS>("var(0) * var(1) == 0"), var(0) * var(1));
        assert_eq!(parse::<BS>("var(1) * var(0) == 0"), var(0) * var(1));
        assert_eq!(parse::<BS>("var(0) * 42 == 0"), var(0) * make_const(42));
        assert_eq!(parse::<BS>("42 * var(0) == 0"), var(0) * make_const(42));
        assert_eq!(
            parse::<BS>("var(0) * var(1) * var(2) == 0"),
            var(0) * var(1) * var(2)
        );
        assert_eq!(
            parse::<BS>("var(2) * var(2) * var(3) == 0"),
            (var(2) ^ 2) * var(3)
        );
        assert_eq!(
            parse::<BS>("var(3) * var(2) * var(3) == 0"),
            (var(3) ^ 2) * var(2)
        );
        assert_eq!(parse::<BS>("var(4) * var(4) * var(4) == 0"), var(4) ^ 3);
    }

    #[test]
    fn test_power() {
        assert_eq!(parse::<BS>("var(0) ^ 0 == 0"), make_const(1));
        assert_eq!(parse::<BS>("var(0) ^ 1 == 0"), var(0));
        assert_eq!(parse::<BS>("var(0) ^ 2 == 0"), var(0) ^ 2);
        assert_eq!(parse::<BS>("var(0) ^ +0 == 0"), make_const(1));
        assert_eq!(parse::<BS>("var(0) ^ +1 == 0"), var(0));
        assert_eq!(parse::<BS>("var(0) ^ +2 == 0"), var(0) ^ 2);
        assert_eq!(parse::<BS>("var(0) ^ -0 == 0"), make_const(1));
        assert_eq!(parse::<BS>("var(0) ^ -1 == 0"), var(0) ^ -1);
        assert_eq!(parse::<BS>("var(0) ^ -2 == 0"), var(0) ^ -2);
    }

    #[test]
    fn test_power_sum_positive_exponent() {
        assert_eq!(
            parse::<BS>("(var(0) + var(1)) ^ 2 == 0"),
            (var(0) + var(1)) ^ 2
        );
        assert_eq!(
            parse::<BS>("(var(0) + var(1)) ^ 3 == 0"),
            (var(0) + var(1)) ^ 3
        );
    }

    #[test]
    #[should_panic(expected = "cannot be raised to a negative power")]
    fn test_power_sum_negative_exponent_panics() {
        let _ = parse::<BS>("(var(0) + var(1)) ^ -2 == 0");
    }

    #[test]
    fn test_division() {
        assert_eq!(parse::<BS>("var(0) / var(0) == 0"), make_const(1));
        assert_eq!(parse::<BS>("var(0) / var(1) == 0"), var(0) / var(1));
        assert_eq!(parse::<BS>("var(1) / var(0) == 0"), var(1) / var(0));
        assert_eq!(parse::<BS>("var(0) / 42 == 0"), var(0) / make_const(42));
        assert_eq!(parse::<BS>("42 / var(0) == 0"), make_const(42) / var(0));
        assert_eq!(
            parse::<BS>("var(0) / var(1) / var(2) == 0"),
            var(0) / var(1) / var(2)
        );
        assert_eq!(parse::<BS>("var(2) / var(2) / var(3) == 0"), var(3) ^ -1);
        assert_eq!(parse::<BS>("var(3) / var(2) / var(3) == 0"), var(2) ^ -1);
        assert_eq!(parse::<BS>("var(4) / var(4) / var(4) == 0"), var(4) ^ -1);
    }

    #[test]
    fn test_brackets() {
        assert_eq!(parse::<BS>("(42) == 0"), make_const(42));
        assert_eq!(parse::<BS>("(var(0)) == 0"), var(0));
        assert_eq!(parse::<BS>("(var(1)) == 0"), var(1));
        assert_eq!(parse::<BS>("(var(1) + var(2)) == 0"), var(1) + var(2));
        assert_eq!(
            parse::<BS>("(var(1) + var(2)) * var(3) == 0"),
            (var(1) + var(2)) * var(3)
        );
        assert_eq!(
            parse::<BS>("var(3) * (var(2) + var(1)) == 0"),
            (var(1) + var(2)) * var(3)
        );
        assert_eq!(parse::<BS>("var(0) ^ (36 + 2 * 3) == 0"), var(0) ^ 42);
        assert_eq!(parse::<BS>("var(0) ^ -(4 - 2) == 0"), var(0) ^ -2);
    }

    #[test]
    fn test_equality() {
        assert_eq!(
            parse::<BS>("(var(0) + var(1)) * var(2) == 42"),
            (var(0) + var(1)) * var(2) - make_const(42)
        );
        assert_eq!(
            parse::<BS>("42 == (var(0) + var(1)) * var(2)"),
            make_const(42) - (var(0) + var(1)) * var(2)
        );
        assert_eq!(
            parse::<BS>("42 * var(2) ^ -1 == var(0) + var(1)"),
            make_const(42) * (var(2) ^ -1) - var(0) - var(1)
        );
    }
}
