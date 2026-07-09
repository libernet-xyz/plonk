use crate::Constraint;
use crate::lexer::Token;
use anyhow::{Result, anyhow};
use regex::Regex;
use starkom_bluesky::Scalar;
use starkom_ff::{Field, Field256};
use std::sync::LazyLock;

static REGEX_VARIABLE_NAME: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^w(\d+)$").unwrap());

/// A recursive descent parser for Starkom's expression syntax.
#[derive(Debug, Clone)]
struct Parser<'a> {
    tokens: &'a [Token],
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self { tokens }
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

    fn parse_leaf(&mut self) -> Result<Constraint> {
        match self.consume_token()? {
            Token::Number(value) => Ok(Constraint::make_const(value)),
            Token::Variable(label) => match REGEX_VARIABLE_NAME.captures(label.as_str()) {
                Some(captures) => Ok(Constraint::make_var(captures[1].parse()?)),
                None => Err(anyhow!(
                    "invalid witness column name: `{}` -- all columns are named `w` followed by a number, e.g. `w0`, `w1`, `w2`, ...",
                    label
                )),
            },
            Token::LeftBracket => {
                let inner = self.parse_sum()?;
                self.skip_token(Token::RightBracket)?;
                Ok(inner)
            }
            _ => Err(anyhow!("syntax error")),
        }
    }

    fn parse_unary_expression(&mut self) -> Result<Constraint> {
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

    fn is_pseudo_negative(value: &Scalar) -> bool {
        const HALF_RANGE: LazyLock<Scalar> = LazyLock::new(|| {
            "0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c00000000000000"
                .parse()
                .unwrap()
        });
        *value > *HALF_RANGE
    }

    fn parse_exponent(&mut self) -> Result<isize> {
        match self.parse_unary_expression()?.get_value_if_constant() {
            Some(value) => {
                const MAX: Scalar = Scalar::from_const(isize::MAX as u64);
                if Self::is_pseudo_negative(&value) {
                    let abs = (Scalar::MAX - value + Scalar::ONE).try_to_u128().unwrap() as i128;
                    if abs > -(isize::MIN as i128) {
                        Err(anyhow!("exponent {} is out of range", value))
                    } else {
                        Ok(-abs as isize)
                    }
                } else {
                    if value > MAX {
                        Err(anyhow!("exponent {} is out of range", value))
                    } else {
                        Ok(value.try_to_u64().unwrap() as isize)
                    }
                }
            }
            None => Err(anyhow!("exponents may not contain variables")),
        }
    }

    fn parse_power(&mut self) -> Result<Constraint> {
        let mut base = self.parse_unary_expression()?;
        loop {
            match self.peek_token()? {
                Token::Power => {
                    if !base.can_raise() {
                        return Err(anyhow!("expression `{}` cannot be raised", base));
                    }
                    self.next_token();
                    base ^= self.parse_exponent()?;
                }
                _ => {
                    return Ok(base);
                }
            }
        }
    }

    fn parse_product(&mut self) -> Result<Constraint> {
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

    fn parse_sum(&mut self) -> Result<Constraint> {
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

    fn parse_equality(&mut self) -> Result<Constraint> {
        let lhs = self.parse_sum()?;
        self.skip_token(Token::Equal)?;
        let rhs = self.parse_sum()?;
        Ok(lhs - rhs)
    }

    fn parse(mut self) -> Result<Constraint> {
        let constraint = self.parse_equality()?;
        self.skip_token(Token::EndOfInput)?;
        Ok(constraint)
    }
}

/// Parses a (tokenized) expression in Starkom's expression syntax.
pub(crate) fn parse(tokens: &[Token]) -> Result<Constraint> {
    Parser::new(tokens).parse()
}

#[cfg(test)]
mod tests {
    use crate::Constraint;
    use crate::lexer;
    use starkom_bluesky::Scalar;

    fn parse(s: &'static str) -> Constraint {
        let tokens = lexer::tokenize(s).unwrap();
        super::parse(tokens.as_slice()).unwrap()
    }

    #[inline(always)]
    fn nop() -> Constraint {
        Constraint::nop()
    }

    fn make_const(value: u64) -> Constraint {
        Constraint::make_const(Scalar::from_const(value))
    }

    #[inline(always)]
    fn var(column_index: usize) -> Constraint {
        Constraint::make_var(column_index)
    }

    #[test]
    fn test_constants() {
        assert_eq!(parse("0 == 0"), make_const(0));
        assert_eq!(parse("12 == 0"), make_const(12));
        assert_eq!(parse("56 == 34"), make_const(22));
    }

    #[test]
    fn test_variables() {
        assert_eq!(parse("w0 == 0"), var(0));
        assert_eq!(parse("w1 == 0"), var(1));
        assert_eq!(parse("w2 == 0"), var(2));
    }

    #[test]
    fn test_unary() {
        assert_eq!(parse("-w0 == 0"), -var(0));
        assert_eq!(parse("--w0 == 0"), var(0));
        assert_eq!(parse("---w0 == 0"), -var(0));
        assert_eq!(parse("+w0 == 0"), var(0));
        assert_eq!(parse("++w0 == 0"), var(0));
        assert_eq!(parse("+-w0 == 0"), -var(0));
        assert_eq!(parse("-+w0 == 0"), -var(0));
        assert_eq!(parse("-++w0 == 0"), -var(0));
        assert_eq!(parse("+-+w0 == 0"), -var(0));
        assert_eq!(parse("++-w0 == 0"), -var(0));
        assert_eq!(parse("+-+-+w0 == 0"), var(0));
        assert_eq!(parse("-+-+-w0 == 0"), -var(0));
    }

    #[test]
    fn test_sum() {
        assert_eq!(parse("w0 + w0 == 0"), var(0) * 2);
        assert_eq!(parse("w0 + w1 == 0"), var(0) + var(1));
        assert_eq!(parse("w1 + w0 == 0"), var(0) + var(1));
        assert_eq!(parse("w0 + 42 == 0"), var(0) + make_const(42));
        assert_eq!(parse("42 + w0 == 0"), var(0) + make_const(42));
        assert_eq!(parse("w0 + w1 + w2 == 0"), var(0) + var(1) + var(2));
        assert_eq!(parse("w2 + w2 + w3 == 0"), var(2) * 2 + var(3));
        assert_eq!(parse("w3 + w2 + w3 == 0"), var(3) * 2 + var(2));
        assert_eq!(parse("w4 + w4 + w4 == 0"), var(4) * 3);
    }

    #[test]
    fn test_subtraction() {
        assert_eq!(parse("w0 - w0 == 0"), nop());
        assert_eq!(parse("w0 - w1 == 0"), var(0) - var(1));
        assert_eq!(parse("w1 - w0 == 0"), var(1) - var(0));
        assert_eq!(parse("w0 - 42 == 0"), var(0) - make_const(42));
        assert_eq!(parse("42 - w0 == 0"), make_const(42) - var(0));
        assert_eq!(parse("w0 - w1 - w2 == 0"), var(0) - var(1) - var(2));
        assert_eq!(parse("w2 - w2 - w3 == 0"), -var(3));
        assert_eq!(parse("w3 - w2 - w3 == 0"), -var(2));
        assert_eq!(parse("w4 - w4 - w4 == 0"), -var(4));
    }

    #[test]
    fn test_product() {
        assert_eq!(parse("w0 * w0 == 0"), var(0) ^ 2);
        assert_eq!(parse("w0 * w1 == 0"), var(0) * var(1));
        assert_eq!(parse("w1 * w0 == 0"), var(0) * var(1));
        assert_eq!(parse("w0 * 42 == 0"), var(0) * make_const(42));
        assert_eq!(parse("42 * w0 == 0"), var(0) * make_const(42));
        assert_eq!(parse("w0 * w1 * w2 == 0"), var(0) * var(1) * var(2));
        assert_eq!(parse("w2 * w2 * w3 == 0"), (var(2) ^ 2) * var(3));
        assert_eq!(parse("w3 * w2 * w3 == 0"), (var(3) ^ 2) * var(2));
        assert_eq!(parse("w4 * w4 * w4 == 0"), var(4) ^ 3);
    }

    #[test]
    fn test_power() {
        assert_eq!(parse("w0 ^ 0 == 0"), make_const(1));
        assert_eq!(parse("w0 ^ 1 == 0"), var(0));
        assert_eq!(parse("w0 ^ 2 == 0"), var(0) ^ 2);
        assert_eq!(parse("w0 ^ +0 == 0"), make_const(1));
        assert_eq!(parse("w0 ^ +1 == 0"), var(0));
        assert_eq!(parse("w0 ^ +2 == 0"), var(0) ^ 2);
        assert_eq!(parse("w0 ^ -0 == 0"), make_const(1));
        assert_eq!(parse("w0 ^ -1 == 0"), var(0) ^ -1);
        assert_eq!(parse("w0 ^ -2 == 0"), var(0) ^ -2);
    }

    #[test]
    fn test_division() {
        assert_eq!(parse("w0 / w0 == 0"), make_const(1));
        assert_eq!(parse("w0 / w1 == 0"), var(0) / var(1));
        assert_eq!(parse("w1 / w0 == 0"), var(1) / var(0));
        assert_eq!(parse("w0 / 42 == 0"), var(0) / make_const(42));
        assert_eq!(parse("42 / w0 == 0"), make_const(42) / var(0));
        assert_eq!(parse("w0 / w1 / w2 == 0"), var(0) / var(1) / var(2));
        assert_eq!(parse("w2 / w2 / w3 == 0"), var(3) ^ -1);
        assert_eq!(parse("w3 / w2 / w3 == 0"), var(2) ^ -1);
        assert_eq!(parse("w4 / w4 / w4 == 0"), var(4) ^ -1);
    }

    #[test]
    fn test_brackets() {
        assert_eq!(parse("(42) == 0"), make_const(42));
        assert_eq!(parse("(w0) == 0"), var(0));
        assert_eq!(parse("(w1) == 0"), var(1));
        assert_eq!(parse("(w1 + w2) == 0"), var(1) + var(2));
        assert_eq!(parse("(w1 + w2) * w3 == 0"), (var(1) + var(2)) * var(3));
        assert_eq!(parse("w3 * (w2 + w1) == 0"), (var(1) + var(2)) * var(3));
        assert_eq!(parse("w0 ^ (36 + 2 * 3) == 0"), var(0) ^ 42);
        assert_eq!(parse("w0 ^ -(4 - 2) == 0"), var(0) ^ -2);
    }

    #[test]
    fn test_equality() {
        assert_eq!(
            parse("(w0 + w1) * w2 == 42"),
            (var(0) + var(1)) * var(2) - make_const(42)
        );
        assert_eq!(
            parse("42 == (w0 + w1) * w2"),
            make_const(42) - (var(0) + var(1)) * var(2)
        );
        assert_eq!(
            parse("42 * w2 ^ -1 == w0 + w1"),
            make_const(42) * (var(2) ^ -1) - var(0) - var(1)
        );
    }
}
