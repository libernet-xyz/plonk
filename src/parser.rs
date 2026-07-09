use crate::Constraint;
use crate::lexer::Token;
use anyhow::{Result, anyhow};
use regex::Regex;
use starkom_bluesky::Scalar;
use starkom_ff::Field256;
use std::sync::LazyLock;

static REGEX_VARIABLE_NAME: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^w(\d+)$").unwrap());

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

    fn parse_exponent(&mut self) -> Result<isize> {
        match self.parse_leaf()?.get_value_if_constant() {
            Some(value) => {
                const MAX: Scalar = Scalar::from_const(isize::MAX as u64);
                if value > MAX {
                    Err(anyhow!("exponent {} is out of range", value))
                } else {
                    Ok(value.try_to_u64().unwrap() as isize)
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

pub(crate) fn parse(tokens: &[Token]) -> Result<Constraint> {
    Parser::new(tokens).parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
