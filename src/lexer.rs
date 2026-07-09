use anyhow::{Result, anyhow};
use regex::{Captures, Regex};
use starkom_bluesky::Scalar;
use std::{collections::BTreeMap, sync::LazyLock};

/// Lexical tokens for Starkom's expression syntax.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Token {
    Number(Scalar),
    Variable(String),
    Plus,
    Minus,
    Multiply,
    Divide,
    Power,
    Equal,
    LeftBracket,
    RightBracket,
    EndOfInput,
}

static REGEX_WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s+").unwrap());

static REGEX_IDENTIFIER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*\b").unwrap());

static REGEX_NUMBER_2: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^0[Bb][01]+\b").unwrap());
static REGEX_NUMBER_8: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^0[0-7]+\b").unwrap());
static REGEX_NUMBER_10: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:0|[1-9]\d*)\b").unwrap());
static REGEX_NUMBER_16: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^0[Xx][0-9a-fA-F]+\b").unwrap());

static SYMBOLS: LazyLock<BTreeMap<&'static str, Token>> = LazyLock::new(|| {
    BTreeMap::from([
        ("+", Token::Plus),
        ("-", Token::Minus),
        ("*", Token::Multiply),
        ("/", Token::Divide),
        ("^", Token::Power),
        ("==", Token::Equal),
        ("(", Token::LeftBracket),
        (")", Token::RightBracket),
    ])
});

#[derive(Debug, Clone)]
struct Lexer<'a> {
    input: &'a str,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self { input }
    }

    fn consume_prefix(&mut self, pattern: &Regex) -> Option<Captures<'a>> {
        match pattern.captures(self.input) {
            Some(captures) => {
                let n = captures[0].len();
                self.input = &self.input[n..];
                Some(captures)
            }
            None => None,
        }
    }

    fn consume_whitespace(&mut self) {
        self.consume_prefix(&*&REGEX_WHITESPACE);
    }

    fn consume_symbol(&mut self) -> Result<Token> {
        for (&symbol, token) in SYMBOLS.iter() {
            if self.input.starts_with(symbol) {
                self.input = &self.input[symbol.len()..];
                return Ok(token.clone());
            }
        }
        Err(anyhow!("syntax error"))
    }

    fn tokenize(mut self) -> Result<Vec<Token>> {
        let mut tokens = vec![];
        while !self.input.is_empty() {
            self.consume_whitespace();
            if let Some(captures) = self.consume_prefix(&*REGEX_IDENTIFIER) {
                tokens.push(Token::Variable(captures[0].to_string()));
            } else if let Some(captures) = self.consume_prefix(&*REGEX_NUMBER_8) {
                tokens.push(Token::Number(captures[0].parse().unwrap()));
            } else if let Some(captures) = self.consume_prefix(&*REGEX_NUMBER_2) {
                tokens.push(Token::Number(captures[0].parse().unwrap()));
            } else if let Some(captures) = self.consume_prefix(&*REGEX_NUMBER_16) {
                tokens.push(Token::Number(captures[0].parse().unwrap()));
            } else if let Some(captures) = self.consume_prefix(&*REGEX_NUMBER_10) {
                tokens.push(Token::Number(captures[0].parse().unwrap()));
            } else {
                tokens.push(self.consume_symbol()?);
            }
        }
        tokens.push(Token::EndOfInput);
        Ok(tokens)
    }
}

/// Scans an expression in Starkom's expression syntax and returns the corresponding list of lexical
/// tokens.
pub(crate) fn tokenize(input: &str) -> Result<Vec<Token>> {
    Lexer::new(input).tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
