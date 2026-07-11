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
        self.consume_whitespace();
        while !self.input.is_empty() {
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
            self.consume_whitespace();
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

    #[inline]
    fn tokenize(input: &'static str) -> Vec<Token> {
        super::tokenize(input).unwrap()
    }

    #[test]
    fn test_vitalik() {
        assert_eq!(
            tokenize("w0 ^ 3 + w0 + 5 == 35"),
            vec![
                Token::Variable("w0".to_string()),
                Token::Power,
                Token::Number(Scalar::from_const(3)),
                Token::Plus,
                Token::Variable("w0".to_string()),
                Token::Plus,
                Token::Number(Scalar::from_const(5)),
                Token::Equal,
                Token::Number(Scalar::from_const(35)),
                Token::EndOfInput
            ]
        );
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(tokenize(""), vec![Token::EndOfInput]);
    }

    #[test]
    fn test_whitespace_only() {
        assert_eq!(tokenize("   \t\n  "), vec![Token::EndOfInput]);
    }

    #[test]
    fn test_whitespace_is_optional_between_symbols() {
        assert_eq!(
            tokenize("1+2*3-4/5^6==7"),
            vec![
                Token::Number(Scalar::from_const(1)),
                Token::Plus,
                Token::Number(Scalar::from_const(2)),
                Token::Multiply,
                Token::Number(Scalar::from_const(3)),
                Token::Minus,
                Token::Number(Scalar::from_const(4)),
                Token::Divide,
                Token::Number(Scalar::from_const(5)),
                Token::Power,
                Token::Number(Scalar::from_const(6)),
                Token::Equal,
                Token::Number(Scalar::from_const(7)),
                Token::EndOfInput
            ]
        );
    }

    #[test]
    fn test_mixed_whitespace_between_tokens() {
        assert_eq!(
            tokenize("1 \t+\n2"),
            vec![
                Token::Number(Scalar::from_const(1)),
                Token::Plus,
                Token::Number(Scalar::from_const(2)),
                Token::EndOfInput
            ]
        );
    }

    #[test]
    fn test_decimal_zero() {
        assert_eq!(
            tokenize("0"),
            vec![Token::Number(Scalar::from_const(0)), Token::EndOfInput]
        );
    }

    #[test]
    fn test_decimal_number() {
        assert_eq!(
            tokenize("123456789"),
            vec![
                Token::Number(Scalar::from_const(123456789)),
                Token::EndOfInput
            ]
        );
    }

    #[test]
    fn test_octal_number() {
        assert_eq!(
            tokenize("017"),
            vec![Token::Number(Scalar::from_const(15)), Token::EndOfInput]
        );
    }

    #[test]
    fn test_octal_number_with_leading_zeros() {
        assert_eq!(
            tokenize("007"),
            vec![Token::Number(Scalar::from_const(7)), Token::EndOfInput]
        );
    }

    #[test]
    fn test_binary_number_lowercase_prefix() {
        assert_eq!(
            tokenize("0b1010"),
            vec![Token::Number(Scalar::from_const(10)), Token::EndOfInput]
        );
    }

    #[test]
    fn test_binary_number_uppercase_prefix() {
        assert_eq!(
            tokenize("0B1010"),
            vec![Token::Number(Scalar::from_const(10)), Token::EndOfInput]
        );
    }

    #[test]
    fn test_hexadecimal_number_lowercase_prefix() {
        assert_eq!(
            tokenize("0xff"),
            vec![Token::Number(Scalar::from_const(255)), Token::EndOfInput]
        );
    }

    #[test]
    fn test_hexadecimal_number_uppercase_prefix() {
        assert_eq!(
            tokenize("0XFF"),
            vec![Token::Number(Scalar::from_const(255)), Token::EndOfInput]
        );
    }

    #[test]
    fn test_variable_names() {
        assert_eq!(
            tokenize("_underscore CamelCase snake_case42"),
            vec![
                Token::Variable("_underscore".to_string()),
                Token::Variable("CamelCase".to_string()),
                Token::Variable("snake_case42".to_string()),
                Token::EndOfInput
            ]
        );
    }

    #[test]
    fn test_all_symbols() {
        assert_eq!(
            tokenize("+ - * / ^ == ( )"),
            vec![
                Token::Plus,
                Token::Minus,
                Token::Multiply,
                Token::Divide,
                Token::Power,
                Token::Equal,
                Token::LeftBracket,
                Token::RightBracket,
                Token::EndOfInput
            ]
        );
    }

    #[test]
    fn test_nested_brackets() {
        assert_eq!(
            tokenize("((1 + 2))"),
            vec![
                Token::LeftBracket,
                Token::LeftBracket,
                Token::Number(Scalar::from_const(1)),
                Token::Plus,
                Token::Number(Scalar::from_const(2)),
                Token::RightBracket,
                Token::RightBracket,
                Token::EndOfInput
            ]
        );
    }

    #[test]
    fn test_unary_minus_is_a_separate_token_from_the_number() {
        assert_eq!(
            tokenize("-5"),
            vec![
                Token::Minus,
                Token::Number(Scalar::from_const(5)),
                Token::EndOfInput
            ]
        );
    }

    #[test]
    fn test_invalid_octal_digit_is_a_syntax_error() {
        assert!(super::tokenize("08").is_err());
    }

    #[test]
    fn test_digit_immediately_followed_by_letter_is_a_syntax_error() {
        assert!(super::tokenize("5x").is_err());
    }

    #[test]
    fn test_incomplete_hexadecimal_prefix_is_a_syntax_error() {
        assert!(super::tokenize("0x").is_err());
    }

    #[test]
    fn test_incomplete_binary_prefix_is_a_syntax_error() {
        assert!(super::tokenize("0b").is_err());
    }

    #[test]
    fn test_unknown_character_is_a_syntax_error() {
        assert!(super::tokenize("$").is_err());
    }

    #[test]
    fn test_single_equal_sign_is_a_syntax_error() {
        assert!(super::tokenize("1=2").is_err());
    }
}
