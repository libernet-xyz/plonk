use crate::witness::Cell;
use anyhow::{Context, Result, anyhow};
use primitive_types::{H256, U256};
use sha3::Digest;
use starkom_ff::{Field, Field256};
use starkom_pcs as pcs;
use std::collections::BTreeSet;

/// Helper function used to derive domain separator tags used in various contexts.
pub(crate) fn make_dst(s: &'static [u8]) -> H256 {
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(s);
    H256::from_slice(hasher.finalize().as_slice())
}

/// Encodes a [`usize`] into an [`H256`] for use in Fiat-Shamir transcripts.
pub(crate) fn encode_usize(value: usize) -> H256 {
    let mut bytes = [0u8; 32];
    bytes[24..32].copy_from_slice(&(value as u64).to_be_bytes());
    H256::from_slice(&bytes)
}

/// Encodes a [`Cell`] into an [`H256`] for use in Fiat-Shamir transcripts.
pub(crate) fn encode_cell(cell: Cell) -> H256 {
    let mut bytes = [0u8; 32];
    bytes[8..16].copy_from_slice(&(cell.row() as u64).to_be_bytes());
    bytes[24..32].copy_from_slice(&(cell.column() as u64).to_be_bytes());
    H256::from_slice(&bytes)
}

/// Converts an [`isize`] to a field element, wrapping negative values around.
pub(crate) fn isize_to_scalar<F: Field>(value: isize) -> F {
    let abs = value.unsigned_abs();
    if value < 0 {
        -F::try_from(abs).unwrap()
    } else {
        F::try_from(abs).unwrap()
    }
}

/// Indicates whether a scalar value looks like a "negative" value.
///
/// In some context (e.g. in constraint expression parsing when interpreting an exponent) we get
/// scalar values that we need to convert to signed [`isize`] values.
pub(crate) fn is_pseudo_negative<F: Field>(&value: &F) -> bool {
    value.to_u256() > (F::MAX.to_u256() >> 1)
}

/// Converts a field element to [`isize`], using [`is_pseudo_negative`] to decide when a field
/// element is to be interpreted as a negative / wrapped-around value.
pub(crate) fn scalar_to_isize<F: Field>(value: F) -> Result<isize> {
    if is_pseudo_negative(&value) {
        let abs = (F::MAX - value + F::ONE).to_u256();
        if abs > U256::from((-(isize::MIN as i128)) as u128) {
            Err(anyhow!("out of range: {} < {}", value, isize::MIN))
        } else {
            Ok((-(abs.as_u128() as i128)) as isize)
        }
    } else {
        let value = value.to_u256();
        if value > U256::from(isize::MAX as u128) {
            Err(anyhow!("out of range: {} > {}", value, isize::MAX))
        } else {
            Ok(value.as_u128() as isize)
        }
    }
}

/// Calculates the final circuit size (number of rows) by adding the minimum number of blinding rows
/// and rounding up to the next power of two. No blinding rows are added if the `blind` flag is
/// false.
///
/// The returned value is the total number of rows, always a power of two and suitable for use as
/// the size of the evaluation domain. Callers are expected to blind *all* of the rows past the
/// `num_rows` witness rows rather than just the minimum: the extra rows are there anyway due to the
/// power-of-two rounding, and padding them with random values rather than zeros adds safety margin
/// at no cost.
///
/// The returned value is never less than 2. Every committed polynomial has degree lower than it,
/// which is what the whole degree accounting of
/// [`quotient_degree_bound`](`crate::plonk::quotient_degree_bound`) rests on, but the numerator
/// factors of the permutation argument contain a `beta * g^i * X` term whose degree is 1 no matter
/// how small the circuit is. A single-row circuit with blinding disabled would otherwise get a
/// one-element domain, where that term alone overflows a quotient degree bound of `(1 - 1) * E`.
///
/// The minimum number of blinding rows is computed so that the added randomness absorbs the
/// information leak caused by opening all FRI queries and all rotations used in the circuit, and
/// adds an extra 256 bits on top of that.
///
/// In the implementation we always force the 0 and +1 rotations into the provided rotation set
/// because the main challenge xi and the shifted challenge xi*omega are always opened (for the
/// final algebraic check and the permutation argument, respectively) even if the circuit doesn't
/// use those rotations.
///
/// NOTE: when using the extension field pattern (e.g. F=Goldilocks, G=Goldilocks^4) each opened
/// rotation `W(xi)`, `W(omega*xi)`, etc. leaks `G::BITS` bits of information about the witness
/// column `W`, not just `F::BITS` bits! That is because the challenge `xi` is in `G` (not in the
/// `F` subfield) and touches all coefficients of the polynomial upon evaluation, so an evaluation
/// yields 256 bits of information. For this reason our formula is:
///
///   min_blinding_rows = (num_rotations + num_queries + 1) * ceil(G::LEN / F::LEN)
///
/// ensuring that every opened FRI query and every opened rotation is absorbed by 256 bits of
/// blinding, and 256 bits of excess are added on top of that.
pub(crate) fn padded_circuit_size<F: Field, G: Field256<BaseField = F>>(
    num_rows: usize,
    security_bits: usize,
    blowup_log2: usize,
    grinding_bits: usize,
    blind: bool,
    rotations: impl IntoIterator<Item = isize>,
) -> usize {
    let min_blinding_rows = if blind {
        let num_rotations = [0isize, 1isize]
            .into_iter()
            .chain(rotations.into_iter())
            .collect::<BTreeSet<isize>>()
            .len();
        (num_rotations + pcs::num_queries(security_bits, blowup_log2, grinding_bits) + 1)
            * G::LEN.div_ceil(F::LEN)
    } else {
        0
    };
    std::cmp::max(2, (num_rows + min_blinding_rows).next_power_of_two())
}

/// Loads an `F` value from its little-endian canonical integer representation.
///
/// REQUIRES: the length of `bytes` must be less than or equal to [`F::LEN`](`Field::LEN`).
///
/// If `bytes` has less than `F::LEN` bytes the value is automatically padded with zeros.
pub(crate) fn load_scalar<F: Field>(bytes: &[u8]) -> Result<F> {
    if bytes.len() > F::LEN {
        return Err(anyhow!(
            "invalid scalar length (expected {}, got {})",
            F::LEN,
            bytes.len()
        ));
    }
    const { assert!(F::LEN <= 32) }
    let mut padded_bytes = [0u8; 32];
    padded_bytes[0..bytes.len()].copy_from_slice(bytes);
    F::try_from_le_bytes(&padded_bytes[0..F::LEN])
        .into_option()
        .context(format!(
            "invalid scalar (exceeds the [0, {}) range)",
            F::MODULUS
        ))
}

/// Serializes an `F` value into its little-endian canonical integer representation.
pub(crate) fn store_scalar<F: Field>(value: F) -> Vec<u8> {
    let mut bytes = vec![0u8; F::LEN];
    bytes.copy_from_slice(&value.to_u256().to_little_endian()[0..F::LEN]);
    bytes
}

/// Loads a 256-bit hash from its serialized byte form, erroring out if the length of the provided
/// slice is not 32.
pub(crate) fn load_hash(bytes: &[u8]) -> Result<H256> {
    if bytes.len() != 32 {
        return Err(anyhow!(
            "invalid hash length (expected 32, got {})",
            bytes.len()
        ));
    }
    Ok(H256::from_slice(bytes))
}

/// Stores a 256-bit hash into its serialized byte form.
pub(crate) fn store_hash(hash: H256) -> Vec<u8> {
    hash.as_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::padded_circuit_size as pcs;
    use super::*;
    use starkom_bluesky::{Scalar as BS, from_const};
    use starkom_goldilocks::{GL, GL4};

    const SECURITY_BITS: usize = 128;

    #[test]
    fn test_isize_to_scalar() {
        assert_eq!(isize_to_scalar::<BS>(0), from_const(0));
        assert_eq!(isize_to_scalar::<BS>(1), from_const(1));
        assert_eq!(isize_to_scalar::<BS>(2), from_const(2));
        assert_eq!(isize_to_scalar::<BS>(-1), BS::MAX);
        assert_eq!(isize_to_scalar::<BS>(-2), BS::MAX - from_const(1));
        assert_eq!(isize_to_scalar::<BS>(-3), BS::MAX - from_const(2));
    }

    #[test]
    fn test_is_pseudo_negative() {
        assert!(!is_pseudo_negative(&from_const(0)));
        assert!(!is_pseudo_negative(&from_const(1)));
        assert!(!is_pseudo_negative(&from_const(2)));
        assert!(is_pseudo_negative(&(BS::MAX)));
        assert!(is_pseudo_negative(&(BS::MAX - from_const(1))));
        assert!(is_pseudo_negative(&(BS::MAX - from_const(2))));
        let half_range = BS::MAX * BS::TWO_INV;
        assert!(!is_pseudo_negative(&(half_range - from_const(2))));
        assert!(!is_pseudo_negative(&(half_range - from_const(1))));
        assert!(!is_pseudo_negative(&(half_range)));
        assert!(is_pseudo_negative(&(half_range + from_const(1))));
        assert!(is_pseudo_negative(&(half_range + from_const(2))));
    }

    #[test]
    fn test_scalar_to_isize() {
        assert_eq!(scalar_to_isize(from_const(0)).unwrap(), 0);
        assert_eq!(scalar_to_isize(from_const(1)).unwrap(), 1);
        assert_eq!(scalar_to_isize(from_const(2)).unwrap(), 2);
        assert_eq!(
            scalar_to_isize(from_const((isize::MAX - 1) as u64)).unwrap(),
            isize::MAX - 1
        );
        assert_eq!(
            scalar_to_isize(from_const(isize::MAX as u64)).unwrap(),
            isize::MAX
        );
        assert!(scalar_to_isize(from_const(isize::MAX as u64) + from_const(1)).is_err());
    }

    #[test]
    fn test_pseudo_negative_scalar_to_isize() {
        assert_eq!(scalar_to_isize(-from_const(0)).unwrap(), 0);
        assert_eq!(scalar_to_isize(-from_const(1)).unwrap(), -1);
        assert_eq!(scalar_to_isize(-from_const(2)).unwrap(), -2);
        assert_eq!(scalar_to_isize(-from_const(3)).unwrap(), -3);
        assert_eq!(-isize::MAX, isize::MIN + 1);
        let min = -from_const(isize::MAX as u64) - from_const(1);
        assert_eq!(
            scalar_to_isize(min + from_const(2)).unwrap(),
            isize::MIN + 2
        );
        assert_eq!(
            scalar_to_isize(min + from_const(1)).unwrap(),
            isize::MIN + 1
        );
        assert_eq!(scalar_to_isize(min).unwrap(), isize::MIN);
        assert!(scalar_to_isize(min - from_const(1)).is_err());
    }

    #[test]
    fn test_padded_circuit_size_bluesky() {
        let l = SECURITY_BITS;
        assert_eq!(pcs::<BS, BS>(1, l, 1, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(2, l, 1, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(3, l, 1, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(124, l, 1, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(125, l, 1, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(126, l, 1, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(127, l, 1, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(380, l, 1, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(381, l, 1, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(382, l, 1, 0, true, [0, 1]), 1024);
        assert_eq!(pcs::<BS, BS>(383, l, 1, 0, true, [0, 1]), 1024);

        assert_eq!(pcs::<BS, BS>(1, l, 2, 0, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(2, l, 2, 0, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(3, l, 2, 0, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(60, l, 2, 0, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(61, l, 2, 0, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(62, l, 2, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(63, l, 2, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(188, l, 2, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(189, l, 2, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(190, l, 2, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(191, l, 2, 0, true, [0, 1]), 512);

        assert_eq!(pcs::<BS, BS>(1, l, 3, 0, true, [0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(2, l, 3, 0, true, [0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(3, l, 3, 0, true, [0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(17, l, 3, 0, true, [0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(18, l, 3, 0, true, [0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(19, l, 3, 0, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(20, l, 3, 0, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(81, l, 3, 0, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(82, l, 3, 0, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(83, l, 3, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(84, l, 3, 0, true, [0, 1]), 256);
    }

    #[test]
    fn test_padded_circuit_size_bluesky_three_rotations() {
        let l = SECURITY_BITS;
        assert_eq!(pcs::<BS, BS>(1, l, 1, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(2, l, 1, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(3, l, 1, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(123, l, 1, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(124, l, 1, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(125, l, 1, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(126, l, 1, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(379, l, 1, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(380, l, 1, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(381, l, 1, 0, true, [-1, 0, 1]), 1024);
        assert_eq!(pcs::<BS, BS>(382, l, 1, 0, true, [-1, 0, 1]), 1024);

        assert_eq!(pcs::<BS, BS>(1, l, 2, 0, true, [-1, 0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(2, l, 2, 0, true, [-1, 0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(3, l, 2, 0, true, [-1, 0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(59, l, 2, 0, true, [-1, 0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(60, l, 2, 0, true, [-1, 0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(61, l, 2, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(62, l, 2, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(187, l, 2, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(188, l, 2, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(189, l, 2, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(190, l, 2, 0, true, [-1, 0, 1]), 512);

        assert_eq!(pcs::<BS, BS>(1, l, 3, 0, true, [-1, 0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(2, l, 3, 0, true, [-1, 0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(3, l, 3, 0, true, [-1, 0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(16, l, 3, 0, true, [-1, 0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(17, l, 3, 0, true, [-1, 0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(18, l, 3, 0, true, [-1, 0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(19, l, 3, 0, true, [-1, 0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(80, l, 3, 0, true, [-1, 0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(81, l, 3, 0, true, [-1, 0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(82, l, 3, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(83, l, 3, 0, true, [-1, 0, 1]), 256);
    }

    #[test]
    fn test_padded_circuit_size_goldilocks() {
        let l = SECURITY_BITS;
        assert_eq!(pcs::<GL, GL4>(1, l, 1, 0, true, [0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(2, l, 1, 0, true, [0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(3, l, 1, 0, true, [0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(499, l, 1, 0, true, [0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(500, l, 1, 0, true, [0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(501, l, 1, 0, true, [0, 1]), 2048);
        assert_eq!(pcs::<GL, GL4>(502, l, 1, 0, true, [0, 1]), 2048);
        assert_eq!(pcs::<GL, GL4>(1523, l, 1, 0, true, [0, 1]), 2048);
        assert_eq!(pcs::<GL, GL4>(1524, l, 1, 0, true, [0, 1]), 2048);
        assert_eq!(pcs::<GL, GL4>(1525, l, 1, 0, true, [0, 1]), 4096);
        assert_eq!(pcs::<GL, GL4>(1526, l, 1, 0, true, [0, 1]), 4096);

        assert_eq!(pcs::<GL, GL4>(1, l, 2, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(2, l, 2, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(3, l, 2, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(243, l, 2, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(244, l, 2, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(245, l, 2, 0, true, [0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(246, l, 2, 0, true, [0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(755, l, 2, 0, true, [0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(756, l, 2, 0, true, [0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(757, l, 2, 0, true, [0, 1]), 2048);
        assert_eq!(pcs::<GL, GL4>(758, l, 2, 0, true, [0, 1]), 2048);

        assert_eq!(pcs::<GL, GL4>(1, l, 3, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<GL, GL4>(2, l, 3, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<GL, GL4>(3, l, 3, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<GL, GL4>(71, l, 3, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<GL, GL4>(72, l, 3, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<GL, GL4>(73, l, 3, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(74, l, 3, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(327, l, 3, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(328, l, 3, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(329, l, 3, 0, true, [0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(330, l, 3, 0, true, [0, 1]), 1024);
    }

    #[test]
    fn test_padded_circuit_size_goldilocks_three_rotations() {
        let l = SECURITY_BITS;
        assert_eq!(pcs::<GL, GL4>(1, l, 1, 0, true, [-1, 0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(2, l, 1, 0, true, [-1, 0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(3, l, 1, 0, true, [-1, 0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(495, l, 1, 0, true, [-1, 0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(496, l, 1, 0, true, [-1, 0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(497, l, 1, 0, true, [-1, 0, 1]), 2048);
        assert_eq!(pcs::<GL, GL4>(498, l, 1, 0, true, [-1, 0, 1]), 2048);
        assert_eq!(pcs::<GL, GL4>(1519, l, 1, 0, true, [-1, 0, 1]), 2048);
        assert_eq!(pcs::<GL, GL4>(1520, l, 1, 0, true, [-1, 0, 1]), 2048);
        assert_eq!(pcs::<GL, GL4>(1521, l, 1, 0, true, [-1, 0, 1]), 4096);
        assert_eq!(pcs::<GL, GL4>(1522, l, 1, 0, true, [-1, 0, 1]), 4096);

        assert_eq!(pcs::<GL, GL4>(1, l, 2, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(2, l, 2, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(3, l, 2, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(239, l, 2, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(240, l, 2, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(241, l, 2, 0, true, [-1, 0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(242, l, 2, 0, true, [-1, 0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(751, l, 2, 0, true, [-1, 0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(752, l, 2, 0, true, [-1, 0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(753, l, 2, 0, true, [-1, 0, 1]), 2048);
        assert_eq!(pcs::<GL, GL4>(754, l, 2, 0, true, [-1, 0, 1]), 2048);

        assert_eq!(pcs::<GL, GL4>(1, l, 3, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<GL, GL4>(2, l, 3, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<GL, GL4>(3, l, 3, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<GL, GL4>(67, l, 3, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<GL, GL4>(68, l, 3, 0, true, [-1, 0, 1]), 256);
        assert_eq!(pcs::<GL, GL4>(69, l, 3, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(70, l, 3, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(323, l, 3, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(324, l, 3, 0, true, [-1, 0, 1]), 512);
        assert_eq!(pcs::<GL, GL4>(325, l, 3, 0, true, [-1, 0, 1]), 1024);
        assert_eq!(pcs::<GL, GL4>(326, l, 3, 0, true, [-1, 0, 1]), 1024);
    }

    #[test]
    fn test_padded_circuit_size_with_grinding() {
        let l = SECURITY_BITS;
        assert_eq!(pcs::<BS, BS>(144, l, 1, 20, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(145, l, 1, 20, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(146, l, 1, 20, true, [0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(147, l, 1, 20, true, [0, 1]), 512);

        assert_eq!(pcs::<BS, BS>(70, l, 2, 20, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(71, l, 2, 20, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(72, l, 2, 20, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(73, l, 2, 20, true, [0, 1]), 256);

        assert_eq!(pcs::<BS, BS>(24, l, 3, 20, true, [0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(25, l, 3, 20, true, [0, 1]), 64);
        assert_eq!(pcs::<BS, BS>(26, l, 3, 20, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(27, l, 3, 20, true, [0, 1]), 128);
    }

    #[test]
    fn test_padded_circuit_size_with_security_bits() {
        assert_eq!(pcs::<BS, BS>(29, 96, 1, 0, true, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(30, 96, 1, 0, true, [0, 1]), 256);

        assert_eq!(pcs::<BS, BS>(125, 128, 1, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(126, 128, 1, 0, true, [0, 1]), 512);

        assert_eq!(pcs::<BS, BS>(61, 192, 1, 0, true, [0, 1]), 256);
        assert_eq!(pcs::<BS, BS>(62, 192, 1, 0, true, [0, 1]), 512);

        assert_eq!(pcs::<BS, BS>(253, 256, 1, 0, true, [0, 1]), 512);
        assert_eq!(pcs::<BS, BS>(254, 256, 1, 0, true, [0, 1]), 1024);
    }

    #[test]
    fn test_padded_circuit_size_with_security_bits_without_blinding() {
        assert_eq!(pcs::<BS, BS>(100, 96, 1, 0, false, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(100, 128, 1, 0, false, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(100, 192, 1, 0, false, [0, 1]), 128);
        assert_eq!(pcs::<BS, BS>(100, 256, 1, 0, false, [0, 1]), 128);
    }

    #[test]
    fn test_padded_circuit_size_duplicate_rotations() {
        let l = SECURITY_BITS;
        assert_eq!(pcs::<BS, BS>(1, l, 1, 0, true, [0, 1, 1]), 256);
        assert_eq!(pcs::<BS, BS>(2, l, 1, 0, true, [0, 1, 1]), 256);
        assert_eq!(pcs::<BS, BS>(3, l, 1, 0, true, [0, 1, 1]), 256);
        assert_eq!(pcs::<BS, BS>(124, l, 1, 0, true, [0, 1, 1]), 256);
        assert_eq!(pcs::<BS, BS>(125, l, 1, 0, true, [0, 1, 1]), 256);
        assert_eq!(pcs::<BS, BS>(126, l, 1, 0, true, [0, 1, 1]), 512);
        assert_eq!(pcs::<BS, BS>(127, l, 1, 0, true, [0, 1, 1]), 512);
        assert_eq!(pcs::<BS, BS>(380, l, 1, 0, true, [0, 1, 1]), 512);
        assert_eq!(pcs::<BS, BS>(381, l, 1, 0, true, [0, 1, 1]), 512);
        assert_eq!(pcs::<BS, BS>(382, l, 1, 0, true, [0, 1, 1]), 1024);
        assert_eq!(pcs::<BS, BS>(383, l, 1, 0, true, [0, 1, 1]), 1024);

        assert_eq!(pcs::<BS, BS>(1, l, 2, 0, true, [0, 1, 1]), 128);
        assert_eq!(pcs::<BS, BS>(2, l, 2, 0, true, [0, 1, 1]), 128);
        assert_eq!(pcs::<BS, BS>(3, l, 2, 0, true, [0, 1, 1]), 128);
        assert_eq!(pcs::<BS, BS>(60, l, 2, 0, true, [0, 1, 1]), 128);
        assert_eq!(pcs::<BS, BS>(61, l, 2, 0, true, [0, 1, 1]), 128);
        assert_eq!(pcs::<BS, BS>(62, l, 2, 0, true, [0, 1, 1]), 256);
        assert_eq!(pcs::<BS, BS>(63, l, 2, 0, true, [0, 1, 1]), 256);
        assert_eq!(pcs::<BS, BS>(188, l, 2, 0, true, [0, 1, 1]), 256);
        assert_eq!(pcs::<BS, BS>(189, l, 2, 0, true, [0, 1, 1]), 256);
        assert_eq!(pcs::<BS, BS>(190, l, 2, 0, true, [0, 1, 1]), 512);
        assert_eq!(pcs::<BS, BS>(191, l, 2, 0, true, [0, 1, 1]), 512);

        assert_eq!(pcs::<BS, BS>(1, l, 3, 0, true, [0, 1, 1]), 64);
        assert_eq!(pcs::<BS, BS>(2, l, 3, 0, true, [0, 1, 1]), 64);
        assert_eq!(pcs::<BS, BS>(3, l, 3, 0, true, [0, 1, 1]), 64);
        assert_eq!(pcs::<BS, BS>(17, l, 3, 0, true, [0, 1, 1]), 64);
        assert_eq!(pcs::<BS, BS>(18, l, 3, 0, true, [0, 1, 1]), 64);
        assert_eq!(pcs::<BS, BS>(19, l, 3, 0, true, [0, 1, 1]), 128);
        assert_eq!(pcs::<BS, BS>(20, l, 3, 0, true, [0, 1, 1]), 128);
        assert_eq!(pcs::<BS, BS>(81, l, 3, 0, true, [0, 1, 1]), 128);
        assert_eq!(pcs::<BS, BS>(82, l, 3, 0, true, [0, 1, 1]), 128);
        assert_eq!(pcs::<BS, BS>(83, l, 3, 0, true, [0, 1, 1]), 256);
        assert_eq!(pcs::<BS, BS>(84, l, 3, 0, true, [0, 1, 1]), 256);
    }

    #[test]
    fn test_load_scalar() {
        let value1: BS = "0x1fcc49bb3722a3a6ac41aa0aac6ed68f218a7240ca2a9befd4e3243251f7ecaf"
            .parse()
            .unwrap();
        let value2: BS = "0x39b3bad9cc2a94bf8f5f3c5277885fa778893898702be5ec79eaf59e08053a23"
            .parse()
            .unwrap();
        assert_eq!(load_scalar::<BS>(&value1.to_le_bytes()).unwrap(), value1);
        assert_eq!(load_scalar::<BS>(&value2.to_le_bytes()).unwrap(), value2);
    }

    fn test_load_truncated_scalar_impl(value: BS) {
        let bytes = value.to_le_bytes();
        let length = bytes
            .iter()
            .rposition(|&byte| byte != 0)
            .map_or(0, |index| index + 1);
        assert_eq!(load_scalar::<BS>(&bytes[0..length]).unwrap(), value);
    }

    #[test]
    fn test_load_truncated_scalar() {
        test_load_truncated_scalar_impl(
            "0x00cc49bb3722a3a6ac41aa0aac6ed68f218a7240ca2a9befd4e3243251f7ecaf"
                .parse()
                .unwrap(),
        );
        test_load_truncated_scalar_impl(
            "0x000049bb3722a3a6ac41aa0aac6ed68f218a7240ca2a9befd4e3243251f7ecaf"
                .parse()
                .unwrap(),
        );
        test_load_truncated_scalar_impl(
            "0x000000000000000000000000000000000000000000000000000000000000ecaf"
                .parse()
                .unwrap(),
        );
        test_load_truncated_scalar_impl(
            "0x00000000000000000000000000000000000000000000000000000000000000af"
                .parse()
                .unwrap(),
        );
        test_load_truncated_scalar_impl(
            "0x0000000000000000000000000000000000000000000000000000000000000000"
                .parse()
                .unwrap(),
        );
    }

    #[test]
    fn test_store_scalar() {
        let value1: BS = "0x1fcc49bb3722a3a6ac41aa0aac6ed68f218a7240ca2a9befd4e3243251f7ecaf"
            .parse()
            .unwrap();
        let value2: BS = "0x39b3bad9cc2a94bf8f5f3c5277885fa778893898702be5ec79eaf59e08053a23"
            .parse()
            .unwrap();
        assert_eq!(
            load_scalar::<BS>(store_scalar(value1).as_slice()).unwrap(),
            value1
        );
        assert_eq!(
            load_scalar::<BS>(store_scalar(value2).as_slice()).unwrap(),
            value2
        );
    }

    #[test]
    fn test_load_sha2_hash() {
        let hash1: H256 = "0x8c313b317b2ff13ecf3abea4e59dad76b24a13e7b5b8fa10a939fc2ab1713175"
            .parse()
            .unwrap();
        let hash2: H256 = "0x51c39e4b5287d491be2e02080c39493b77be8fd3d032227815357c5360e3d230"
            .parse()
            .unwrap();
        assert_eq!(load_hash(hash1.as_bytes()).unwrap(), hash1);
        assert_eq!(load_hash(hash2.as_bytes()).unwrap(), hash2);
        assert!(load_hash(&hash1.as_bytes()[0..31]).is_err());
    }

    #[test]
    fn test_store_hash() {
        let hash1: H256 = "0x8c313b317b2ff13ecf3abea4e59dad76b24a13e7b5b8fa10a939fc2ab1713175"
            .parse()
            .unwrap();
        let hash2: H256 = "0x51c39e4b5287d491be2e02080c39493b77be8fd3d032227815357c5360e3d230"
            .parse()
            .unwrap();
        assert_eq!(load_hash(store_hash(hash1).as_slice()).unwrap(), hash1);
        assert_eq!(load_hash(store_hash(hash2).as_slice()).unwrap(), hash2);
    }
}
