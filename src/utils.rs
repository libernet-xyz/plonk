use crate::witness::Cell;
use anyhow::{Result, anyhow};
use primitive_types::{H256, U256};
use sha3::Digest;
use starkom_ff::Field;
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

/// Calculates the final circuit size (number of rows) by adding the correct number of blinding rows
/// and rounding up to the next power of two.
///
/// The returned pair is `(degree_bound, num_blinding_rows)`, with `degree_bound` indicating the
/// total number of rows (always a power of two and suitable for use as the size of the evaluation
/// domain).
///
/// The number of blinding rows added is computed so that the added randomness absorbs the
/// information leak caused by opening all `rotations` used in the circuit and adds an extra 256
/// bits on top of that.
///
/// In the implementation we always force the 0 and +1 rotations into the set because the main
/// challenge xi and the shifted challenge xi*omega are always opened (for the final algebraic check
/// and the permutation argument, respectively) even if the circuit doesn't use those rotations.
///
/// NOTE: when using the extension field pattern (e.g. F=Goldilocks, G=Goldilocks^4) each opened
/// rotation `W(xi)`, `W(omega*xi)`, etc. reveals `G::BITS` bits of information about the witness
/// column `W`, not just `F::BITS` bits! That is because the challenge `xi` is in `G` (not in the
/// `F` subfield) and touches all coefficients of the polynomial upon evaluation, so an evaluation
/// yields 256 bits of information. For this reason our formula is:
///
///   num_blinding_rows = (num_rotations + 1) * ceil(32 / F::LEN)
///
/// ensuring that every opened rotation is absorbed by 256 bits of blinding and 256 bits of excess
/// are added on top of that.
pub(crate) fn padded_circuit_size<F: Field>(
    num_rows: usize,
    rotations: impl IntoIterator<Item = isize>,
) -> (usize, usize) {
    let num_rotations = [0isize, 1isize]
        .into_iter()
        .chain(rotations.into_iter())
        .collect::<BTreeSet<isize>>()
        .len();
    let num_blinding_rows = (num_rotations + 1) * 32usize.div_ceil(F::LEN);
    let degree_bound = (num_rows + num_blinding_rows).next_power_of_two();
    (degree_bound, num_blinding_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use starkom_bluesky::{Scalar as BS, from_const};

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
}
