use anyhow::{Result, anyhow};
use primitive_types::H256;
use sha3::Digest;
use starkom_ff::{Field, Field256};
use std::collections::BTreeSet;

/// Helper function used to derive domain separator tags used in various contexts.
pub(crate) fn make_dst(s: &'static [u8]) -> H256 {
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(s);
    H256::from_slice(hasher.finalize().as_slice())
}

/// Converts an [`isize`] to a [`Scalar`], wrapping negative values around.
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
/// [`Scalar`] values that we need to convert to signed [`isize`] values.
pub(crate) fn is_pseudo_negative<F: Field>(&value: &F) -> bool {
    value > F::MAX * F::TWO_INV
}

pub(crate) fn scalar_to_isize<F: Field>(value: F) -> Result<isize> {
    if is_pseudo_negative(&value) {
        let abs = (F::MAX - value + F::ONE).try_to_u128().unwrap() as i128;
        if abs > -(isize::MIN as i128) {
            Err(anyhow!("out of range: {}", value))
        } else {
            Ok(-abs as isize)
        }
    } else {
        let isize_max = F::from(isize::MAX as u64);
        if value > isize_max {
            Err(anyhow!("out of range: {}", value))
        } else {
            Ok(value.try_to_u64().unwrap() as isize)
        }
    }
}

/// Calculates the final circuit size (number of rows) by adding the correct number of blinding rows
/// and rounding up to the next power of two.
///
/// The returned pair is `(degree_bound, nun_blinding_rows)`, with `degree_bound` indicating the
/// total number of rows (always a power of two and suitable for use as the size of the evaluation
/// domain).
///
/// The number of blinding rows added must be strictly greater than the number of non-public opened
/// points, so we calculate it as the total number of different variable rotations present in the
/// circuit plus one. We force the 0 and +1 rotations into the rotation set because the main
/// challenge xi and the shifted challenge xi*omega are always opened (for the final algebraic check
/// and the permutation argument, respectively) even if the circuit doesn't use those rotations.
pub(crate) fn padded_circuit_size<R: IntoIterator<Item = isize>>(
    num_rows: usize,
    rotations: R,
) -> (usize, usize) {
    let num_blinding_rows = [0isize, 1isize]
        .into_iter()
        .chain(rotations.into_iter())
        .collect::<BTreeSet<isize>>()
        .len()
        + 1;
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
