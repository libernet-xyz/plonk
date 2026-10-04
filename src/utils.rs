use crate::witness::Cell;
use anyhow::{Result, anyhow};
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
    blowup_log2: usize,
    blind: bool,
    rotations: impl IntoIterator<Item = isize>,
) -> usize {
    let min_blinding_rows = if blind {
        let num_rotations = [0isize, 1isize]
            .into_iter()
            .chain(rotations.into_iter())
            .collect::<BTreeSet<isize>>()
            .len();
        (num_rotations + pcs::num_queries(blowup_log2) + 1) * G::LEN.div_ceil(F::LEN)
    } else {
        0
    };
    (num_rows + min_blinding_rows).next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;
    use starkom_bluesky::{Scalar as BS, from_const};
    use starkom_goldilocks::{GL, GL4};

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
        assert_eq!(padded_circuit_size::<BS, BS>(1, 1, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(2, 1, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(3, 1, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(124, 1, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(125, 1, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(126, 1, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(127, 1, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(380, 1, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(381, 1, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(382, 1, true, [0, 1]), 1024);
        assert_eq!(padded_circuit_size::<BS, BS>(383, 1, true, [0, 1]), 1024);

        assert_eq!(padded_circuit_size::<BS, BS>(1, 2, true, [0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(2, 2, true, [0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(3, 2, true, [0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(60, 2, true, [0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(61, 2, true, [0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(62, 2, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(63, 2, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(188, 2, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(189, 2, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(190, 2, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(191, 2, true, [0, 1]), 512);

        assert_eq!(padded_circuit_size::<BS, BS>(1, 3, true, [0, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(2, 3, true, [0, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(3, 3, true, [0, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(17, 3, true, [0, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(18, 3, true, [0, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(19, 3, true, [0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(20, 3, true, [0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(81, 3, true, [0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(82, 3, true, [0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(83, 3, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(84, 3, true, [0, 1]), 256);
    }

    #[test]
    fn test_padded_circuit_size_bluesky_three_rotations() {
        assert_eq!(padded_circuit_size::<BS, BS>(1, 1, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(2, 1, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(3, 1, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(123, 1, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(124, 1, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(125, 1, true, [-1, 0, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(126, 1, true, [-1, 0, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(379, 1, true, [-1, 0, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(380, 1, true, [-1, 0, 1]), 512);
        assert_eq!(
            padded_circuit_size::<BS, BS>(381, 1, true, [-1, 0, 1]),
            1024
        );
        assert_eq!(
            padded_circuit_size::<BS, BS>(382, 1, true, [-1, 0, 1]),
            1024
        );

        assert_eq!(padded_circuit_size::<BS, BS>(1, 2, true, [-1, 0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(2, 2, true, [-1, 0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(3, 2, true, [-1, 0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(59, 2, true, [-1, 0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(60, 2, true, [-1, 0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(61, 2, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(62, 2, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(187, 2, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(188, 2, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(189, 2, true, [-1, 0, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(190, 2, true, [-1, 0, 1]), 512);

        assert_eq!(padded_circuit_size::<BS, BS>(1, 3, true, [-1, 0, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(2, 3, true, [-1, 0, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(3, 3, true, [-1, 0, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(16, 3, true, [-1, 0, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(17, 3, true, [-1, 0, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(18, 3, true, [-1, 0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(19, 3, true, [-1, 0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(80, 3, true, [-1, 0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(81, 3, true, [-1, 0, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(82, 3, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(83, 3, true, [-1, 0, 1]), 256);
    }

    #[test]
    fn test_padded_circuit_size_goldilocks() {
        assert_eq!(padded_circuit_size::<GL, GL4>(1, 1, true, [0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(2, 1, true, [0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(3, 1, true, [0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(499, 1, true, [0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(500, 1, true, [0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(501, 1, true, [0, 1]), 2048);
        assert_eq!(padded_circuit_size::<GL, GL4>(502, 1, true, [0, 1]), 2048);
        assert_eq!(padded_circuit_size::<GL, GL4>(1523, 1, true, [0, 1]), 2048);
        assert_eq!(padded_circuit_size::<GL, GL4>(1524, 1, true, [0, 1]), 2048);
        assert_eq!(padded_circuit_size::<GL, GL4>(1525, 1, true, [0, 1]), 4096);
        assert_eq!(padded_circuit_size::<GL, GL4>(1526, 1, true, [0, 1]), 4096);

        assert_eq!(padded_circuit_size::<GL, GL4>(1, 2, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(2, 2, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(3, 2, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(243, 2, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(244, 2, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(245, 2, true, [0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(246, 2, true, [0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(755, 2, true, [0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(756, 2, true, [0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(757, 2, true, [0, 1]), 2048);
        assert_eq!(padded_circuit_size::<GL, GL4>(758, 2, true, [0, 1]), 2048);

        assert_eq!(padded_circuit_size::<GL, GL4>(1, 3, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<GL, GL4>(2, 3, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<GL, GL4>(3, 3, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<GL, GL4>(71, 3, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<GL, GL4>(72, 3, true, [0, 1]), 256);
        assert_eq!(padded_circuit_size::<GL, GL4>(73, 3, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(74, 3, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(327, 3, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(328, 3, true, [0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(329, 3, true, [0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(330, 3, true, [0, 1]), 1024);
    }

    #[test]
    fn test_padded_circuit_size_goldilocks_three_rotations() {
        assert_eq!(padded_circuit_size::<GL, GL4>(1, 1, true, [-1, 0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(2, 1, true, [-1, 0, 1]), 1024);
        assert_eq!(padded_circuit_size::<GL, GL4>(3, 1, true, [-1, 0, 1]), 1024);
        assert_eq!(
            padded_circuit_size::<GL, GL4>(495, 1, true, [-1, 0, 1]),
            1024
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(496, 1, true, [-1, 0, 1]),
            1024
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(497, 1, true, [-1, 0, 1]),
            2048
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(498, 1, true, [-1, 0, 1]),
            2048
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(1519, 1, true, [-1, 0, 1]),
            2048
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(1520, 1, true, [-1, 0, 1]),
            2048
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(1521, 1, true, [-1, 0, 1]),
            4096
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(1522, 1, true, [-1, 0, 1]),
            4096
        );

        assert_eq!(padded_circuit_size::<GL, GL4>(1, 2, true, [-1, 0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(2, 2, true, [-1, 0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(3, 2, true, [-1, 0, 1]), 512);
        assert_eq!(
            padded_circuit_size::<GL, GL4>(239, 2, true, [-1, 0, 1]),
            512
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(240, 2, true, [-1, 0, 1]),
            512
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(241, 2, true, [-1, 0, 1]),
            1024
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(242, 2, true, [-1, 0, 1]),
            1024
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(751, 2, true, [-1, 0, 1]),
            1024
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(752, 2, true, [-1, 0, 1]),
            1024
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(753, 2, true, [-1, 0, 1]),
            2048
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(754, 2, true, [-1, 0, 1]),
            2048
        );

        assert_eq!(padded_circuit_size::<GL, GL4>(1, 3, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<GL, GL4>(2, 3, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<GL, GL4>(3, 3, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<GL, GL4>(67, 3, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<GL, GL4>(68, 3, true, [-1, 0, 1]), 256);
        assert_eq!(padded_circuit_size::<GL, GL4>(69, 3, true, [-1, 0, 1]), 512);
        assert_eq!(padded_circuit_size::<GL, GL4>(70, 3, true, [-1, 0, 1]), 512);
        assert_eq!(
            padded_circuit_size::<GL, GL4>(323, 3, true, [-1, 0, 1]),
            512
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(324, 3, true, [-1, 0, 1]),
            512
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(325, 3, true, [-1, 0, 1]),
            1024
        );
        assert_eq!(
            padded_circuit_size::<GL, GL4>(326, 3, true, [-1, 0, 1]),
            1024
        );
    }

    #[test]
    fn test_padded_circuit_size_duplicate_rotations() {
        assert_eq!(padded_circuit_size::<BS, BS>(1, 1, true, [0, 1, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(2, 1, true, [0, 1, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(3, 1, true, [0, 1, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(124, 1, true, [0, 1, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(125, 1, true, [0, 1, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(126, 1, true, [0, 1, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(127, 1, true, [0, 1, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(380, 1, true, [0, 1, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(381, 1, true, [0, 1, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(382, 1, true, [0, 1, 1]), 1024);
        assert_eq!(padded_circuit_size::<BS, BS>(383, 1, true, [0, 1, 1]), 1024);

        assert_eq!(padded_circuit_size::<BS, BS>(1, 2, true, [0, 1, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(2, 2, true, [0, 1, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(3, 2, true, [0, 1, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(60, 2, true, [0, 1, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(61, 2, true, [0, 1, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(62, 2, true, [0, 1, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(63, 2, true, [0, 1, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(188, 2, true, [0, 1, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(189, 2, true, [0, 1, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(190, 2, true, [0, 1, 1]), 512);
        assert_eq!(padded_circuit_size::<BS, BS>(191, 2, true, [0, 1, 1]), 512);

        assert_eq!(padded_circuit_size::<BS, BS>(1, 3, true, [0, 1, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(2, 3, true, [0, 1, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(3, 3, true, [0, 1, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(17, 3, true, [0, 1, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(18, 3, true, [0, 1, 1]), 64);
        assert_eq!(padded_circuit_size::<BS, BS>(19, 3, true, [0, 1, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(20, 3, true, [0, 1, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(81, 3, true, [0, 1, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(82, 3, true, [0, 1, 1]), 128);
        assert_eq!(padded_circuit_size::<BS, BS>(83, 3, true, [0, 1, 1]), 256);
        assert_eq!(padded_circuit_size::<BS, BS>(84, 3, true, [0, 1, 1]), 256);
    }
}
