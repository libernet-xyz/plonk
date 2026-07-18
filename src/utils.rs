use anyhow::{Result, anyhow};
use primitive_types::H512;
use sha3::Digest;
use starkom_bluesky::Scalar;
use starkom_ff::{Field, Field256, PrimeField};
use std::collections::BTreeSet;
use std::sync::LazyLock;

/// Hashes an arbitrary text string into a uniformly distributed BlueSky scalar.
///
/// Under the hood this function works by hashing the string with SHA3-512 and converting the
/// resulting 64 bytes to a BlueSky scalar via modular reduction.
pub(crate) fn hash_to_scalar(message: &[u8]) -> Scalar {
    let mut hasher = sha3::Sha3_512::new();
    hasher.update(message);
    Scalar::from_h512(H512::from_slice(hasher.finalize().as_slice()))
}

/// Converts an [`isize`] to a [`Scalar`], wrapping negative values around.
pub(crate) fn isize_to_scalar(value: isize) -> Scalar {
    let abs = value.unsigned_abs();
    if value < 0 {
        -Scalar::try_from(abs).unwrap()
    } else {
        Scalar::try_from(abs).unwrap()
    }
}

/// Indicates whether a scalar value looks like a "negative" value.
///
/// In some context (e.g. in constraint expression parsing when interpreting an exponent) we get
/// [`Scalar`] values that we need to convert to signed [`isize`] values.
pub(crate) fn is_pseudo_negative(&value: &Scalar) -> bool {
    static HALF_RANGE: LazyLock<Scalar> = LazyLock::new(|| Scalar::MAX * Scalar::TWO_INV);
    value > *HALF_RANGE
}

pub(crate) fn scalar_to_isize(value: Scalar) -> Result<isize> {
    const MAX: Scalar = Scalar::from_const(isize::MAX as u64);
    if is_pseudo_negative(&value) {
        let abs = (Scalar::MAX - value + Scalar::ONE).try_to_u128().unwrap() as i128;
        if abs > -(isize::MIN as i128) {
            Err(anyhow!("out of range: {}", value))
        } else {
            Ok(-abs as isize)
        }
    } else {
        if value > MAX {
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
    use starkom_bluesky::parse_scalar;

    #[test]
    fn test_hash_to_scalar() {
        assert_eq!(
            hash_to_scalar(b"lorem ipsum dolor sit amet"),
            parse_scalar("0x69c562c4b39c86fc322322c86cfe5be83fbd472c6a38862bdd2f362bfa442ad6")
        );
        assert_eq!(
            hash_to_scalar(b"sator arepo tenet opera rotas"),
            parse_scalar("0x027880d47636bf77d55804a6cf2d5ec8f09427cdf678e2ed3d74c432cc2efa7a")
        );
    }

    // TODO
}
