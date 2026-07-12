use anyhow::{Result, anyhow};
use primitive_types::H512;
use sha3::Digest;
use starkom_bluesky::Scalar;
use starkom_ff::{Field, Field256, PrimeField};
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
}
