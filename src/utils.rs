use ff::{Field, PrimeField};
use primitive_types::{H512, U512};
use sha3::Digest;
use starkom_bluesky::Scalar;
use std::sync::LazyLock;

/// Interprets the provided 64 bytes in little-endian order and converts them to a BlueSky scalar
/// via modular reduction.
pub(crate) fn h512_to_scalar(h512: H512) -> Scalar {
    static MODULUS: LazyLock<U512> = LazyLock::new(|| Scalar::MODULUS.parse().unwrap());
    let dividend = U512::from_little_endian(h512.as_bytes());
    let remainder = dividend % *MODULUS;
    let bytes = &remainder.to_little_endian()[0..32];
    Scalar::from_repr_vartime(bytes).unwrap()
}

/// Hashes an arbitrary text string into a uniformly distributed BlueSky scalar.
///
/// Under the hood this function works by hashing the string with SHA3-512 and converting the
/// resulting 64 bytes to a BlueSky scalar via modular reduction.
pub(crate) fn hash_to_scalar(message: &[u8]) -> Scalar {
    let mut hasher = sha3::Sha3_512::new();
    hasher.update(message);
    h512_to_scalar(H512::from_slice(hasher.finalize().as_slice()))
}

/// Generates a new, uniformly distributed, random scalar using a CSPRNG.
pub fn get_random_scalar<F: Field>() -> F {
    F::random(rand_core::OsRng)
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
