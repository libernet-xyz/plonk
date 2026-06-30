use primitive_types::H512;
use sha3::Digest;
use starkom_bluesky::Scalar;
use starkom_ff::Field256;

/// Hashes an arbitrary text string into a uniformly distributed BlueSky scalar.
///
/// Under the hood this function works by hashing the string with SHA3-512 and converting the
/// resulting 64 bytes to a BlueSky scalar via modular reduction.
pub(crate) fn hash_to_scalar(message: &[u8]) -> Scalar {
    let mut hasher = sha3::Sha3_512::new();
    hasher.update(message);
    Scalar::from_h512(H512::from_slice(hasher.finalize().as_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
