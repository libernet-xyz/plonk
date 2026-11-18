use crate::starkom::proto::plonk::v11 as proto;
use crate::utils;
use starkom_ff::PrimeField256;
use starkom_scpf::Encoder;

fn load_scalar<F: PrimeField256>(bytes: &[u8]) -> F {
    utils::load_scalar(bytes).unwrap()
}

fn push_scalars<F: PrimeField256>(encoder: &mut Encoder, values: &[Vec<u8>]) {
    encoder.push_scalars(values.iter().map(|value| load_scalar::<F>(value)));
}

fn push_hashes(encoder: &mut Encoder, hashes: &[Vec<u8>]) {
    encoder.push_hashes(hashes.iter().map(|hash| utils::load_hash(hash).unwrap()));
}

/// Transcodes the protobuf of a [PLONK proof](`proto::Proof`) to SCPF, with the layout specified in
/// `SCPF.md`.
///
/// REQUIRES: `proof` must come from [`Proof::to_proto`](`crate::Proof::to_proto`).
pub(crate) fn encode_proof<F: PrimeField256>(proof: &proto::Proof) -> Vec<u8> {
    let commitment = proof.commitment.as_ref().unwrap();
    let inner_proof = proof.inner_proof.as_ref().unwrap();

    let mut encoder = Encoder::default();
    encoder.push_scalars(
        proof
            .public_inputs
            .iter()
            .map(|input| load_scalar::<F>(&input.value)),
    );
    encoder.push_hashes(
        commitment
            .batches
            .iter()
            .skip(1) // skip the circuit batch
            .map(|batch| utils::load_hash(&batch.root).unwrap()),
    );
    encoder.push_u64(inner_proof.security_bits.into());
    encoder.push_u64(inner_proof.grinding_bits.into());
    encoder.push_u64(inner_proof.nonce);
    for point in &inner_proof.points {
        push_scalars::<F>(&mut encoder, &point.v);
    }
    push_hashes(&mut encoder, &inner_proof.fri_roots);
    encoder.push_usize(inner_proof.folded_quotient.len());
    push_scalars::<F>(&mut encoder, &inner_proof.folded_quotient);
    for (opening, query) in inner_proof.openings.iter().zip(&inner_proof.queries) {
        for leaf_proof in &opening.proofs {
            push_scalars::<F>(&mut encoder, &leaf_proof.leaf_values);
            push_hashes(&mut encoder, &leaf_proof.merkle_path);
        }
        for fold in &query.folds {
            push_scalars::<F>(&mut encoder, &fold.coset_values);
            push_hashes(&mut encoder, &fold.merkle_path);
        }
    }
    encoder.finish::<proto::Proof>()
}
