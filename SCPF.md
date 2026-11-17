# SCPF Layout of PLONK Proofs

This document specifies the layout of the SCPF data of PLONK proofs, i.e. of the
`starkom.proto.plonk.v10.Proof` protobuf message, as encoded by [`Proof::to_scpf`][to-scpf]. The
binary format itself (the header, the pool, and the encoding of varints, inline words, and pool
references) is specified in [`FORMAT.md`][format] in the `scpf` repository.

Only proofs over prime fields of about 256 bits, such as BlueSky, can be encoded in SCPF. The hash
backend isn't encoded, so decoders must know it in advance; EVM verifiers use Keccak-256.

## Header

The layout identifier is `keccak256("starkom.proto.plonk.v10.Proof")`, or `781bb0c4b3513d20fb38966e93697ab9c21ea4d2e48cc19ae82d8d00cd1c015e`.

## Stream

The stream holds the following items, in this order:

| Items                                                   | Kind            | Count                                    |
| ------------------------------------------------------- | --------------- | ---------------------------------------- |
| Values of the public inputs                             | Inline words    | One for every public cell of the circuit |
| Roots of the witness, permutation, and quotient batches | Pool references | 3                                        |
| `security_bits` of the DEEP-FRI proof                   | Varint          | 1                                        |
| `grinding_bits` of the DEEP-FRI proof                   | Varint          | 1                                        |
| Proof-of-work nonce                                     | Varint          | 1                                        |
| Evaluations of the committed polynomials                | Inline words    | Number of polynomials × number of points |
| Merkle roots of the FRI folding rounds                  | Pool references | Number of rounds                         |
| Number of coefficients of the folded quotient           | Varint          | 1                                        |
| Coefficients of the folded quotient                     | Inline words    | As many as the previous item says        |
| Queries                                                 | See below       | Number of queries                        |

The public inputs are in increasing order of cell, i.e. by row and then by column, and the
evaluations are grouped by point, the points being in increasing order of X-coordinate. Every group
has the evaluations of all the committed polynomials in commitment order.

Every query holds, in order:

1. the opening of every batch in commitment order (circuit, witness, permutation argument, and
   quotient): the leaf values as inline words, one for every polynomial of the batch, followed by
   the Merkle path from the leaf up, excluding the root, as pool references;
2. the fold of every FRI round in order: the `2^fold_rate` values of the opened coset as inline
   words, in bit-reversed order, followed by the Merkle path from the root of the coset subtree up,
   excluding the root, as pool references.

## Derived values

Everything else is derived by the verifier, mostly from the circuit (see `CompressedCircuit`):

- The **public cells** are those of the circuit.
- There are always **4 batches**: circuit, witness, permutation argument, and quotient. The root of
  the circuit batch is the commitment of the circuit, so it's not encoded. The number of polynomials
  of every batch is:
  - circuit: one gate selector for every distinct selector index of the circuit, one sigma
    polynomial for every column, and the circuit area selector;
  - witness: one polynomial for every column (witness randomizers are never needed over a prime
    field);
  - permutation argument: the accumulator and one helper for every column;
  - quotient: the chunks of the quotient, plus one FRI randomizer if the circuit is blinded.
- The **protocol parameters** other than `security_bits` and `grinding_bits`, i.e. `degree_bound`,
  `blowup_log2`, `fold_rate`, and `max_rounds`, are those of the circuit. `security_bits` and
  `grinding_bits` must be at least those of the circuit.
- The **opened points** are `xi * omega^r` for every rotation `r` of the circuit (which always
  include 0 and 1), `omega` being the generator of the evaluation domain and `xi` the Fiat-Shamir
  challenge derived from the transcript hash of the 4 Merkle roots. Their evaluations are matched to
  them by sorting them by increasing value.
- The **number of FRI rounds** is `min(max_rounds, floor(log2(degree_bound) / fold_rate))`.
- The **number of queries** is `max(1, ceil((security_bits - grinding_bits) / blowup_log2))`, with
  the subtraction clamped to 0.
- The **query indices** are derived from the proof of work, like in the DEEP-FRI verifier.
- The **Merkle paths** of the batch openings have `log2(N)` hashes, `N` being the size of the
  extended evaluation domain (`degree_bound << blowup_log2`), and the Merkle path of the fold of
  round `i` (starting from 0) has `log2(N) - (i + 1) * fold_rate` hashes.

## Relation with the protobuf

`Proof::to_scpf` serializes the proof to its protobuf and transcodes the protobuf, so the stream
follows the fields of `starkom.proto.plonk.v10.Proof` and of the `starkom.proto.pcs.v10.Commitment`
and `starkom.proto.pcs.v10.Proof` messages that it contains, with two differences:

- the fields that the verifier derives aren't encoded: the hash backends, the circuit root and the
  `base_field` and `num_polys` fields of the batches, `degree_bound`, `blowup_log2`, `fold_rate`,
  `max_rounds`, the X-coordinates `z` of the points, the cells of the public inputs, and the query
  indices;
- the batch openings and the FRI queries are interleaved rather than following each other, so that
  a verifier can check every query as soon as it reads it.

Decoders rely on the shape of the proof that they derive from the circuit, so the data of a proof
that doesn't have it, e.g. one with extra public inputs or opened points, doesn't decode to the same
proof.

[format]: https://github.com/libernet-xyz/scpf/blob/main/FORMAT.md
[to-scpf]: https://docs.rs/starkom-plonk/latest/starkom_plonk/struct.Proof.html#method.to_scpf
