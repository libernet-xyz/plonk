# Starkom PLONK

[![CI](https://img.shields.io/github/actions/workflow/status/libernet-xyz/plonk/ci.yml?label=CI)](https://github.com/libernet-xyz/plonk/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/starkom-plonk)](https://crates.io/crates/starkom-plonk)
[![license](https://img.shields.io/crates/l/starkom-plonk)](https://github.com/libernet-xyz/plonk/blob/main/LICENSE)

## Overview

This repo contains Starkom's PLONK arithmetization and proving system.

The specific arithmetization type is PLONKish: it has an arbitrary number of columns, custom gate
constraints, and arbitrarily shaped gates. It's based on the DEEP-FRI polynomial commitment scheme
provided by the [`starkom-pcs`][starkom-pcs] crate, which is plausibly quantum-safe and requires no
trusted setup, so the whole Starkom engine is a transparent (aka "zkSTARK") proving system.

Zero-knowledge is optional in Starkom and you need to enable it explicitly in the
[`CircuitOptions::blind`][blind-option] flag when building your circuit.

The system is field-agnostic: it can run on any finite field as long as it has sufficient
two-adicity (for FRI to run) and implements the [`starkom-ff`][starkom-ff] interface.

Proofs can be serialized to and deserialized from [protobufs][protobuf]. Proofs over ~256-bit prime
fields (e.g. [BlueSky][bluesky]) can also be serialized to [SCPF][starkom-scpf], the Starkom Compact
Proof Format, which minimizes the gas cost for verification on the Ethereum Virtual Machine (see
[`SCPF.md`][scpf-layout] for the layout).

[blind-option]: https://docs.rs/starkom-plonk/latest/starkom_plonk/struct.CircuitOptions.html#structfield.blind
[bluesky]: https://crates.io/crates/starkom-bluesky
[protobuf]: https://protobuf.dev/
[scpf-layout]: https://github.com/libernet-xyz/plonk/blob/main/SCPF.md
[starkom-ff]: https://crates.io/crates/starkom-ff
[starkom-pcs]: https://crates.io/crates/starkom-pcs
[starkom-scpf]: https://crates.io/crates/starkom-scpf
