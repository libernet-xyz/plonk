use crate::circuit::{Wire, WireOrUnconstrained};
use crate::plonk::CircuitBuilder;
use crate::witness::Witness;
use anyhow::Result;

/// Represents a reusable PLONK chip that you can use to build circuits.
pub trait Chip<const I: usize, const O: usize> {
    fn build(
        &self,
        builder: &mut CircuitBuilder,
        inputs: [Option<Wire>; I],
    ) -> Result<[Option<Wire>; O]>;

    fn witness(
        &self,
        witness: &mut Witness,
        inputs: [WireOrUnconstrained; I],
    ) -> Result<[WireOrUnconstrained; O]>;
}

/// A reusable PLONK chip with a variable number of inputs and outputs.
///
/// NOTE: PLONK circuits have a fixed structure, so the number of inputs and outputs must be known
/// at circuit build time; but this trait doesn't require knowing it when compiling the Rust source.
pub trait DynamicChip {
    fn build(
        &self,
        builder: &mut CircuitBuilder,
        inputs: &[Option<Wire>],
    ) -> Result<Vec<Option<Wire>>>;

    fn witness(
        &self,
        witness: &mut Witness,
        inputs: &[WireOrUnconstrained],
    ) -> Result<Vec<WireOrUnconstrained>>;
}
