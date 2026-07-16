use crate::plonk::CircuitView;
use crate::witness::{Cell, CellOrUnconstrained, WitnessView};
use anyhow::Result;

/// Represents a reusable PLONK chip that you can use to build circuits.
pub trait Chip<const I: usize, const O: usize> {
    fn build(
        &self,
        builder: &mut impl CircuitView,
        inputs: [Option<Cell>; I],
    ) -> Result<[Option<Cell>; O]>;

    fn witness(
        &self,
        witness: &mut impl WitnessView,
        inputs: [CellOrUnconstrained; I],
    ) -> Result<[CellOrUnconstrained; O]>;
}
