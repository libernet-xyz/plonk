use crate::plonk::CircuitView;
use crate::witness::{Cell, CellOrUnconstrained, Witness};
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
        witness: &mut Witness,
        inputs: [CellOrUnconstrained; I],
    ) -> Result<[CellOrUnconstrained; O]>;
}
