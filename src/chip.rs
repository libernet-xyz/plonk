use crate::plonk::CircuitView;
use crate::witness::{Cell, CellOrUnconstrained, WitnessView};
use anyhow::Result;

/// Represents a reusable PLONK chip that you can use to build circuits.
///
/// The generic parameter `I` is the number of inputs and `O` is the number of outputs.
pub trait Chip<const I: usize, const O: usize> {
    /// Returns the number of columns required by the chip.
    fn width(&self) -> usize;

    /// Builds the chip on the provided [`CircuitView`].
    fn build(
        &self,
        builder: &mut impl CircuitView,
        inputs: [Option<Cell>; I],
    ) -> Result<[Option<Cell>; O]>;

    /// Witnesses execution of the chip in the provided [`WitnessView`].
    fn witness(
        &self,
        witness: &mut impl WitnessView,
        inputs: [CellOrUnconstrained; I],
    ) -> Result<[CellOrUnconstrained; O]>;
}
