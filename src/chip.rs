use crate::plonk::CircuitView;
use crate::witness::{Cell, CellOrUnconstrained, WitnessView};
use anyhow::Result;
use starkom_ff::{Field, Field256};

/// Represents a reusable PLONK chip that you can use to build circuits.
///
/// The generic parameter `I` is the number of inputs and `O` is the number of outputs.
pub trait Chip<F: Field, const I: usize, const O: usize> {
    /// Returns the number of columns required by the chip.
    fn width(&self) -> usize;

    /// Returns the number of rows used by the chip.
    fn height(&self) -> usize;

    /// Builds the chip on the provided [`CircuitView`].
    fn build<G: Field256 + From<F>>(
        &self,
        view: &mut impl CircuitView<F, G>,
        inputs: [Option<Cell>; I],
    ) -> Result<[Option<Cell>; O]>;

    /// Witnesses execution of the chip in the provided [`WitnessView`].
    fn witness(
        &self,
        view: &mut impl WitnessView<F>,
        inputs: [CellOrUnconstrained<F>; I],
    ) -> Result<[CellOrUnconstrained<F>; O]>;
}

/// A reusable chip whose number of inputs and outputs is not known at compile time.
///
/// Note that dynamically sized circuits need to be recompiled on both the prover and verifier side
/// for every different shape used, so it's recommended to avoid this kind of chips.
pub trait DynamicChip<F: Field> {
    /// Returns the number of columns required by the chip.
    fn width(&self) -> usize;

    /// Returns the number of rows used by the chip.
    fn height(&self) -> usize;

    /// Builds the chip on the provided [`CircuitView`].
    fn build<G: Field256 + From<F>>(
        &self,
        view: &mut impl CircuitView<F, G>,
        inputs: Vec<Option<Cell>>,
    ) -> Result<Vec<Option<Cell>>>;

    /// Witnesses execution of the chip in the provided [`WitnessView`].
    fn witness(
        &self,
        view: &mut impl WitnessView<F>,
        inputs: Vec<CellOrUnconstrained<F>>,
    ) -> Result<Vec<CellOrUnconstrained<F>>>;
}
