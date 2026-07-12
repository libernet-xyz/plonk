use crate::expr::Constraint;
use crate::utils::isize_to_scalar;
use crate::witness::{Cell, Partitioner};
use starkom_bluesky::Scalar;
use starkom_poly;
use std::collections::{BTreeMap, BTreeSet};

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Default blowup factor (16) in logarithmic form.
///
/// Used with the underlying PCS to compute low-degree extensions.
pub const OPTIONS_DEFAULT_BLOWUP_LOG2: usize = 4;

#[inline]
pub fn var(column_index: usize, rotation: isize) -> Constraint {
    Constraint::make_var(column_index, rotation)
}

#[inline]
pub fn make_const(value: isize) -> Constraint {
    Constraint::make_const(isize_to_scalar(value))
}

#[inline]
pub const fn cell(row: usize, column: usize) -> Cell {
    Cell::new(row, column)
}

/// Circuit compilation & proving options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilationOptions {
    /// Converts all constraints to canonical form using [`Constraint::canonicalize`].
    ///
    /// When disabled, proving errors out rather than attempting canonicalization if there are
    /// negative exponents.
    ///
    /// Canonicalization is carried out inside [`CircuitBuilder::build`].
    ///
    /// WARNING: canonicalized constraints may be more permissive than their original form because a
    /// negative exponent requires the variable to be different from zero. Starkom does not allow
    /// proving with negative exponents, so enable this flag only if your circuit is correctly
    /// constrained even when those variables are zero.
    pub canonicalize_constraints: bool,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            canonicalize_constraints: false,
        }
    }
}

/// Circuit compilation & proving options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvingOptions {
    /// Log2 of the blowup factor used to compute the low-degree extensions for the underlying PCS.
    pub blowup_log2: usize,
}

impl Default for ProvingOptions {
    fn default() -> Self {
        Self {
            blowup_log2: OPTIONS_DEFAULT_BLOWUP_LOG2,
        }
    }
}

pub trait CircuitView {
    /// Returns the number of columns included in the view.
    ///
    /// If this is the root view, that is the raw [`CircuitBuilder`] instance, the width is
    /// unbounded and `None` is returned.
    fn width(&self) -> Option<usize>;

    /// INTERNAL ONLY. Not for public usage.
    fn add_gate_internal(&mut self, row: usize, column: usize, constraint: Constraint);

    /// Adds a gate to the circuit.
    fn add_gate(&mut self, row_offset: usize, constraint: Constraint) {
        self.add_gate_internal(row_offset, 0, constraint);
    }
}

/// Allows building PLONK [`Circuit`]s.
#[derive(Debug, Default, Clone)]
pub struct CircuitBuilder {
    /// Current number of rows in the circuit.
    num_rows: usize,

    /// Current number of columns in the circuit.
    num_columns: usize,

    /// The gates of the circuit, indexed by constraint.
    ///
    /// For every gate type (that is, for every unique gate constraint) this map associates the list
    /// of places where the gate has been instantiated. Such list is represented as an array of
    /// "root cells", each root cell being the reference cell for the gate instance: the row of the
    /// root cell corresponds to rotation 0 of all variables referenced by the gate, and the column
    /// corresponds to the column offset of the [`CircuitView`] used to instantiate the gate.
    ///
    /// NOTE: in order to minimize the number of different gate types stored in a circuit, the
    /// constraints stored in this map are _not_ [remapped](`Constraint::remap_variables`). This map
    /// basically keeps "raw gate types".
    gates: BTreeMap<Constraint, Vec<Cell>>,

    /// Cell partitioning inferred from the connections made with [`Self::connect`].
    partitioner: Partitioner,

    /// List of rows that are revealed in the proofs.
    public_rows: BTreeSet<usize>,
}

impl CircuitView for CircuitBuilder {
    fn width(&self) -> Option<usize> {
        None
    }

    fn add_gate_internal(&mut self, row: usize, column: usize, constraint: Constraint) {
        let root_cell = cell(row, column);
        match self.gates.get_mut(&constraint) {
            Some(root_cells) => {
                root_cells.push(root_cell);
            }
            None => {
                self.gates.insert(constraint, vec![root_cell]);
            }
        }
    }
}

#[derive(Debug)]
struct SubCircuitBuilder<'a, P: CircuitView> {
    parent: &'a mut P,
    row_offset: usize,
    column_offset: usize,
    width: usize,
}

impl<'a, P: CircuitView> CircuitView for SubCircuitBuilder<'a, P> {
    fn width(&self) -> Option<usize> {
        Some(self.width)
    }

    fn add_gate_internal(&mut self, row: usize, column: usize, constraint: Constraint) {
        self.parent.add_gate_internal(
            self.row_offset + row,
            self.column_offset + column,
            constraint,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
