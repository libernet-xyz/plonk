use crate::circuit::{Wire, WireOrUnconstrained, padded_size};
use starkom_bluesky::Scalar;
use starkom_ff::Field;

#[derive(Debug, Clone)]
pub struct Witness {
    pub(crate) size: usize,
    pub(crate) gate_counter: usize,
    pub(crate) left: Vec<Scalar>,
    pub(crate) right: Vec<Scalar>,
    pub(crate) out: Vec<Scalar>,
}

impl PartialEq for Witness {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
            && self.left == other.left
            && self.right == other.right
            && self.out == other.out
    }
}

impl Eq for Witness {}

impl Witness {
    pub fn new(size: usize) -> Self {
        let padded_size = padded_size(size);
        Self {
            size,
            gate_counter: 0,
            left: vec![Scalar::ZERO; padded_size],
            right: vec![Scalar::ZERO; padded_size],
            out: vec![Scalar::ZERO; padded_size],
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn get(&self, wire: Wire) -> Scalar {
        match wire {
            Wire::LeftIn(index) => self.left[index as usize],
            Wire::RightIn(index) => self.right[index as usize],
            Wire::Out(index) => self.out[index as usize],
        }
    }

    pub fn set(&mut self, wire: Wire, value: Scalar) {
        match wire {
            Wire::LeftIn(index) => self.left[index as usize] = value,
            Wire::RightIn(index) => self.right[index as usize] = value,
            Wire::Out(index) => self.out[index as usize] = value,
        };
    }

    pub fn copy(&mut self, from: WireOrUnconstrained, to: Wire) -> Scalar {
        let value = match from {
            WireOrUnconstrained::Wire(from) => self.get(from),
            WireOrUnconstrained::Unconstrained(value) => value,
        };
        self.set(to, value);
        value
    }

    pub fn pop_gate(&mut self) -> usize {
        let gate = self.gate_counter;
        self.gate_counter += 1;
        gate
    }

    pub fn nop(
        &mut self,
        lhs: WireOrUnconstrained,
        rhs: WireOrUnconstrained,
        out: WireOrUnconstrained,
    ) -> usize {
        let gate = self.pop_gate();
        self.copy(lhs, Wire::LeftIn(gate));
        self.copy(rhs, Wire::RightIn(gate));
        self.copy(out, Wire::Out(gate));
        gate
    }

    pub fn assert_constant(&mut self, value: Scalar) -> Wire {
        let wire = Wire::Out(self.pop_gate());
        self.set(wire, value);
        wire
    }

    pub fn add(&mut self, lhs: WireOrUnconstrained, rhs: WireOrUnconstrained) -> Wire {
        let gate = self.pop_gate();
        let lhs = self.copy(lhs, Wire::LeftIn(gate));
        let rhs = self.copy(rhs, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, lhs + rhs);
        out
    }

    pub fn add_const(&mut self, lhs: WireOrUnconstrained, rhs: Scalar) -> Wire {
        let gate = self.pop_gate();
        self.copy(lhs, Wire::LeftIn(gate));
        let lhs = self.copy(lhs, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, lhs + rhs);
        out
    }

    pub fn sub(&mut self, lhs: WireOrUnconstrained, rhs: WireOrUnconstrained) -> Wire {
        let gate = self.pop_gate();
        let lhs = self.copy(lhs, Wire::LeftIn(gate));
        let rhs = self.copy(rhs, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, lhs - rhs);
        out
    }

    pub fn sub_const(&mut self, lhs: WireOrUnconstrained, rhs: Scalar) -> Wire {
        let gate = self.pop_gate();
        self.copy(lhs, Wire::LeftIn(gate));
        let lhs = self.copy(lhs, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, lhs - rhs);
        out
    }

    pub fn sub_from_const(&mut self, lhs: Scalar, rhs: WireOrUnconstrained) -> Wire {
        let gate = self.pop_gate();
        self.copy(rhs, Wire::LeftIn(gate));
        let rhs = self.copy(rhs, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, lhs - rhs);
        out
    }

    pub fn mul(&mut self, lhs: WireOrUnconstrained, rhs: WireOrUnconstrained) -> Wire {
        let gate = self.pop_gate();
        let lhs = self.copy(lhs, Wire::LeftIn(gate));
        let rhs = self.copy(rhs, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, lhs * rhs);
        out
    }

    pub fn square(&mut self, wire: WireOrUnconstrained) -> Wire {
        let gate = self.pop_gate();
        let lhs = self.copy(wire, Wire::LeftIn(gate));
        let rhs = self.copy(wire, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, lhs * rhs);
        out
    }

    pub fn mul_by_const(&mut self, lhs: WireOrUnconstrained, rhs: Scalar) -> Wire {
        let gate = self.pop_gate();
        self.copy(lhs, Wire::LeftIn(gate));
        let lhs = self.copy(lhs, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, lhs * rhs);
        out
    }

    pub fn combine(
        &mut self,
        c1: Scalar,
        lhs: WireOrUnconstrained,
        c2: Scalar,
        rhs: WireOrUnconstrained,
    ) -> Wire {
        let gate = self.pop_gate();
        let lhs = self.copy(lhs, Wire::LeftIn(gate));
        let rhs = self.copy(rhs, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, c1 * lhs + c2 * rhs);
        out
    }

    pub fn poly2(
        &mut self,
        c1: Scalar,
        c2: Scalar,
        c3: Scalar,
        input: WireOrUnconstrained,
    ) -> Wire {
        let gate = self.pop_gate();
        self.copy(input, Wire::LeftIn(gate));
        let input = self.copy(input, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, c1 * input.square() + c2 * input + c3);
        out
    }

    pub fn assert_bit(&mut self, input: WireOrUnconstrained) {
        let gate = self.pop_gate();
        self.copy(input, Wire::LeftIn(gate));
        self.copy(input, Wire::RightIn(gate));
    }

    pub fn assert_trit(&mut self, input: WireOrUnconstrained) {
        let lhs = self.poly2(
            Scalar::from_const(1),
            -Scalar::from_const(3),
            Scalar::from_const(2),
            input,
        );
        let gate = self.pop_gate();
        self.copy(lhs.into(), Wire::LeftIn(gate));
        self.copy(input, Wire::RightIn(gate));
    }

    pub fn not(&mut self, input: WireOrUnconstrained) -> Wire {
        let gate = self.pop_gate();
        self.copy(input, Wire::LeftIn(gate));
        let input = self.copy(input, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, Scalar::from_const(1) - input);
        out
    }

    pub fn and(&mut self, lhs: WireOrUnconstrained, rhs: WireOrUnconstrained) -> Wire {
        let gate = self.pop_gate();
        let lhs = self.copy(lhs, Wire::LeftIn(gate));
        let rhs = self.copy(rhs, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, lhs * rhs);
        out
    }

    pub fn or(&mut self, lhs: WireOrUnconstrained, rhs: WireOrUnconstrained) -> Wire {
        let gate = self.pop_gate();
        let lhs = self.copy(lhs, Wire::LeftIn(gate));
        let rhs = self.copy(rhs, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, lhs + rhs - lhs * rhs);
        out
    }

    pub fn xor(&mut self, lhs: WireOrUnconstrained, rhs: WireOrUnconstrained) -> Wire {
        let gate = self.pop_gate();
        let lhs = self.copy(lhs, Wire::LeftIn(gate));
        let rhs = self.copy(rhs, Wire::RightIn(gate));
        let out = Wire::Out(gate);
        self.set(out, lhs + rhs - Scalar::from_const(2) * lhs * rhs);
        out
    }

    pub(crate) fn blind_row(&mut self) {
        let gate = self.pop_gate();
        self.set(Wire::LeftIn(gate), Scalar::random_default());
        self.set(Wire::RightIn(gate), Scalar::random_default());
        self.set(Wire::Out(gate), Scalar::random_default());
    }

    pub(crate) fn blind(&mut self) {
        self.blind_row();
        self.blind_row();
        self.blind_row();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starkom_bluesky::from_const;

    #[test]
    fn test_one_row_initial_state() {
        let witness = Witness::new(1);
        assert_eq!(witness.size(), 1);
        assert_eq!(witness.get(Wire::LeftIn(0)), from_const(0));
        assert_eq!(witness.get(Wire::RightIn(0)), from_const(0));
        assert_eq!(witness.get(Wire::Out(0)), from_const(0));
    }

    #[test]
    fn test_two_rows_initial_state() {
        let witness = Witness::new(2);
        assert_eq!(witness.size(), 2);
        assert_eq!(witness.get(Wire::LeftIn(0)), from_const(0));
        assert_eq!(witness.get(Wire::RightIn(0)), from_const(0));
        assert_eq!(witness.get(Wire::Out(0)), from_const(0));
        assert_eq!(witness.get(Wire::LeftIn(1)), from_const(0));
        assert_eq!(witness.get(Wire::RightIn(1)), from_const(0));
        assert_eq!(witness.get(Wire::Out(1)), from_const(0));
    }

    #[test]
    fn test_one_row_update() {
        let mut witness = Witness::new(1);
        witness.set(Wire::LeftIn(0), from_const(12));
        witness.set(Wire::RightIn(0), from_const(34));
        witness.set(Wire::Out(0), from_const(56));
        assert_eq!(witness.size(), 1);
        assert_eq!(witness.get(Wire::LeftIn(0)), from_const(12));
        assert_eq!(witness.get(Wire::RightIn(0)), from_const(34));
        assert_eq!(witness.get(Wire::Out(0)), from_const(56));
    }

    #[test]
    fn test_two_rows_update() {
        let mut witness = Witness::new(2);
        witness.set(Wire::LeftIn(0), from_const(65));
        witness.set(Wire::RightIn(0), from_const(43));
        witness.set(Wire::Out(0), from_const(21));
        witness.set(Wire::LeftIn(1), from_const(12));
        witness.set(Wire::RightIn(1), from_const(34));
        witness.set(Wire::Out(1), from_const(56));
        assert_eq!(witness.size(), 2);
        assert_eq!(witness.get(Wire::LeftIn(0)), from_const(65));
        assert_eq!(witness.get(Wire::RightIn(0)), from_const(43));
        assert_eq!(witness.get(Wire::Out(0)), from_const(21));
        assert_eq!(witness.get(Wire::LeftIn(1)), from_const(12));
        assert_eq!(witness.get(Wire::RightIn(1)), from_const(34));
        assert_eq!(witness.get(Wire::Out(1)), from_const(56));
    }

    #[test]
    fn test_copy_within_same_row() {
        let mut witness = Witness::new(1);
        witness.set(Wire::LeftIn(0), from_const(12));
        witness.set(Wire::RightIn(0), from_const(34));
        assert_eq!(
            witness.copy(Wire::RightIn(0).into(), Wire::Out(0)),
            from_const(34)
        );
        assert_eq!(witness.size(), 1);
        assert_eq!(witness.get(Wire::LeftIn(0)), from_const(12));
        assert_eq!(witness.get(Wire::RightIn(0)), from_const(34));
        assert_eq!(witness.get(Wire::Out(0)), from_const(34));
    }

    #[test]
    fn test_copy_across_rows() {
        let mut witness = Witness::new(2);
        witness.set(Wire::LeftIn(0), from_const(12));
        witness.set(Wire::RightIn(0), from_const(34));
        witness.set(Wire::Out(0), from_const(56));
        assert_eq!(
            witness.copy(Wire::RightIn(0).into(), Wire::LeftIn(1)),
            from_const(34)
        );
        assert_eq!(
            witness.copy(Wire::LeftIn(0).into(), Wire::RightIn(1)),
            from_const(12)
        );
        witness.set(Wire::Out(1), from_const(56));
        assert_eq!(witness.size(), 2);
        assert_eq!(witness.get(Wire::LeftIn(0)), from_const(12));
        assert_eq!(witness.get(Wire::RightIn(0)), from_const(34));
        assert_eq!(witness.get(Wire::Out(0)), from_const(56));
        assert_eq!(witness.get(Wire::LeftIn(1)), from_const(34));
        assert_eq!(witness.get(Wire::RightIn(1)), from_const(12));
        assert_eq!(witness.get(Wire::Out(1)), from_const(56));
    }

    fn test_assert_constant_impl(value: u64) {
        let mut witness = Witness::new(1);
        let wire = witness.assert_constant(value.into());
        assert_eq!(wire, Wire::Out(0));
        assert_eq!(witness.get(wire), value.into());
    }

    #[test]
    fn test_assert_constant() {
        test_assert_constant_impl(42);
        test_assert_constant_impl(43);
        test_assert_constant_impl(44);
    }

    fn test_add_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), lhs.into());
        witness.set(Wire::RightIn(0), rhs.into());
        assert_eq!(
            witness.add(Wire::LeftIn(0).into(), Wire::RightIn(0).into()),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(1)), rhs.into());
        assert_eq!(witness.get(Wire::Out(1)), out.into());
    }

    #[test]
    fn test_add() {
        test_add_impl(12, 34, 46);
        test_add_impl(34, 12, 46);
        test_add_impl(56, 78, 134);
    }

    fn test_unconstrained_add_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(1);
        assert_eq!(
            witness.add(
                WireOrUnconstrained::Unconstrained(lhs.into()),
                WireOrUnconstrained::Unconstrained(rhs.into())
            ),
            Wire::Out(0)
        );
        assert_eq!(witness.get(Wire::LeftIn(0)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(0)), rhs.into());
        assert_eq!(witness.get(Wire::Out(0)), out.into());
    }

    #[test]
    fn test_unconstrained_add() {
        test_unconstrained_add_impl(12, 34, 46);
        test_unconstrained_add_impl(34, 12, 46);
        test_unconstrained_add_impl(56, 78, 134);
    }

    fn test_add_const_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), lhs.into());
        assert_eq!(
            witness.add_const(Wire::LeftIn(0).into(), rhs.into()),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::Out(1)), out.into());
    }

    #[test]
    fn test_add_const() {
        test_add_const_impl(12, 34, 46);
        test_add_const_impl(34, 12, 46);
        test_add_const_impl(56, 78, 134);
    }

    fn test_unconstrained_add_const_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(1);
        witness.set(Wire::LeftIn(0), lhs.into());
        assert_eq!(
            witness.add_const(WireOrUnconstrained::Unconstrained(lhs.into()), rhs.into()),
            Wire::Out(0)
        );
        assert_eq!(witness.get(Wire::LeftIn(0)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(0)), lhs.into());
        assert_eq!(witness.get(Wire::Out(0)), out.into());
    }

    #[test]
    fn test_unconstrained_add_const() {
        test_unconstrained_add_const_impl(12, 34, 46);
        test_unconstrained_add_const_impl(34, 12, 46);
        test_unconstrained_add_const_impl(56, 78, 134);
    }

    fn test_sub_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), lhs.into());
        witness.set(Wire::RightIn(0), rhs.into());
        assert_eq!(
            witness.sub(Wire::LeftIn(0).into(), Wire::RightIn(0).into()),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(1)), rhs.into());
        assert_eq!(witness.get(Wire::Out(1)), out.into());
    }

    #[test]
    fn test_sub() {
        test_sub_impl(34, 12, 22);
        test_sub_impl(78, 56, 22);
        test_sub_impl(78, 34, 44);
    }

    fn test_unconstrained_sub_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(1);
        assert_eq!(
            witness.sub(
                WireOrUnconstrained::Unconstrained(lhs.into()),
                WireOrUnconstrained::Unconstrained(rhs.into())
            ),
            Wire::Out(0)
        );
        assert_eq!(witness.get(Wire::LeftIn(0)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(0)), rhs.into());
        assert_eq!(witness.get(Wire::Out(0)), out.into());
    }

    #[test]
    fn test_unconstrained_sub() {
        test_unconstrained_sub_impl(34, 12, 22);
        test_unconstrained_sub_impl(78, 56, 22);
        test_unconstrained_sub_impl(78, 34, 44);
    }

    fn test_sub_const_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), lhs.into());
        assert_eq!(
            witness.sub_const(Wire::LeftIn(0).into(), rhs.into()),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::Out(1)), out.into());
    }

    #[test]
    fn test_sub_const() {
        test_sub_const_impl(34, 12, 22);
        test_sub_const_impl(78, 56, 22);
        test_sub_const_impl(78, 34, 44);
    }

    fn test_unconstrained_sub_const_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(1);
        witness.set(Wire::LeftIn(0), lhs.into());
        assert_eq!(
            witness.sub_const(WireOrUnconstrained::Unconstrained(lhs.into()), rhs.into()),
            Wire::Out(0)
        );
        assert_eq!(witness.get(Wire::LeftIn(0)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(0)), lhs.into());
        assert_eq!(witness.get(Wire::Out(0)), out.into());
    }

    #[test]
    fn test_unconstrained_sub_const() {
        test_unconstrained_sub_const_impl(34, 12, 22);
        test_unconstrained_sub_const_impl(78, 56, 22);
        test_unconstrained_sub_const_impl(78, 34, 44);
    }

    fn test_sub_from_const_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::RightIn(0), rhs.into());
        assert_eq!(
            witness.sub_from_const(lhs.into(), Wire::RightIn(0).into()),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), rhs.into());
        assert_eq!(witness.get(Wire::RightIn(1)), rhs.into());
        assert_eq!(witness.get(Wire::Out(1)), out.into());
    }

    #[test]
    fn test_sub_from_const() {
        test_sub_from_const_impl(34, 12, 22);
        test_sub_from_const_impl(78, 56, 22);
        test_sub_from_const_impl(78, 34, 44);
    }

    fn test_unconstrained_sub_from_const_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(1);
        witness.set(Wire::LeftIn(0), lhs.into());
        assert_eq!(
            witness.sub_from_const(lhs.into(), WireOrUnconstrained::Unconstrained(rhs.into())),
            Wire::Out(0)
        );
        assert_eq!(witness.get(Wire::LeftIn(0)), rhs.into());
        assert_eq!(witness.get(Wire::RightIn(0)), rhs.into());
        assert_eq!(witness.get(Wire::Out(0)), out.into());
    }

    #[test]
    fn test_unconstrained_sub_from_const() {
        test_unconstrained_sub_from_const_impl(34, 12, 22);
        test_unconstrained_sub_from_const_impl(78, 56, 22);
        test_unconstrained_sub_from_const_impl(78, 34, 44);
    }

    fn test_mul_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), lhs.into());
        witness.set(Wire::RightIn(0), rhs.into());
        assert_eq!(
            witness.mul(Wire::LeftIn(0).into(), Wire::RightIn(0).into()),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(1)), rhs.into());
        assert_eq!(witness.get(Wire::Out(1)), out.into());
    }

    #[test]
    fn test_mul() {
        test_mul_impl(12, 34, 408);
        test_mul_impl(34, 12, 408);
        test_mul_impl(56, 78, 4368);
    }

    fn test_unconstrained_mul_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(1);
        assert_eq!(
            witness.mul(
                WireOrUnconstrained::Unconstrained(lhs.into()),
                WireOrUnconstrained::Unconstrained(rhs.into())
            ),
            Wire::Out(0)
        );
        assert_eq!(witness.get(Wire::LeftIn(0)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(0)), rhs.into());
        assert_eq!(witness.get(Wire::Out(0)), out.into());
    }

    #[test]
    fn test_unconstrained_mul() {
        test_unconstrained_mul_impl(12, 34, 408);
        test_unconstrained_mul_impl(34, 12, 408);
        test_unconstrained_mul_impl(56, 78, 4368);
    }

    fn test_square_impl(input: u64, output: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), input.into());
        assert_eq!(witness.square(Wire::LeftIn(0).into()), Wire::Out(1));
        assert_eq!(witness.get(Wire::LeftIn(1)), input.into());
        assert_eq!(witness.get(Wire::RightIn(1)), input.into());
        assert_eq!(witness.get(Wire::Out(1)), output.into());
    }

    #[test]
    fn test_square() {
        test_square_impl(0, 0);
        test_square_impl(1, 1);
        test_square_impl(2, 4);
        test_square_impl(3, 9);
    }

    fn test_mul_by_const_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), lhs.into());
        assert_eq!(
            witness.mul_by_const(Wire::LeftIn(0).into(), rhs.into()),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::Out(1)), out.into());
    }

    #[test]
    fn test_mul_by_const() {
        test_mul_by_const_impl(12, 34, 408);
        test_mul_by_const_impl(34, 12, 408);
        test_mul_by_const_impl(56, 78, 4368);
    }

    fn test_combine_impl(c1: u64, lhs: u64, c2: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), lhs.into());
        witness.set(Wire::RightIn(0), rhs.into());
        assert_eq!(
            witness.combine(
                c1.into(),
                Wire::LeftIn(0).into(),
                c2.into(),
                Wire::RightIn(0).into()
            ),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(1)), rhs.into());
        assert_eq!(witness.get(Wire::Out(1)), out.into());
    }

    #[test]
    fn test_combine() {
        test_combine_impl(1, 2, 3, 4, 14);
        test_combine_impl(5, 6, 7, 8, 86);
        test_combine_impl(12, 34, 56, 78, 4776);
        test_combine_impl(34, 12, 56, 78, 4776);
        test_combine_impl(12, 34, 78, 56, 4776);
        test_combine_impl(56, 78, 12, 34, 4776);
    }

    fn test_poly2_impl(input: Scalar, output: Scalar) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), input.into());
        assert_eq!(
            witness.poly2(from_const(12), from_const(34), from_const(56), input.into()),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), input.into());
        assert_eq!(witness.get(Wire::RightIn(1)), input.into());
        assert_eq!(witness.get(Wire::Out(1)), output.into());
    }

    #[test]
    fn test_poly2() {
        test_poly2_impl(from_const(42), from_const(22652));
        test_poly2_impl(from_const(43), from_const(23706));
    }

    fn test_assert_bit_impl(input: Scalar) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), input.into());
        witness.assert_bit(Wire::LeftIn(0).into());
        assert_eq!(witness.get(Wire::LeftIn(1)), input.into());
        assert_eq!(witness.get(Wire::RightIn(1)), input.into());
    }

    #[test]
    fn test_assert_bit() {
        test_assert_bit_impl(from_const(0));
        test_assert_bit_impl(from_const(1));
    }

    fn test_assert_trit_impl(input: Scalar) {
        let mut witness = Witness::new(3);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), input.into());
        witness.assert_trit(Wire::LeftIn(0).into());
        assert_eq!(witness.get(Wire::LeftIn(1)), input);
        assert_eq!(witness.get(Wire::RightIn(1)), input);
    }

    #[test]
    fn test_assert_trit() {
        test_assert_trit_impl(from_const(0));
        test_assert_trit_impl(from_const(1));
        test_assert_trit_impl(from_const(2));
    }

    fn test_not_impl(input: Scalar, output: Scalar) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), input.into());
        assert_eq!(witness.not(input.into()), Wire::Out(1));
        assert_eq!(witness.get(Wire::LeftIn(1)), input.into());
        assert_eq!(witness.get(Wire::RightIn(1)), input.into());
        assert_eq!(witness.get(Wire::Out(1)), output.into());
    }

    #[test]
    fn test_not() {
        test_not_impl(from_const(0), from_const(1));
        test_not_impl(from_const(1), from_const(0));
    }

    fn test_and_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), lhs.into());
        witness.set(Wire::RightIn(0), rhs.into());
        assert_eq!(
            witness.and(Wire::LeftIn(0).into(), Wire::RightIn(0).into()),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(1)), rhs.into());
        assert_eq!(witness.get(Wire::Out(1)), out.into());
    }

    #[test]
    fn test_and() {
        test_and_impl(0, 0, 0);
        test_and_impl(0, 1, 0);
        test_and_impl(1, 0, 0);
        test_and_impl(1, 1, 1);
    }

    fn test_or_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), lhs.into());
        witness.set(Wire::RightIn(0), rhs.into());
        assert_eq!(
            witness.or(Wire::LeftIn(0).into(), Wire::RightIn(0).into()),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(1)), rhs.into());
        assert_eq!(witness.get(Wire::Out(1)), out.into());
    }

    #[test]
    fn test_or() {
        test_or_impl(0, 0, 0);
        test_or_impl(0, 1, 1);
        test_or_impl(1, 0, 1);
        test_or_impl(1, 1, 1);
    }

    fn test_xor_impl(lhs: u64, rhs: u64, out: u64) {
        let mut witness = Witness::new(2);
        witness.pop_gate();
        witness.set(Wire::LeftIn(0), lhs.into());
        witness.set(Wire::RightIn(0), rhs.into());
        assert_eq!(
            witness.xor(Wire::LeftIn(0).into(), Wire::RightIn(0).into()),
            Wire::Out(1)
        );
        assert_eq!(witness.get(Wire::LeftIn(1)), lhs.into());
        assert_eq!(witness.get(Wire::RightIn(1)), rhs.into());
        assert_eq!(witness.get(Wire::Out(1)), out.into());
    }

    #[test]
    fn test_xor() {
        test_xor_impl(0, 0, 0);
        test_xor_impl(0, 1, 1);
        test_xor_impl(1, 0, 1);
        test_xor_impl(1, 1, 0);
    }
}
