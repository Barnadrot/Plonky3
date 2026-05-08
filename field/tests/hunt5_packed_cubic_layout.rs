//! hunt-5: `PackedCubicTrinomialExtensionField::as_slice` and `from_fn`
//! must agree on the lane layout.
//!
//! The struct is `repr(transparent)` over `[PF; 3]` where `PF` is a
//! packed base field of `WIDTH` lanes, so its memory layout is
//! `[[F; W]; 3]` (struct-of-arrays). Per-lane reads are scattered:
//! lane `l` lives at offsets `(0, W, 2W) + l`.
//!
//! `from_fn(|i| f(i))` correctly writes that SoA layout — it fills
//! `result.value[j].as_slice_mut()[i] = val.value[j]` for each lane.
//!
//! But `as_slice` does a raw transmute:
//!   `slice::from_raw_parts(self as *const Self as *const Self::Value, WIDTH)`
//! which interprets the SoA buffer as `[[F; 3]; W]` (array-of-structs).
//!
//! Hypothesis: the as_slice transmute violates the SoA layout. Round-tripping
//! `from_fn → as_slice` will return garbled values, not the lane inputs.
//! This is also the path used inside the `Div` impl
//!   `let rhs_inv = Self::from_fn(|i| rhs.as_slice()[i].inverse());`
//! so a layout mismatch turns `Div` lane-wise wrong.

use p3_field::PackedValue;
use p3_field::extension::{CubicTrinomialExtensionField, PackedCubicTrinomialExtensionField};
use p3_field::integers::QuotientMap;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;
type PF = <F as p3_field::Field>::Packing;
type CExt = CubicTrinomialExtensionField<F>;
type PCExt = PackedCubicTrinomialExtensionField<F, PF>;

fn ce(a: u32, b: u32, c: u32) -> CExt {
    CExt::new([F::from_int(a as u64), F::from_int(b as u64), F::from_int(c as u64)])
}

#[test]
fn from_fn_then_as_slice_roundtrip() {
    let width = PCExt::WIDTH;
    // Per-lane distinct cubic extension values.
    let inputs: Vec<CExt> = (0..width)
        .map(|i| ce((i * 100 + 1) as u32, (i * 100 + 2) as u32, (i * 100 + 3) as u32))
        .collect();

    let packed = PCExt::from_fn(|i| inputs[i]);
    let slice = packed.as_slice();

    for (lane, expected) in inputs.iter().enumerate() {
        assert_eq!(
            slice[lane], *expected,
            "lane {lane}: as_slice returned {:?}, expected {:?}",
            slice[lane], expected
        );
    }
}
