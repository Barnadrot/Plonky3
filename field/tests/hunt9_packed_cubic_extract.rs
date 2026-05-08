//! hunt-9: PackedFieldExtension::extract on PackedCubic should return
//! lane i's value correctly, even though PackedValue::extract (which
//! routes through the broken as_slice) does not.
//!
//! Both PackedValue::extract (default impl `self.as_slice()[lane]`) and
//! PackedFieldExtension::extract (default impl using
//! `as_basis_coefficients_slice`) coexist on PackedCubic. The call
//! `pkd.extract(lane)` is therefore potentially ambiguous; this test
//! exercises the unambiguous trait-method paths and asserts only the
//! `PackedFieldExtension` route is correct.
//!
//! Hypothesis: `PackedValue::extract` returns garbled data (same root
//! cause as bh3-5); `PackedFieldExtension::extract` returns the right
//! lane value.

use p3_field::extension::{CubicTrinomialExtensionField, PackedCubicTrinomialExtensionField};
use p3_field::integers::QuotientMap;
use p3_field::{PackedFieldExtension, PackedValue};
use p3_goldilocks::Goldilocks;

type F = Goldilocks;
type PF = <F as p3_field::Field>::Packing;
type CExt = CubicTrinomialExtensionField<F>;
type PCExt = PackedCubicTrinomialExtensionField<F, PF>;

fn ce(a: u64, b: u64, c: u64) -> CExt {
    CExt::new([F::from_int(a), F::from_int(b), F::from_int(c)])
}

#[test]
fn packed_field_extension_extract_is_correct() {
    let width = PCExt::WIDTH;
    let inputs: Vec<CExt> = (0..width)
        .map(|i| ce((i * 100 + 1) as u64, (i * 100 + 2) as u64, (i * 100 + 3) as u64))
        .collect();
    let packed = PCExt::from_fn(|i| inputs[i]);

    for lane in 0..width {
        let extracted = <PCExt as PackedFieldExtension<F, CExt>>::extract(&packed, lane);
        assert_eq!(
            extracted, inputs[lane],
            "PackedFieldExtension::extract returns wrong lane value at lane {lane}"
        );
    }
}
