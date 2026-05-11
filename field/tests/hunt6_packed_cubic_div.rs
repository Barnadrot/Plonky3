//! hunt-6: `PackedCubicTrinomialExtensionField::div(self, rhs)` must
//! agree with per-lane scalar division.
//!
//! The Div impl computes `Self::from_fn(|i| rhs.as_slice()[i].inverse())`,
//! which depends on `as_slice()` returning the lane-i scalar value. As
//! shown by hunt-5, `as_slice` violates the SoA layout — its raw
//! transmute interprets `[PF; 3]` (SoA) as `[CExt; W]` (AoS), so
//! `as_slice()[i]` returns a garbled triple instead of lane i's value.
//!
//! Hypothesis: the per-lane inverse is computed on the wrong scalar, so
//! the resulting Self / Self disagrees with the lane-wise
//! `numer.lane(i) / denom.lane(i)` reference for any input where the
//! garbled triple differs from lane i.

#[allow(unused_imports)]
use p3_field::PrimeField64;
use p3_field::extension::{CubicTrinomialExtensionField, PackedCubicTrinomialExtensionField};
use p3_field::integers::QuotientMap;
use p3_field::{BasedVectorSpace, PackedValue};
use p3_goldilocks::Goldilocks; // for as_canonical_u64 if needed

type F = Goldilocks;
type PF = <F as p3_field::Field>::Packing;
type CExt = CubicTrinomialExtensionField<F>;
type PCExt = PackedCubicTrinomialExtensionField<F, PF>;

fn ce(a: u64, b: u64, c: u64) -> CExt {
    CExt::new([F::from_int(a), F::from_int(b), F::from_int(c)])
}

#[test]
fn div_lane_matches_scalar_div() {
    let width = PCExt::WIDTH;

    // Per-lane distinct nonzero numerator and denominator values.
    let nums: Vec<CExt> = (0..width)
        .map(|i| ce((i + 1) as u64, (2 * i + 1) as u64, (3 * i + 1) as u64))
        .collect();
    let dens: Vec<CExt> = (0..width)
        .map(|i| {
            ce(
                (10 * i + 7) as u64,
                (10 * i + 11) as u64,
                (10 * i + 13) as u64,
            )
        })
        .collect();

    let p_num = PCExt::from_fn(|i| nums[i]);
    let p_den = PCExt::from_fn(|i| dens[i]);
    let p_quot = p_num / p_den;

    // Reference: per-lane scalar division. Extract lane i by gathering
    // scalar coords from the SoA arrays directly (this is the trustworthy
    // path; do not use as_slice on PCExt).
    for lane in 0..width {
        let num_lane = nums[lane];
        let den_lane = dens[lane];
        let expected = num_lane / den_lane;
        // Pull lane `lane` directly from the SoA basis-coefficient arrays.
        let basis = <PCExt as BasedVectorSpace<PF>>::as_basis_coefficients_slice(&p_quot);
        let actual = CExt::new([
            basis[0].as_slice()[lane],
            basis[1].as_slice()[lane],
            basis[2].as_slice()[lane],
        ]);
        assert_eq!(
            actual, expected,
            "lane {lane}: packed Div produced {actual:?}, scalar gives {expected:?}"
        );
    }
}
