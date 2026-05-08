//! hunt-1: cross-check the new `mixed_dot_product` override on
//! `PackedBinomialExtensionField` against per-lane scalar `BinomialExtensionField`
//! arithmetic with adversarial mixed-lane inputs.
//!
//! Hypothesis: the override (#1597) decomposes the extension dot product
//! coordinate-wise into D base-level dot products, each delegating to
//! `PF::dot_product::<N>`. If the per-coordinate decomposition or the
//! per-lane lane-wise interpretation has any off-by-one, sign error, or
//! mis-broadcast, lane-mixed boundary values would expose it where the
//! existing all-lanes-broadcast tests would not.
//!
//! What "boundary" means here:
//!   - lane 0: zeros
//!   - lane 1: one (Montgomery rep different from canonical)
//!   - lane 2: P-1
//!   - lane 3..: random non-trivial values
//! Inputs are constructed so that no two lanes have identical values, so
//! a bug that swaps or mis-broadcasts lanes would show up.
//!
//! If the override is correct, the result must match the default lane-wise
//! computation (computed via `BinomialExtensionField`'s scalar `mixed_dot_product`).
#![cfg(any(target_feature = "avx2", target_feature = "avx512f"))]

use p3_baby_bear::BabyBear;
use p3_field::extension::{BinomialExtensionField, PackedBinomialExtensionField};
use p3_field::integers::QuotientMap;
use p3_field::{
    Algebra, BasedVectorSpace, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type PF = <F as Field>::Packing;
type PEF = PackedBinomialExtensionField<F, PF, 4>;

const N: usize = 8;

fn extract_lane_ext(p: &PEF, lane: usize) -> EF {
    EF::from_basis_coefficients_fn(|d| {
        <PEF as BasedVectorSpace<PF>>::as_basis_coefficients_slice(p)[d].as_slice()[lane]
    })
}

fn extract_lane_pf(p: &PF, lane: usize) -> F {
    p.as_slice()[lane]
}

#[test]
fn mixed_dot_product_lane_mixed_boundary_vs_scalar() {
    let width = PF::WIDTH;
    let mut rng = SmallRng::seed_from_u64(0xBABE_BEEF);

    // Boundary scalar values per lane.
    let boundary_lanes: Vec<F> = {
        let mut v = vec![F::ZERO, F::ONE, F::NEG_ONE, F::TWO];
        while v.len() < width {
            v.push(F::from_int(rng.next_u32() as u64));
        }
        v.truncate(width);
        v
    };

    // Construct N adversarial extension inputs: each `a[i]` is a packed
    // extension element where each lane carries an independent EF value
    // with at least one boundary coord per lane.
    let a_packed: [PEF; N] = core::array::from_fn(|i| {
        // For each lane, build a per-lane EF with each coord chosen from
        // a mix of boundaries and random values.
        let lane_efs: Vec<EF> = (0..width)
            .map(|lane| {
                EF::from_basis_coefficients_fn(|d| match (i + lane + d) % 5 {
                    0 => boundary_lanes[lane],
                    1 => F::ZERO,
                    2 => F::ONE,
                    3 => F::NEG_ONE,
                    _ => F::from_int(rng.next_u32() as u64),
                })
            })
            .collect();
        PEF::from_ext_slice(&lane_efs)
    });

    // Construct N adversarial base packed inputs.
    let f_packed: [PF; N] = core::array::from_fn(|i| {
        PF::from_fn(|lane| match (i + lane) % 4 {
            0 => boundary_lanes[lane],
            1 => F::ZERO,
            2 => F::NEG_ONE,
            _ => F::from_int(rng.next_u32() as u64),
        })
    });

    // Override path: <PEF as Algebra<PF>>::mixed_dot_product
    let result_override: PEF = <PEF as Algebra<PF>>::mixed_dot_product::<N>(&a_packed, &f_packed);

    // Reference: per-lane scalar EF computation.
    for lane in 0..width {
        let a_scalar: [EF; N] = core::array::from_fn(|i| extract_lane_ext(&a_packed[i], lane));
        let f_scalar: [F; N] = core::array::from_fn(|i| extract_lane_pf(&f_packed[i], lane));
        let expected: EF = <EF as Algebra<F>>::mixed_dot_product::<N>(&a_scalar, &f_scalar);

        let actual = extract_lane_ext(&result_override, lane);

        assert_eq!(
            actual, expected,
            "lane {lane}: override = {actual:?}, scalar = {expected:?}"
        );
    }
}
