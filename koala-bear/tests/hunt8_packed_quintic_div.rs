//! hunt-8: PackedQuinticTrinomialExtensionField::div(Self, Self) had the
//! same SoA/AoS layout bug as PackedCubic (bh3-6). Verify the fix
//! produces correct lane-wise division.
//!
//! Quintic packed field is implemented for KoalaBear (the implemented
//! QuinticTrinomialExtendable type).

use p3_field::extension::{PackedQuinticTrinomialExtensionField, QuinticTrinomialExtensionField};
use p3_field::integers::QuotientMap;
use p3_field::{BasedVectorSpace, Field, PackedValue};
use p3_koala_bear::KoalaBear;

type F = KoalaBear;
type PF = <F as Field>::Packing;
type QExt = QuinticTrinomialExtensionField<F>;
type PQExt = PackedQuinticTrinomialExtensionField<F, PF>;

fn qe(a: u64, b: u64, c: u64, d: u64, e: u64) -> QExt {
    QExt::new([
        F::from_int(a),
        F::from_int(b),
        F::from_int(c),
        F::from_int(d),
        F::from_int(e),
    ])
}

#[test]
fn quintic_div_lane_matches_scalar_div() {
    let width = PQExt::WIDTH;

    // Per-lane distinct nonzero numerator and denominator values.
    let nums: Vec<QExt> = (0..width)
        .map(|i| {
            qe(
                (i + 1) as u64,
                (2 * i + 1) as u64,
                (3 * i + 1) as u64,
                (5 * i + 1) as u64,
                (7 * i + 1) as u64,
            )
        })
        .collect();
    let dens: Vec<QExt> = (0..width)
        .map(|i| {
            qe(
                (10 * i + 7) as u64,
                (10 * i + 11) as u64,
                (10 * i + 13) as u64,
                (10 * i + 17) as u64,
                (10 * i + 19) as u64,
            )
        })
        .collect();

    let p_num = PQExt::from_fn(|i| nums[i]);
    let p_den = PQExt::from_fn(|i| dens[i]);
    let p_quot = p_num / p_den;

    let basis = <PQExt as BasedVectorSpace<PF>>::as_basis_coefficients_slice(&p_quot);

    for lane in 0..width {
        let expected = nums[lane] / dens[lane];
        let actual = QExt::new([
            basis[0].as_slice()[lane],
            basis[1].as_slice()[lane],
            basis[2].as_slice()[lane],
            basis[3].as_slice()[lane],
            basis[4].as_slice()[lane],
        ]);
        assert_eq!(
            actual, expected,
            "lane {lane}: packed Div produced {actual:?}, scalar gives {expected:?}"
        );
    }
}
