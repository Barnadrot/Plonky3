//! hunt-2: packed `sumcheck_coefficients_prefix` should agree with scalar
//! after lane-summing.
//!
//! Hypothesis: the K=8 chunked path with the new
//! `mixed_dot_product` override (#1597) might produce a result that does
//! not match the scalar implementation when summed across lanes. Possible
//! bug shapes:
//!   - tail residual handling has off-by-one when `half % K != 0`,
//!   - chunk boundary loses the (w_hi - w_lo) sign on a specific tile,
//!   - the diffs_e/diffs_w materialisation accidentally subtracts in the
//!     wrong order (leading-coefficient sign flip).
//!
//! Test strategy: choose total scalar length n = 2 * WIDTH * num_packed
//! where num_packed is chosen so that `half = num_packed/2` exercises
//! BOTH the `half % K == 0` (no tail) AND `half % K != 0` (with tail)
//! cases. Compare:
//!   scalar (h0, hinf)  =  sumcheck_coefficients_prefix(scalar_evals, scalar_weights)
//!   packed (h0p, hinfp) = sumcheck_coefficients_prefix(packed_evals, packed_weights)
//! and assert sum(h0p.lanes) == h0 and sum(hinfp.lanes) == hinf.

use p3_baby_bear::BabyBear;
use p3_field::extension::{BinomialExtensionField, PackedBinomialExtensionField};
use p3_field::integers::QuotientMap;
use p3_field::{
    BasedVectorSpace, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_whir::sumcheck::strategy::sumcheck_coefficients_prefix;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type FP = <F as Field>::Packing;
type EFP = PackedBinomialExtensionField<F, FP, 4>;

fn pack_evals(evals: &[F]) -> Vec<FP> {
    FP::pack_slice(evals).to_vec()
}

fn pack_weights(weights: &[EF]) -> Vec<EFP> {
    weights
        .chunks(FP::WIDTH)
        .map(EFP::from_ext_slice)
        .collect()
}

fn lane_sum_ef(packed: EFP) -> EF {
    let mut s = EF::ZERO;
    for lane in 0..FP::WIDTH {
        s += EF::from_basis_coefficients_fn(|d| {
            <EFP as BasedVectorSpace<FP>>::as_basis_coefficients_slice(&packed)[d].as_slice()[lane]
        });
    }
    s
}

fn run_one(num_packed_pairs: usize, seed: u64) {
    // Total scalar length: 2 * WIDTH * num_packed_pairs.
    // half (in packed elements) = num_packed_pairs.
    let width = FP::WIDTH;
    let n = 2 * width * num_packed_pairs;
    let mut rng = SmallRng::seed_from_u64(seed);

    let scalar_evals: Vec<F> = (0..n).map(|_| F::from_int(rng.next_u32() as u64)).collect();
    let scalar_weights: Vec<EF> = (0..n).map(|_| rng.random()).collect();

    let packed_evals = pack_evals(&scalar_evals);
    let packed_weights = pack_weights(&scalar_weights);

    let (h0, hinf) = sumcheck_coefficients_prefix(scalar_evals.as_slice(), scalar_weights.as_slice());
    let (h0p, hinfp) = sumcheck_coefficients_prefix(packed_evals.as_slice(), packed_weights.as_slice());

    let h0p_summed = lane_sum_ef(h0p);
    let hinfp_summed = lane_sum_ef(hinfp);

    assert_eq!(
        h0, h0p_summed,
        "h0 mismatch (num_packed_pairs={num_packed_pairs}, n={n})"
    );
    assert_eq!(
        hinf, hinfp_summed,
        "hinf mismatch (num_packed_pairs={num_packed_pairs}, n={n})"
    );
}

#[test]
fn sumcheck_prefix_packed_vs_scalar_no_tail() {
    // half % K == 0 cases (K=8): num_packed_pairs in {8, 16}.
    run_one(8, 0xA1);
    run_one(16, 0xA2);
}

#[test]
fn sumcheck_prefix_packed_vs_scalar_with_tail_1() {
    // half % K = 1: num_packed_pairs = 9.
    run_one(9, 0xB1);
}

#[test]
fn sumcheck_prefix_packed_vs_scalar_with_tail_7() {
    // half % K = 7: num_packed_pairs = 7. Pure-tail, no main body.
    run_one(7, 0xC1);
}

#[test]
fn sumcheck_prefix_packed_vs_scalar_with_tail_mixed() {
    // half % K = 3: num_packed_pairs = 11 (8 main + 3 tail).
    run_one(11, 0xD1);
    // half % K = 5: num_packed_pairs = 13 (8 main + 5 tail).
    run_one(13, 0xD2);
}

#[test]
fn sumcheck_prefix_packed_vs_scalar_par_threshold() {
    // half > 2^14 to force par path.
    // half = 2^14 + 1 = 16385, total scalar = 2 * WIDTH * 16385.
    run_one(16385, 0xE1);
}
