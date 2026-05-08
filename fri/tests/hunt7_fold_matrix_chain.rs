//! hunt-7: `TwoAdicFriFolding::fold_matrix(log_arity=k)` should equal the
//! sequential chain `fold_matrix(log_arity=1, beta=beta)` →
//! `fold_matrix(log_arity=1, beta=beta^2)` → … on the same matrix.
//!
//! From the implementation comment:
//!   "an arity-2^k fold with a single challenge beta is equivalent to
//!    k arity-2 folds with challenges beta, beta^2, beta^4, ..."
//!
//! Hypothesis: the in-place update of `halve_inv_powers`
//!     halve_inv_powers[j] = two * halve_inv_powers[j << 1].square();
//! could be off-by-one or the bit-reversal could be inconsistent across
//! steps, breaking the chained equivalence.

use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriFoldingStrategy, TwoAdicFriFolding};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

fn rand_matrix(seed: u64, height: usize, width: usize) -> RowMajorMatrix<EF> {
    let mut rng = SmallRng::seed_from_u64(seed);
    RowMajorMatrix::new(
        (0..height * width).map(|_| rng.random()).collect(),
        width,
    )
}

#[test]
fn fold_matrix_arity4_equals_chain_arity2() {
    let folding: TwoAdicFriFolding<(), ()> = TwoAdicFriFolding(core::marker::PhantomData);

    // Matrix of height 2^6 (so we can fold by arity 2^2 = 4 once and end at
    // height 2^4) with width 4 (= log_arity=2 column count).
    let mat = rand_matrix(0xF010, 64, 4);

    let beta: EF = SmallRng::seed_from_u64(0xBEEF1).random();

    // Path A: single arity-4 fold with challenge beta.
    let single = <TwoAdicFriFolding<(), ()> as FriFoldingStrategy<F, EF>>::fold_matrix(&folding, beta, 2, mat.clone());

    // Path B: arity-2(beta) on a matrix of width=2, then arity-2(beta^2)
    // on the result. To compare apples-to-apples, restructure the input
    // matrix so the first arity-2 fold consumes columns 0..2, the second
    // consumes columns 2..4 — i.e., the same evaluation layout as arity-4.
    //
    // For our purposes, the simpler comparison is: fold each two-column
    // sub-matrix with the chained challenges. fold_matrix takes a (height,
    // width=arity) matrix. So:
    //   arity-4 input has width=4 columns (4 evaluations per row).
    //   The chained equivalent is: rebuild the matrix as height=2*64=128
    //   and width=2, fold once with beta to get height=64 width=1 ... but
    //   the API takes width=arity, not width=1. Let me re-read.
    //
    // Actually, fold_matrix takes an arity-column matrix and folds the
    // ARITY columns into ONE per row. So:
    //   arity-4: input mat (height=64, width=4) → output Vec<EF> of length 64.
    //   chained arity-2 first: split mat into upper half / lower half by ROWS?
    //
    // Reading fold_matrix more carefully (the log_arity > 1 branch):
    //   data = m.values  (length = height*width)
    //   step 0: data.len() = height*width, height_step = (height*width)/2.
    //           Consumes pairs (data[2i], data[2i+1]) → out[i].
    //
    // So fold_matrix treats data as a flat slice and folds adjacent pairs.
    // The "width" matters for HOW the matrix is row-organized for input,
    // but the algorithm collapses everything into a single 1D fold.
    //
    // For the chain comparison: arity-4 on (h=64, w=4) folds 64*4=256 values
    // into 64 values via 2 internal arity-2 steps with betas beta, beta^2.
    //
    // Equivalent: arity-2 on (h=128, w=2) folds 256 values into 128 via 1
    // arity-2 step with beta. Then arity-2 on (h=64, w=2) which expects
    // 64*2=128 values folded into 64 with beta^2.
    let _h = mat.height();
    let w = mat.width();
    let chain_input = RowMajorMatrix::new(mat.values.clone(), w / 2);
    let after_first = <TwoAdicFriFolding<(), ()> as FriFoldingStrategy<F, EF>>::fold_matrix(&folding, beta, 1, chain_input);
    // after_first is Vec<EF> of length h * w / 2 = 128.
    // Reshape into a w=2 matrix:
    let half_mat = RowMajorMatrix::new(after_first, 2);
    let chain = <TwoAdicFriFolding<(), ()> as FriFoldingStrategy<F, EF>>::fold_matrix(&folding, beta.square(), 1, half_mat);

    assert_eq!(single, chain, "arity-4(beta) != chain(arity-2(beta), arity-2(beta^2))");
}

#[test]
fn fold_matrix_arity8_equals_chain_arity2() {
    let folding: TwoAdicFriFolding<(), ()> = TwoAdicFriFolding(core::marker::PhantomData);

    let mat = rand_matrix(0xF020, 32, 8);
    let beta: EF = SmallRng::seed_from_u64(0xBEEF2).random();

    // Single arity-8 (log_arity=3) fold.
    let single = <TwoAdicFriFolding<(), ()> as FriFoldingStrategy<F, EF>>::fold_matrix(&folding, beta, 3, mat.clone());

    // Chain: arity-2(beta) on width=4, arity-2(beta^2) on width=4 of
    // result, arity-2(beta^4) on width=2 of result — wait, the algorithm
    // collapses 2:1 each step on flat data. Let's flatten + chain.
    let _h = mat.height();
    let _w = mat.width();
    let mut data = mat.values.clone();

    let mut current_beta = beta;
    for _step in 0..3 {
        let half_h = data.len() / 2;
        let intermediate = RowMajorMatrix::new(data, 2);
        data = <TwoAdicFriFolding<(), ()> as FriFoldingStrategy<F, EF>>::fold_matrix(&folding, current_beta, 1, intermediate);
        assert_eq!(data.len(), half_h);
        current_beta = current_beta.square();
    }

    assert_eq!(single, data, "arity-8(beta) disagrees with chain of three arity-2 folds");
}
