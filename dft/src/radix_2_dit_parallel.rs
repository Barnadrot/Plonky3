// dft/src/radix_2_dit_parallel.rs
use alloc::collections::BTreeMap;
use alloc::slice;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::mem::{MaybeUninit, transmute};

use itertools::{Itertools, izip};
use p3_field::integers::QuotientMap;
use p3_field::{Field, PackedField, PackedValue, Powers, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut};
use p3_matrix::util::reverse_matrix_index_bits;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
use spin::RwLock;
use tracing::{debug_span, instrument};

use crate::TwoAdicSubgroupDft;
use crate::butterflies::{Butterfly, DitButterfly, ScaledDitButterfly, TwiddleFreeButterfly};

/// A parallel FFT algorithm which divides a butterfly network's layers into two halves.
///
/// For the first half, we apply a butterfly network with smaller blocks in earlier layers,
/// i.e. either DIT or Bowers G. Then we bit-reverse, and for the second half, we continue executing
/// the same network but in bit-revised order. This way we're always working with small blocks,
/// so within each half, we can have a certain amount of parallelism with no cross-thread
/// communication.
#[derive(Default, Clone, Debug)]
pub struct Radix2DitParallel<F> {
    /// Twiddles based on roots of unity, used in the forward DFT.
    twiddles: Arc<RwLock<BTreeMap<usize, Arc<VectorPair<F>>>>>,

    /// A map from `(log_h, shift)` to forward DFT twiddles with that coset shift baked in.
    #[allow(clippy::type_complexity)]
    coset_twiddles: Arc<RwLock<BTreeMap<(usize, F), Arc<[Vec<F>]>>>>,

    /// Twiddles based on inverse roots of unity, used in the inverse DFT.
    inverse_twiddles: Arc<RwLock<BTreeMap<usize, Arc<VectorPair<F>>>>>,
}

/// A pair of vectors, one with twiddle factors in their natural order, the other bit-reversed.
#[derive(Default, Clone, Debug)]
struct VectorPair<F> {
    twiddles: Vec<F>,
    bitrev_twiddles: Vec<F>,
}

impl<F> Radix2DitParallel<F>
where
    F: TwoAdicField + Ord,
{
    fn get_or_compute_twiddles(&self, log_h: usize) -> Arc<VectorPair<F>> {
        // Fast path: Check for the value with a cheap read lock.
        if let Some(pair) = self.twiddles.read().get(&log_h) {
            return pair.clone();
        }

        // Slow path: The value doesn't exist. Acquire a write lock.
        let mut w_lock = self.twiddles.write();

        // Double-check and compute if necessary.
        w_lock
            .entry(log_h)
            .or_insert_with(|| {
                let half_h = (1 << log_h) >> 1;
                let root = F::two_adic_generator(log_h);
                let twiddles = root.powers().collect_n(half_h);
                let mut bitrev_twiddles = twiddles.clone();
                reverse_slice_index_bits(&mut bitrev_twiddles);

                Arc::new(VectorPair {
                    twiddles,
                    bitrev_twiddles,
                })
            })
            .clone()
    }

    fn get_or_compute_coset_twiddles(&self, (log_h, shift): (usize, F)) -> Arc<[Vec<F>]> {
        let key = (log_h, shift);
        // Fast path: Try to get the value with a cheap read lock first.
        if let Some(twiddles) = self.coset_twiddles.read().get(&key) {
            return twiddles.clone();
        }
        // Slow path: The value isn't there, so we need to compute it.
        // Acquire a write lock to ensure only one thread does the computation.
        let mut w_lock = self.coset_twiddles.write();
        // Double-check: Another thread might have inserted it while we waited for the lock.
        // The `entry` API handles this check and insertion atomically.
        w_lock
            .entry(key)
            .or_insert_with(|| {
                let mid = log_h.div_ceil(2);
                let h = 1 << log_h;
                let root = F::two_adic_generator(log_h);
                (0..log_h)
                    .map(|layer| {
                        let shift_power = shift.exp_power_of_2(layer);
                        let powers = Powers {
                            base: root.exp_power_of_2(layer),
                            current: shift_power,
                        };
                        let mut twiddles = powers.collect_n(h >> (layer + 1));
                        let layer_rev = log_h - 1 - layer;
                        if layer_rev >= mid {
                            reverse_slice_index_bits(&mut twiddles);
                        }
                        twiddles
                    })
                    .collect::<Vec<_>>()
                    .into()
            })
            .clone()
    }

    fn get_or_compute_inverse_twiddles(&self, log_h: usize) -> Arc<VectorPair<F>> {
        // Fast path: First, check for the value using a cheap read lock.
        if let Some(pair) = self.inverse_twiddles.read().get(&log_h) {
            return pair.clone();
        }
        // Slow path: The value doesn't exist. Acquire a write lock.
        let mut w_lock = self.inverse_twiddles.write();
        // Double-check: Another thread might have created the entry while we waited.
        // The `entry` API handles this check and the insertion atomically.
        w_lock
            .entry(log_h)
            .or_insert_with(|| {
                // This computation only runs if the entry is truly vacant.
                let half_h = (1 << log_h) >> 1;
                let root_inv = F::two_adic_generator(log_h).inverse();
                let twiddles = root_inv.powers().collect_n(half_h);
                let mut bitrev_twiddles = twiddles.clone();
                reverse_slice_index_bits(&mut bitrev_twiddles);

                Arc::new(VectorPair {
                    twiddles,
                    bitrev_twiddles,
                })
            })
            .clone()
    }
}

impl<F: TwoAdicField + Ord> TwoAdicSubgroupDft<F> for Radix2DitParallel<F> {
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> Self::Evaluations {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        // Compute twiddle factors, or take memoized ones if already available.
        let twiddles = self.get_or_compute_twiddles(log_h);

        let mid = log_h.div_ceil(2);

        // The first half looks like a normal DIT.
        reverse_matrix_index_bits(&mut mat);
        first_half(&mut mat, mid, &twiddles.twiddles);

        // For the second half, we flip the DIT, working in bit-reversed order.
        reverse_matrix_index_bits(&mut mat);
        second_half(&mut mat, mid, &twiddles.bitrev_twiddles, None);

        mat.bit_reverse_rows()
    }

    #[instrument(skip_all, level = "debug", fields(dims = %mat.dimensions(), added_bits = added_bits))]
    fn coset_lde_batch(
        &self,
        mut mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Self::Evaluations {
        let w = mat.width;
        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let mid = log_h.div_ceil(2);

        let inverse_twiddles = self.get_or_compute_inverse_twiddles(log_h);

        // The first half looks like a normal DIT.
        reverse_matrix_index_bits(&mut mat);
        first_half(&mut mat, mid, &inverse_twiddles.twiddles);

        // For the second half, we flip the DIT, working in bit-reversed order.
        reverse_matrix_index_bits(&mut mat);
        // We'll also scale by 1/h, as per the usual inverse DFT algorithm.
        // If F isn't a PrimeField, (and is thus an extension field) it's much cheaper to
        // invert in F::PrimeSubfield.
        let h_inv_subfield = F::PrimeSubfield::from_int(h).try_inverse();
        let scale = h_inv_subfield.map(F::from_prime_subfield);
        second_half(&mut mat, mid, &inverse_twiddles.bitrev_twiddles, scale);
        // We skip the final bit-reversal, since the next FFT expects bit-reversed input.

        let lde_elems = w * (h << added_bits);
        let elems_to_add = lde_elems - w * h;
        debug_span!("reserve_exact").in_scope(|| mat.values.reserve_exact(elems_to_add));

        let g_big = F::two_adic_generator(log_h + added_bits);

        let mat_ptr = mat.values.as_mut_ptr();
        let rest_ptr = unsafe { (mat_ptr as *mut MaybeUninit<F>).add(w * h) };
        let first_slice: &mut [F] = unsafe { slice::from_raw_parts_mut(mat_ptr, w * h) };
        let rest_slice: &mut [MaybeUninit<F>] =
            unsafe { slice::from_raw_parts_mut(rest_ptr, lde_elems - w * h) };
        let mut first_coset_mat = RowMajorMatrixViewMut::new(first_slice, w);
        let mut rest_cosets_mat = rest_slice
            .chunks_exact_mut(w * h)
            .map(|slice| RowMajorMatrixViewMut::new(slice, w))
            .collect_vec();

        for coset_idx in 1..(1 << added_bits) {
            let total_shift = g_big.exp_u64(coset_idx as u64) * shift;
            let coset_idx = reverse_bits_len(coset_idx, added_bits);
            let dest = &mut rest_cosets_mat[coset_idx - 1]; // - 1 because we removed the first matrix.
            coset_dft_oop(self, &first_coset_mat.as_view(), dest, total_shift);
        }

        // Now run a forward DFT on the very first coset, this time in-place.
        coset_dft(self, &mut first_coset_mat.as_view_mut(), shift);

        // SAFETY: We wrote all values above.
        unsafe {
            mat.values.set_len(lde_elems);
        }
        BitReversalPerm::new_view(mat)
    }
}

#[instrument(level = "debug", skip_all)]
fn coset_dft<F: TwoAdicField + Ord>(
    dft: &Radix2DitParallel<F>,
    mat: &mut RowMajorMatrixViewMut<'_, F>,
    shift: F,
) {
    let log_h = log2_strict_usize(mat.height());
    let mid = log_h.div_ceil(2);

    let twiddles = dft.get_or_compute_coset_twiddles((log_h, shift));

    // The first half looks like a normal DIT.
    first_half_general(mat, mid, &twiddles);

    // For the second half, we flip the DIT, working in bit-reversed order.
    reverse_matrix_index_bits(mat);

    second_half_general(mat, mid, &twiddles);
}

/// Like `coset_dft`, except out-of-place.
#[instrument(level = "debug", skip_all)]
fn coset_dft_oop<F: TwoAdicField + Ord>(
    dft: &Radix2DitParallel<F>,
    src: &RowMajorMatrixView<'_, F>,
    dst_maybe: &mut RowMajorMatrixViewMut<'_, MaybeUninit<F>>,
    shift: F,
) {
    assert_eq!(src.dimensions(), dst_maybe.dimensions());

    let log_h = log2_strict_usize(dst_maybe.height());

    if log_h == 0 {
        // This is an edge case where first_half_general_oop doesn't work, as it expects there to be
        // at least one layer in the network, so we just copy instead.
        let src_maybe = unsafe {
            transmute::<&RowMajorMatrixView<'_, F>, &RowMajorMatrixView<'_, MaybeUninit<F>>>(src)
        };
        dst_maybe.copy_from(src_maybe);
        return;
    }

    let mid = log_h.div_ceil(2);

    let twiddles = dft.get_or_compute_coset_twiddles((log_h, shift));

    // The first half looks like a normal DIT.
    first_half_general_oop(src, dst_maybe, mid, &twiddles);

    // dst is now initialized.
    let dst = unsafe {
        transmute::<&mut RowMajorMatrixViewMut<'_, MaybeUninit<F>>, &mut RowMajorMatrixViewMut<'_, F>>(
            dst_maybe,
        )
    };

    // For the second half, we flip the DIT, working in bit-reversed order.
    reverse_matrix_index_bits(dst);

    second_half_general(dst, mid, &twiddles);
}

/// This can be used as the first half of a DIT butterfly network.
///
/// For layer 0, all twiddle factors are 1 (root^0 = 1), so we use `TwiddleFreeButterfly`
/// to avoid a Montgomery multiply by 1 across the entire matrix.
///
/// For layers 1..mid-1, the first twiddle in each block is also always 1 (twiddles[0] = 1),
/// so we special-case the first row-pair of each block to use `TwiddleFreeButterfly` as well.
#[instrument(level = "debug", skip_all)]
fn first_half<F: Field>(mat: &mut RowMajorMatrix<F>, mid: usize, twiddles: &[F]) {
    let log_h = log2_strict_usize(mat.height());

    // max block size: 2^mid
    mat.par_row_chunks_exact_mut(1 << mid)
        .for_each(|mut submat| {
            let mut backwards = false;
            for layer in 0..mid {
                if layer == 0 {
                    // For layer 0, half_block_size=1 and each block clones the twiddle
                    // iterator from the start, consuming only twiddles[0] = root^0 = 1.
                    // Use TwiddleFreeButterfly to skip the multiply entirely.
                    dit_layer_twiddle_free(&mut submat, backwards);
                } else {
                    let layer_rev = log_h - 1 - layer;
                    let layer_pow = 1 << layer_rev;
                    // For layers 1..mid-1, twiddles[0] = root^0 = 1 is always the first
                    // twiddle consumed per block. Use the optimized version that applies
                    // TwiddleFreeButterfly for the first row-pair of each block.
                    dit_layer_first_one(
                        &mut submat,
                        layer,
                        twiddles.iter().step_by(layer_pow),
                        backwards,
                    );
                }
                backwards = !backwards;
            }
        });
}

/// Like `first_half`, except supporting different twiddle factors per layer, enabling coset shifts
/// to be baked into them.
///
/// For layer 0, all blocks in the entire matrix share the same twiddle `twiddles[layer_rev][0]`
/// (the coset shift). We pre-broadcast it once and use a flat packed loop, avoiding per-block
/// iterator cloning overhead.
#[instrument(level = "debug", skip_all)]
fn first_half_general<F: Field>(
    mat: &mut RowMajorMatrixViewMut<'_, F>,
    mid: usize,
    twiddles: &[Vec<F>],
) {
    let log_h = log2_strict_usize(mat.height());
    mat.par_row_chunks_exact_mut(1 << mid)
        .for_each(|mut submat| {
            let mut backwards = false;
            for layer in 0..mid {
                let layer_rev = log_h - 1 - layer;
                if layer == 0 {
                    // Layer 0: half_block_size=1. All blocks in this submat use the same
                    // twiddle twiddles[layer_rev][0], since in the bit-reversed twiddle layout
                    // for layer_rev >= mid, each parallel chunk shares one twiddle at position 0.
                    // We pass this single twiddle directly to avoid iterator cloning overhead.
                    let twiddle = twiddles[layer_rev][0];
                    dit_layer_uniform_twiddle(&mut submat, twiddle, backwards);
                } else {
                    dit_layer(&mut submat, layer, twiddles[layer_rev].iter(), backwards);
                }
                backwards = !backwards;
            }
        });
}

/// Like `first_half_general`, except out-of-place.
///
/// Assumes there's at least one layer in the network, i.e. `src.height() > 1`.
/// Undefined behavior otherwise.
#[instrument(level = "debug", skip_all)]
fn first_half_general_oop<F: Field>(
    src: &RowMajorMatrixView<'_, F>,
    dst_maybe: &mut RowMajorMatrixViewMut<'_, MaybeUninit<F>>,
    mid: usize,
    twiddles: &[Vec<F>],
) {
    let log_h = log2_strict_usize(src.height());
    src.par_row_chunks_exact(1 << mid)
        .zip(dst_maybe.par_row_chunks_exact_mut(1 << mid))
        .for_each(|(src_submat, mut dst_submat_maybe)| {
            debug_assert_eq!(src_submat.dimensions(), dst_submat_maybe.dimensions());

            // The first layer is special, done out-of-place.
            // (Recall from the mid definition that there must be at least one layer here.)
            let layer_rev = log_h - 1;
            dit_layer_oop(
                &src_submat,
                &mut dst_submat_maybe,
                0,
                twiddles[layer_rev].iter(),
            );

            // submat is now initialized.
            let mut dst_submat = unsafe {
                transmute::<RowMajorMatrixViewMut<'_, MaybeUninit<F>>, RowMajorMatrixViewMut<'_, F>>(
                    dst_submat_maybe,
                )
            };

            // Subsequent layers.
            let mut backwards = true;
            for layer in 1..mid {
                let layer_rev = log_h - 1 - layer;
                dit_layer(
                    &mut dst_submat,
                    layer,
                    twiddles[layer_rev].iter(),
                    backwards,
                );
                backwards = !backwards;
            }
        });
}

/// This can be used as the second half of a DIT butterfly network. It works in bit-reversed order.
///
/// The optional `scale` parameter is used to scale the matrix by a constant factor. Rather than
/// doing a separate pass over memory, we fold the scaling into the first butterfly layer to
/// eliminate an extra memory pass.
///
/// When there are at least 2 layers in the second half, the last two layers (layer_rev==1 and
/// layer_rev==0) are fused into a single 4-row pass to reduce memory traffic, analogous to
/// the same optimization in `second_half_general`.
#[instrument(level = "debug", skip_all)]
#[inline(always)] // To avoid branch on scale
fn second_half<F: Field>(
    mat: &mut RowMajorMatrix<F>,
    mid: usize,
    twiddles_rev: &[F],
    scale: Option<F>,
) {
    let log_h = log2_strict_usize(mat.height());
    let num_second_half_layers = log_h - mid;

    // max block size: 2^(log_h - mid)
    mat.par_row_chunks_exact_mut(1 << (log_h - mid))
        .enumerate()
        .for_each(|(thread, mut submat)| {
            let mut backwards = false;
            let mut scale_applied = false;

            // When there are >= 2 second-half layers, we fuse the last two (layer_rev==1 and
            // layer_rev==0) into a single 4-row pass to reduce memory traffic.
            // The general layers are mid .. log_h-2 (if fusing), else mid .. log_h.
            let fuse_last_two = num_second_half_layers >= 2;
            let general_end = if fuse_last_two { log_h - 2 } else { log_h };

            for layer in mid..general_end {
                let first_block = thread << (layer - mid);
                if !scale_applied {
                    scale_applied = true;
                    dit_layer_rev_scaled(
                        &mut submat,
                        log_h,
                        layer,
                        twiddles_rev[first_block..].iter().copied(),
                        backwards,
                        scale,
                    );
                } else {
                    dit_layer_rev(
                        &mut submat,
                        log_h,
                        layer,
                        twiddles_rev[first_block..].iter().copied(),
                        backwards,
                    );
                }
                backwards = !backwards;
            }

            if fuse_last_two {
                // Fuse the last two layers (layer_rev==1 and layer_rev==0) into a single
                // 4-row pass to reduce memory traffic.
                //
                // In the flat bitrev_twiddles array for thread `t`:
                //   layer=log_h-2 (layer_rev=1): first_block = t << (log_h-2-mid)
                //   layer=log_h-1 (layer_rev=0): first_block = t << (log_h-1-mid)
                let first_block_layer1 = thread << (log_h - 2 - mid);
                let first_block_layer0 = thread << (log_h - 1 - mid);

                if !scale_applied {
                    // num_second_half_layers == 2: scale hasn't been applied yet.
                    // Fold it into the fused last-two-layer pass.
                    dit_layer_rev_last2_flat_scaled(
                        &mut submat,
                        &twiddles_rev[first_block_layer1..],
                        &twiddles_rev[first_block_layer0..],
                        scale,
                    );
                } else {
                    dit_layer_rev_last2_flat(
                        &mut submat,
                        &twiddles_rev[first_block_layer1..],
                        &twiddles_rev[first_block_layer0..],
                    );
                }
            }

            // Handle case where there are no layers in the second half (mid == log_h).
            // In that case, we still need to apply the scale.
            if !scale_applied && !fuse_last_two {
                if let Some(s) = scale {
                    submat.scale(s);
                }
            }
        });
}

/// Like `second_half`, except supporting different twiddle factors per layer, enabling coset shifts
/// to be baked into them.
///
/// Fuses the last two layers (layer_rev == 1 and layer_rev == 0) into a single 4-row pass
/// when both layers are present, reducing memory traffic for those layers.
#[instrument(level = "debug", skip_all)]
fn second_half_general<F: Field>(
    mat: &mut RowMajorMatrixViewMut<'_, F>,
    mid: usize,
    twiddles_rev: &[Vec<F>],
) {
    let log_h = log2_strict_usize(mat.height());
    // Determine how many layers the second half has.
    let num_second_half_layers = log_h - mid;

    mat.par_row_chunks_exact_mut(1 << (log_h - mid))
        .enumerate()
        .for_each(|(thread, mut submat)| {
            let mut backwards = false;
            let mut layer = mid;
            while layer < log_h {
                let layer_rev = log_h - 1 - layer;
                let first_block = thread << (layer - mid);

                // Fuse the last two layers (layer_rev == 1 and layer_rev == 0) into a single
                // 4-row pass when both are available. This reads/writes each row only once
                // instead of twice, halving memory traffic for these two layers.
                if layer_rev == 1 && num_second_half_layers >= 2 {
                    // layer_rev == 1 is this layer; layer_rev == 0 is the next layer.
                    // Fuse them: process 4 rows at a time.
                    dit_layer_rev_last2(
                        &mut submat,
                        &twiddles_rev[1][first_block..],
                        &twiddles_rev[0][first_block * 2..],
                    );
                    // We consumed two layers; skip the next layer (layer_rev == 0).
                    layer += 2;
                    // backwards would have toggled twice, so it ends up the same.
                    // No change to backwards needed.
                    continue;
                } else if layer_rev == 0 {
                    // Last layer: half_block_size=1, each block is 2 rows.
                    // Use a specialized flat loop to reduce per-block overhead.
                    dit_layer_rev_last(
                        &mut submat,
                        &twiddles_rev[0][first_block..],
                    );
                } else {
                    dit_layer_rev(
                        &mut submat,
                        log_h,
                        layer,
                        twiddles_rev[layer_rev][first_block..].iter().copied(),
                        backwards,
                    );
                }
                backwards = !backwards;
                layer += 1;
            }
        });
}

/// One layer of a DIT butterfly network where all twiddle factors are 1 (i.e., layer 0).
///
/// This is equivalent to `dit_layer` with `layer=0` and `twiddles[0]=1`, but uses
/// `TwiddleFreeButterfly` to avoid a Montgomery multiplication by 1 in the hot loop.
///
/// Correctness: For layer=0, `half_block_size=1` and each block clones the twiddle
/// iterator from position 0, consuming only `twiddles[0] = generator^0 = 1`.
/// Since multiplying by 1 is a no-op, `TwiddleFreeButterfly` gives identical results.
fn dit_layer_twiddle_free<F: Field>(submat: &mut RowMajorMatrixViewMut<'_, F>, backwards: bool) {
    // layer=0 means half_block_size=1, block_size=2.
    let width = submat.width();
    debug_assert!(submat.height() >= 2);

    let process_block = move |block: &mut [F]| {
        // Each block is exactly 2 rows: lo = block[0..width], hi = block[width..2*width]
        let (lo, hi) = block.split_at_mut(width);
        TwiddleFreeButterfly.apply_to_rows(lo, hi);
    };

    let blocks = submat.values.chunks_mut(2 * width);
    if backwards {
        for block in blocks.rev() {
            process_block(block);
        }
    } else {
        for block in blocks {
            process_block(block);
        }
    }
}

/// One layer of a DIT butterfly network where the first twiddle factor per block is always 1.
///
/// This is used in `first_half` for layers 1..mid-1 of the standard (non-coset) DFT/inverse DFT,
/// where `twiddles[0] = root^0 = 1`. The first row-pair of each block uses `TwiddleFreeButterfly`
/// to avoid one Montgomery multiplication per block, while subsequent row-pairs use `DitButterfly`.
///
/// Correctness: The twiddle iterator yields `twiddles[0], twiddles[step], twiddles[2*step], ...`
/// where `twiddles[0] = root^0 = 1`. Only used when this property holds.
fn dit_layer_first_one<'a, F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    layer: usize,
    twiddles: impl Iterator<Item = &'a F> + Clone,
    backwards: bool,
) {
    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;
    let width = submat.width();
    debug_assert!(submat.height() >= block_size);
    debug_assert!(
        half_block_size >= 2,
        "layer must be >= 1 for dit_layer_first_one"
    );

    let process_block = move |block: &mut [F]| {
        let (lows, highs) = block.split_at_mut(half_block_size * width);
        let mut tw_iter = twiddles.clone();
        // First row-pair: twiddle is always 1, use TwiddleFreeButterfly to skip the multiply.
        let _ = tw_iter.next(); // consume twiddles[0] = 1
        let (lo0, lo_rest) = lows.split_at_mut(width);
        let (hi0, hi_rest) = highs.split_at_mut(width);
        TwiddleFreeButterfly.apply_to_rows(lo0, hi0);
        // Remaining row-pairs use DitButterfly with their respective twiddle factors.
        for (lo, hi, twiddle) in izip!(
            lo_rest.chunks_mut(width),
            hi_rest.chunks_mut(width),
            tw_iter
        ) {
            DitButterfly(*twiddle).apply_to_rows(lo, hi);
        }
    };

    let blocks = submat.values.chunks_mut(block_size * width);
    if backwards {
        for block in blocks.rev() {
            process_block(block);
        }
    } else {
        for block in blocks {
            process_block(block);
        }
    }
}

/// One layer of a DIT butterfly network where all blocks share a single uniform twiddle factor.
///
/// This is used in `first_half_general` for layer 0. At layer 0, the twiddle layout ensures
/// that all `h/2` blocks (each a row-pair) use the same twiddle value `twiddles[layer_rev][0]`.
/// By pre-broadcasting this single twiddle into a packed field once and then processing all
/// row-pairs in a flat loop, we eliminate the per-block iterator-clone overhead of `dit_layer`.
///
/// Correctness: Only valid when all blocks at this layer share the same twiddle (layer 0).
fn dit_layer_uniform_twiddle<F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    twiddle: F,
    backwards: bool,
) {
    let width = submat.width();
    // Pre-broadcast the scalar twiddle into a packed field once for all blocks.
    let twiddle_packed = F::Packing::from(twiddle);

    let process_pair = |pair: &mut [F]| {
        let (lo, hi) = pair.split_at_mut(width);
        let (lo_packed, lo_suffix) = F::Packing::pack_slice_with_suffix_mut(lo);
        let (hi_packed, hi_suffix) = F::Packing::pack_slice_with_suffix_mut(hi);
        for (lp, hp) in lo_packed.iter_mut().zip(hi_packed.iter_mut()) {
            let x2t = *hp * twiddle_packed;
            let new_lo = *lp + x2t;
            *hp = *lp - x2t;
            *lp = new_lo;
        }
        for (ls, hs) in lo_suffix.iter_mut().zip(hi_suffix.iter_mut()) {
            let x2t = *hs * twiddle;
            let new_lo = *ls + x2t;
            *hs = *ls - x2t;
            *ls = new_lo;
        }
    };

    // Each block is 2 rows = 2*width elements.
    let blocks = submat.values.chunks_mut(2 * width);
    if backwards {
        for pair in blocks.rev() {
            process_pair(pair);
        }
    } else {
        for pair in blocks {
            process_pair(pair);
        }
    }
}

/// One layer of a DIT butterfly network.
fn dit_layer<'a, F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    layer: usize,
    twiddles: impl Iterator<Item = &'a F> + Clone,
    backwards: bool,
) {
    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;
    let width = submat.width();
    debug_assert!(submat.height() >= block_size);

    let process_block = move |block: &mut [F]| {
        let (lows, highs) = block.split_at_mut(half_block_size * width);
        for (lo, hi, twiddle) in izip!(
            lows.chunks_mut(width),
            highs.chunks_mut(width),
            twiddles.clone()
        ) {
            DitButterfly(*twiddle).apply_to_rows(lo, hi);
        }
    };

    let blocks = submat.values.chunks_mut(block_size * width);
    if backwards {
        for block in blocks.rev() {
            process_block(block);
        }
    } else {
        for block in blocks {
            process_block(block);
        }
    }
}

/// One layer of a DIT butterfly network, out-of-place.
fn dit_layer_oop<'a, F: Field>(
    src: &RowMajorMatrixView<'_, F>,
    dst: &mut RowMajorMatrixViewMut<'_, MaybeUninit<F>>,
    layer: usize,
    twiddles: impl Iterator<Item = &'a F> + Clone,
) {
    debug_assert_eq!(src.dimensions(), dst.dimensions());
    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;
    let width = dst.width();
    debug_assert!(dst.height() >= block_size);

    let process_blocks = move |src_block: &[F], dst_block: &mut [MaybeUninit<F>]| {
        let (src_lows, src_highs) = src_block.split_at(half_block_size * width);
        let (dst_lows, dst_highs) = dst_block.split_at_mut(half_block_size * width);

        for (src_lo, dst_lo, src_hi, dst_hi, twiddle) in izip!(
            src_lows.chunks(width),
            dst_lows.chunks_mut(width),
            src_highs.chunks(width),
            dst_highs.chunks_mut(width),
            twiddles.clone()
        ) {
            DitButterfly(*twiddle).apply_to_rows_oop(src_lo, dst_lo, src_hi, dst_hi);
        }
    };

    let src_chunks = src.values.chunks(block_size * width);
    let dst_chunks = dst.values.chunks_mut(block_size * width);

    for (src_block, dst_block) in src_chunks.zip(dst_chunks) {
        process_blocks(src_block, dst_block);
    }
}

/// Like `dit_layer_rev`, except with an optional scale factor folded into the butterfly.
///
/// This avoids an extra memory pass when scaling is required (e.g., 1/N in inverse DFT).
/// When `scale` is `None`, this is identical to `dit_layer_rev`.
///
/// When `scale` is `Some(s)`, uses `ScaledDitButterfly::new(twiddle, s)` which precomputes
/// `twiddle * scale` once per block, reducing multiplications in the hot loop from 3 to 2.
fn dit_layer_rev_scaled<F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    log_h: usize,
    layer: usize,
    twiddles_rev: impl DoubleEndedIterator<Item = F> + ExactSizeIterator,
    backwards: bool,
    scale: Option<F>,
) {
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer_rev;
    let block_size = half_block_size * 2;
    let width = submat.width();
    debug_assert!(submat.height() >= block_size);

    match scale {
        None => {
            // No scaling: same as regular dit_layer_rev
            let blocks_and_twiddles = submat
                .values
                .chunks_mut(block_size * width)
                .zip(twiddles_rev);
            if backwards {
                for (block, twiddle) in blocks_and_twiddles.rev() {
                    let (lo, hi) = block.split_at_mut(half_block_size * width);
                    DitButterfly(twiddle).apply_to_rows(lo, hi);
                }
            } else {
                for (block, twiddle) in blocks_and_twiddles {
                    let (lo, hi) = block.split_at_mut(half_block_size * width);
                    DitButterfly(twiddle).apply_to_rows(lo, hi);
                }
            }
        }
        Some(s) => {
            // Fold scaling into the butterfly to avoid a separate memory pass.
            // ScaledDitButterfly::new precomputes twiddle * scale once per block,
            // so the hot loop only needs 2 multiplications instead of 3.
            let blocks_and_twiddles = submat
                .values
                .chunks_mut(block_size * width)
                .zip(twiddles_rev);
            if backwards {
                for (block, twiddle) in blocks_and_twiddles.rev() {
                    let (lo, hi) = block.split_at_mut(half_block_size * width);
                    ScaledDitButterfly::new(twiddle, s).apply_to_rows(lo, hi);
                }
            } else {
                for (block, twiddle) in blocks_and_twiddles {
                    let (lo, hi) = block.split_at_mut(half_block_size * width);
                    ScaledDitButterfly::new(twiddle, s).apply_to_rows(lo, hi);
                }
            }
        }
    }
}

/// Like `dit_layer`, except the matrix and twiddles are encoded in bit-reversed order.
/// This can also be viewed as a layer of the Bowers G^T network.
fn dit_layer_rev<F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    log_h: usize,
    layer: usize,
    twiddles_rev: impl DoubleEndedIterator<Item = F> + ExactSizeIterator,
    backwards: bool,
) {
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer_rev;
    let block_size = half_block_size * 2;
    let width = submat.width();
    debug_assert!(submat.height() >= block_size);

    let blocks_and_twiddles = submat
        .values
        .chunks_mut(block_size * width)
        .zip(twiddles_rev);
    if backwards {
        for (block, twiddle) in blocks_and_twiddles.rev() {
            let (lo, hi) = block.split_at_mut(half_block_size * width);
            DitButterfly(twiddle).apply_to_rows(lo, hi);
        }
    } else {
        for (block, twiddle) in blocks_and_twiddles {
            let (lo, hi) = block.split_at_mut(half_block_size * width);
            DitButterfly(twiddle).apply_to_rows(lo, hi);
        }
    }
}

/// Specialized last layer of the second half: `layer_rev == 0`, so `half_block_size == 1`.
///
/// Each block is exactly 2 rows. Instead of iterating block-by-block with `chunks_mut(2*width)`
/// and calling `DitButterfly::apply_to_rows` for each block (which broadcasts the twiddle into
/// a packed field inside the call), we inline the broadcast and packed computation directly.
///
/// Since all blocks are independent, the processing order does not affect correctness, so we
/// always iterate forward regardless of the `backwards` flag.
///
/// The twiddle slice provides one twiddle factor per block (row-pair), starting at `first_block`.
fn dit_layer_rev_last<F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    twiddles: &[F],
) {
    let width = submat.width();
    // Each block is 2 rows = 2*width elements.
    for (pair, &twiddle) in submat.values.chunks_mut(2 * width).zip(twiddles.iter()) {
        let (lo, hi) = pair.split_at_mut(width);
        // Pre-broadcast the scalar twiddle into a packed field once per block.
        let twiddle_packed = F::Packing::from(twiddle);
        let (lo_packed, lo_suffix) = F::Packing::pack_slice_with_suffix_mut(lo);
        let (hi_packed, hi_suffix) = F::Packing::pack_slice_with_suffix_mut(hi);
        for (lp, hp) in lo_packed.iter_mut().zip(hi_packed.iter_mut()) {
            let x2t = *hp * twiddle_packed;
            let new_lo = *lp + x2t;
            *hp = *lp - x2t;
            *lp = new_lo;
        }
        for (ls, hs) in lo_suffix.iter_mut().zip(hi_suffix.iter_mut()) {
            let x2t = *hs * twiddle;
            let new_lo = *ls + x2t;
            *hs = *ls - x2t;
            *ls = new_lo;
        }
    }
}

/// Fused last two layers of the second half: processes layer_rev==1 and layer_rev==0 together.
///
/// Used in `second_half_general`. Each "mega-block" is 4 rows: [r0, r1, r2, r3].
/// Layer rev==1 butterfly (half_block_size=2):
///   - block 0: twiddle1_0 applied to (r0, r2)
///   - block 1: twiddle1_1 applied to (r1, r3)
/// Layer rev==0 butterfly (half_block_size=1):
///   - block 0: twiddle0_0 applied to (r0, r1)
///   - block 1: twiddle0_1 applied to (r2, r3)
///
/// By processing both layers in a single pass over memory, each row is loaded and stored
/// only once instead of twice, halving memory bandwidth for these two layers.
///
/// `twiddles1` provides twiddles for layer_rev==1 (2 per mega-block, but each block of 4 rows
/// has 1 twiddle for layer_rev==1 per half-block, giving 2 twiddles per 4-row group... wait,
/// actually: for layer_rev==1, block_size=4, half_block_size=2, so there's 1 twiddle per
/// 4-row block). For layer_rev==0, half_block_size=1, block_size=2, so 2 twiddles per 4-row group.
///
/// Concretely for a 4-row mega-block with rows [r0, r1, r2, r3]:
///   Layer rev==1 (block_size=4, half_block_size=2, 1 twiddle per 4-row):
///     Apply twiddle `t1` to the pair (lo=[r0,r1], hi=[r2,r3]):
///       r0' = r0 + r2*t1,  r2' = r0 - r2*t1
///       r1' = r1 + r3*t1,  r1' = r1 - r3*t1  (same t1 for all row-pairs in the block)
///   Layer rev==0 (block_size=2, half_block_size=1, 2 twiddles per 4-row group):
///     Apply twiddle `t0_a` to pair (r0', r1'):
///       out0 = r0' + r1'*t0_a,  out1 = r0' - r1'*t0_a
///     Apply twiddle `t0_b` to pair (r2', r3'):
///       out2 = r2' + r3'*t0_b,  out3 = r2' - r3'*t0_b
fn dit_layer_rev_last2<F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    twiddles1: &[F],  // layer_rev==1 twiddles: 1 per 4-row block
    twiddles0: &[F],  // layer_rev==0 twiddles: 2 per 4-row block (interleaved pairs)
) {
    let width = submat.width();
    // Each mega-block is 4 rows = 4*width elements.
    // twiddles1: 1 entry per mega-block
    // twiddles0: 2 entries per mega-block (for the two 2-row sub-blocks after the first layer)
    for (quad, (&t1, t0_pair)) in submat
        .values
        .chunks_mut(4 * width)
        .zip(twiddles1.iter().zip(twiddles0.chunks(2)))
    {
        let t0_a = t0_pair[0];
        let t0_b = t0_pair[1];

        // Split the 4-row block into 4 individual rows.
        let (r0, rest) = quad.split_at_mut(width);
        let (r1, rest) = rest.split_at_mut(width);
        let (r2, r3) = rest.split_at_mut(width);

        // Pre-broadcast twiddles into packed fields.
        let t1_packed = F::Packing::from(t1);
        let t0a_packed = F::Packing::from(t0_a);
        let t0b_packed = F::Packing::from(t0_b);

        // Process packed chunks.
        let (r0_packed, r0_suffix) = F::Packing::pack_slice_with_suffix_mut(r0);
        let (r1_packed, r1_suffix) = F::Packing::pack_slice_with_suffix_mut(r1);
        let (r2_packed, r2_suffix) = F::Packing::pack_slice_with_suffix_mut(r2);
        let (r3_packed, r3_suffix) = F::Packing::pack_slice_with_suffix_mut(r3);

        for (p0, p1, p2, p3) in izip!(
            r0_packed.iter_mut(),
            r1_packed.iter_mut(),
            r2_packed.iter_mut(),
            r3_packed.iter_mut()
        ) {
            // Layer rev==1: apply t1 to (r0,r2) and to (r1,r3) simultaneously.
            let r2t = *p2 * t1_packed;
            let r3t = *p3 * t1_packed;
            let new_r0 = *p0 + r2t;
            let new_r1 = *p1 + r3t;
            let new_r2 = *p0 - r2t;
            let new_r3 = *p1 - r3t;

            // Layer rev==0: apply t0_a to (new_r0, new_r1), t0_b to (new_r2, new_r3).
            let r1t = new_r1 * t0a_packed;
            let r3t = new_r3 * t0b_packed;
            *p0 = new_r0 + r1t;
            *p1 = new_r0 - r1t;
            *p2 = new_r2 + r3t;
            *p3 = new_r2 - r3t;
        }

        // Scalar suffix.
        for (s0, s1, s2, s3) in izip!(
            r0_suffix.iter_mut(),
            r1_suffix.iter_mut(),
            r2_suffix.iter_mut(),
            r3_suffix.iter_mut()
        ) {
            // Layer rev==1.
            let r2t = *s2 * t1;
            let r3t = *s3 * t1;
            let new_r0 = *s0 + r2t;
            let new_r1 = *s1 + r3t;
            let new_r2 = *s0 - r2t;
            let new_r3 = *s1 - r3t;

            // Layer rev==0.
            let r1t = new_r1 * t0_a;
            let r3t = new_r3 * t0_b;
            *s0 = new_r0 + r1t;
            *s1 = new_r0 - r1t;
            *s2 = new_r2 + r3t;
            *s3 = new_r2 - r3t;
        }
    }
}

/// Fused last two layers for `second_half` (flat bitrev_twiddles layout), without scaling.
///
/// In `second_half`, twiddles are stored in a single flat bit-reversed array. For thread `t`:
/// - `twiddles1 = &twiddles_rev[t << (log_h-2-mid) ..]`: layer_rev==1, 1 twiddle per 4-row block
/// - `twiddles0 = &twiddles_rev[t << (log_h-1-mid) ..]`: layer_rev==0, 2 twiddles per 4-row block
///
/// The butterfly computation is identical to `dit_layer_rev_last2`.
fn dit_layer_rev_last2_flat<F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    twiddles1: &[F],
    twiddles0: &[F],
) {
    let width = submat.width();
    for (quad, (&t1, t0_pair)) in submat
        .values
        .chunks_mut(4 * width)
        .zip(twiddles1.iter().zip(twiddles0.chunks(2)))
    {
        let t0_a = t0_pair[0];
        let t0_b = t0_pair[1];

        let (r0, rest) = quad.split_at_mut(width);
        let (r1, rest) = rest.split_at_mut(width);
        let (r2, r3) = rest.split_at_mut(width);

        let t1_packed = F::Packing::from(t1);
        let t0a_packed = F::Packing::from(t0_a);
        let t0b_packed = F::Packing::from(t0_b);

        let (r0_packed, r0_suffix) = F::Packing::pack_slice_with_suffix_mut(r0);
        let (r1_packed, r1_suffix) = F::Packing::pack_slice_with_suffix_mut(r1);
        let (r2_packed, r2_suffix) = F::Packing::pack_slice_with_suffix_mut(r2);
        let (r3_packed, r3_suffix) = F::Packing::pack_slice_with_suffix_mut(r3);

        for (p0, p1, p2, p3) in izip!(
            r0_packed.iter_mut(),
            r1_packed.iter_mut(),
            r2_packed.iter_mut(),
            r3_packed.iter_mut()
        ) {
            // Layer rev==1.
            let r2t = *p2 * t1_packed;
            let r3t = *p3 * t1_packed;
            let new_r0 = *p0 + r2t;
            let new_r1 = *p1 + r3t;
            let new_r2 = *p0 - r2t;
            let new_r3 = *p1 - r3t;

            // Layer rev==0.
            let r1t = new_r1 * t0a_packed;
            let r3t = new_r3 * t0b_packed;
            *p0 = new_r0 + r1t;
            *p1 = new_r0 - r1t;
            *p2 = new_r2 + r3t;
            *p3 = new_r2 - r3t;
        }

        for (s0, s1, s2, s3) in izip!(
            r0_suffix.iter_mut(),
            r1_suffix.iter_mut(),
            r2_suffix.iter_mut(),
            r3_suffix.iter_mut()
        ) {
            let r2t = *s2 * t1;
            let r3t = *s3 * t1;
            let new_r0 = *s0 + r2t;
            let new_r1 = *s1 + r3t;
            let new_r2 = *s0 - r2t;
            let new_r3 = *s1 - r3t;

            let r1t = new_r1 * t0_a;
            let r3t = new_r3 * t0_b;
            *s0 = new_r0 + r1t;
            *s1 = new_r0 - r1t;
            *s2 = new_r2 + r3t;
            *s3 = new_r2 - r3t;
        }
    }
}

/// Fused last two layers for `second_half` (flat bitrev_twiddles layout) with optional scaling.
///
/// Used when `scale` hasn't been applied yet at the fused-last-two-layers point (i.e.,
/// `num_second_half_layers == 2`). The scale is folded into layer_rev==1 so that the
/// final outputs equal `(butterfly result) * scale`.
///
/// When `scale` is `None`, delegates to `dit_layer_rev_last2_flat`.
/// When `scale` is `Some(s)`:
///   - Layer rev==1 butterfly output is scaled by `s`:
///       new_r0 = (r0 + r2*t1) * s = r0*s + r2*(t1*s)
///       new_r1 = (r1 + r3*t1) * s
///       new_r2 = (r0 - r2*t1) * s
///       new_r3 = (r1 - r3*t1) * s
///   - Layer rev==0 butterfly uses unscaled t0a/t0b since inputs are already scaled:
///       out0 = new_r0 + new_r1 * t0a
///       out1 = new_r0 - new_r1 * t0a
///       etc.
fn dit_layer_rev_last2_flat_scaled<F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    twiddles1: &[F],
    twiddles0: &[F],
    scale: Option<F>,
) {
    match scale {
        None => {
            dit_layer_rev_last2_flat(submat, twiddles1, twiddles0);
        }
        Some(s) => {
            let width = submat.width();
            let s_packed = F::Packing::from(s);

            for (quad, (&t1, t0_pair)) in submat
                .values
                .chunks_mut(4 * width)
                .zip(twiddles1.iter().zip(twiddles0.chunks(2)))
            {
                let t0_a = t0_pair[0];
                let t0_b = t0_pair[1];

                let (r0, rest) = quad.split_at_mut(width);
                let (r1, rest) = rest.split_at_mut(width);
                let (r2, r3) = rest.split_at_mut(width);

                // Precompute t1*s to fold scaling into layer_rev==1.
                let t1s = t1 * s;
                let t1s_packed = F::Packing::from(t1s);
                let t0a_packed = F::Packing::from(t0_a);
                let t0b_packed = F::Packing::from(t0_b);

                let (r0_packed, r0_suffix) = F::Packing::pack_slice_with_suffix_mut(r0);
                let (r1_packed, r1_suffix) = F::Packing::pack_slice_with_suffix_mut(r1);
                let (r2_packed, r2_suffix) = F::Packing::pack_slice_with_suffix_mut(r2);
                let (r3_packed, r3_suffix) = F::Packing::pack_slice_with_suffix_mut(r3);

                for (p0, p1, p2, p3) in izip!(
                    r0_packed.iter_mut(),
                    r1_packed.iter_mut(),
                    r2_packed.iter_mut(),
                    r3_packed.iter_mut()
                ) {
                    // Layer rev==1 with scale: new_ri = (pi +/- pj*t1) * s
                    // = pi*s +/- pj*(t1*s)
                    let p0s = *p0 * s_packed;
                    let p1s = *p1 * s_packed;
                    let r2ts = *p2 * t1s_packed;
                    let r3ts = *p3 * t1s_packed;
                    let new_r0 = p0s + r2ts;
                    let new_r1 = p1s + r3ts;
                    let new_r2 = p0s - r2ts;
                    let new_r3 = p1s - r3ts;

                    // Layer rev==0: inputs already carry scale, use plain twiddles.
                    let r1t = new_r1 * t0a_packed;
                    let r3t = new_r3 * t0b_packed;
                    *p0 = new_r0 + r1t;
                    *p1 = new_r0 - r1t;
                    *p2 = new_r2 + r3t;
                    *p3 = new_r2 - r3t;
                }

                for (s0, s1, s2, s3) in izip!(
                    r0_suffix.iter_mut(),
                    r1_suffix.iter_mut(),
                    r2_suffix.iter_mut(),
                    r3_suffix.iter_mut()
                ) {
                    let s0s = *s0 * s;
                    let s1s = *s1 * s;
                    let r2ts = *s2 * t1s;
                    let r3ts = *s3 * t1s;
                    let new_r0 = s0s + r2ts;
                    let new_r1 = s1s + r3ts;
                    let new_r2 = s0s - r2ts;
                    let new_r3 = s1s - r3ts;

                    let r1t = new_r1 * t0_a;
                    let r3t = new_r3 * t0_b;
                    *s0 = new_r0 + r1t;
                    *s1 = new_r0 - r1t;
                    *s2 = new_r2 + r3t;
                    *s3 = new_r2 - r3t;
                }
            }
        }
    }
}
