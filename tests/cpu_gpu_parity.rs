//! The GPU backend must agree with this library's own CPU backend.
//!
//! This is the test that was missing. The GPU crate's own suite only checked that the
//! diagonal of a self-distance matrix is ~0 and explicitly excused CPU/GPU disagreement as
//! "different boundary conditions" -- so several independent wrong-answer bugs sat in the
//! wavefront untouched: the series were zero-padded to a multiple of the subgroup width
//! and the DP was run over the padding; the dispatch schedule (`rows_count`) assumed
//! tile-aligned lengths; the result matrix came back transposed whenever the inputs were
//! swapped; excess invocations wrote past the end of the diagonal buffer; and MSM's cost
//! function had a typo that the reference implementation shared.
//!
//! Every case here failed before those fixes. Lengths are deliberately *not* all multiples
//! of 64: the padding bug was invisible at exactly the sizes a benchmark would pick.
//!
//! Run against a second driver too -- the barrier and thread-guard bugs only reproduced at
//! a small subgroup width:
//!   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json cargo test ...

use tsdistances::core;

/// GPU is f32, CPU is f64, so exact agreement is not expected. 1e-5 relative is orders of
/// magnitude tighter than any of the bugs above and comfortably looser than f32 rounding.
const TOL: f64 = 1e-5;

fn make_series(num: usize, len: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) * 4.0 - 2.0
    };
    (0..num)
        .map(|_| (0..len).map(|_| next()).collect())
        .collect()
}

/// Worst relative difference, or `INFINITY` if either side has a non-finite entry or the
/// two matrices disagree on shape.
fn worst(gpu: &[Vec<f64>], cpu: &[Vec<f64>]) -> f64 {
    if gpu.len() != cpu.len() || gpu[0].len() != cpu[0].len() {
        return f64::INFINITY;
    }
    let mut w: f64 = 0.0;
    for (rg, rc) in gpu.iter().zip(cpu) {
        for (x, y) in rg.iter().zip(rc) {
            if !x.is_finite() || !y.is_finite() {
                return f64::INFINITY;
            }
            w = w.max((x - y).abs() / y.abs().max(1.0));
        }
    }
    w
}

fn check(name: &str, shape: &str, gpu: &[Vec<f64>], cpu: &[Vec<f64>]) {
    let e = worst(gpu, cpu);
    assert!(
        e < TOL,
        "{name} @ {shape}: gpu backend disagrees with cpu backend by {e:e} (tol {TOL:e}); \
         gpu is {}x{}, cpu is {}x{}",
        gpu.len(),
        gpu[0].len(),
        cpu.len(),
        cpu[0].len()
    );
}

/// All seven distances, across lengths that straddle the subgroup width in both
/// directions. `63/65/96/129` are the cases the zero-padding silently corrupted; `16/32`
/// are shorter than one tile, which used to dispatch no work at all.
#[test]
fn gpu_matches_cpu_across_lengths() {
    for &len in &[16usize, 32, 63, 64, 65, 96, 128, 129, 192, 256] {
        let x = make_series(3, len, 1);
        let y = make_series(3, len, 2);
        let (p, q) = (x.as_slice(), Some(y.as_slice()));
        let s = &format!("len {len}");

        check(
            "DTW",
            s,
            &core::dtw(p, q, 1.0, false, "gpu").unwrap(),
            &core::dtw(p, q, 1.0, false, "cpu").unwrap(),
        );
        check(
            "ERP",
            s,
            &core::erp(p, q, 1.0, 0.0, false, "gpu").unwrap(),
            &core::erp(p, q, 1.0, 0.0, false, "cpu").unwrap(),
        );
        check(
            "LCSS",
            s,
            &core::lcss(p, q, 1.0, 1.0, false, "gpu").unwrap(),
            &core::lcss(p, q, 1.0, 1.0, false, "cpu").unwrap(),
        );
        check(
            "MSM",
            s,
            &core::msm(p, q, 1.0, false, "gpu").unwrap(),
            &core::msm(p, q, 1.0, false, "cpu").unwrap(),
        );
        check(
            "TWE",
            s,
            &core::twe(p, q, 1.0, 0.001, 1.0, false, "gpu").unwrap(),
            &core::twe(p, q, 1.0, 0.001, 1.0, false, "cpu").unwrap(),
        );
        check(
            "WDTW",
            s,
            &core::wdtw(p, q, 1.0, 0.05, false, "gpu").unwrap(),
            &core::wdtw(p, q, 1.0, 0.05, false, "cpu").unwrap(),
        );
        check(
            "ADTW",
            s,
            &core::adtw(p, q, 1.0, 0.1, false, "gpu").unwrap(),
            &core::adtw(p, q, 1.0, 0.1, false, "cpu").unwrap(),
        );
    }
}

/// Shapes rather than lengths: unequal series counts, unequal series lengths, and enough
/// pairs to cross the `a_chunk`/`b_chunk` tiling. The `128/64` rows are the ones that came
/// back transposed -- with the wrong dimensions when the counts also differed.
#[test]
fn gpu_matches_cpu_across_shapes() {
    // (count_a, count_b, len_a, len_b)
    let cases: &[(usize, usize, usize, usize)] = &[
        (3, 3, 64, 64),
        (3, 3, 64, 128),
        (3, 3, 128, 64),
        (3, 4, 128, 64),
        (4, 3, 64, 128),
        (1, 1, 64, 64),
        (1, 7, 64, 64),
        (7, 1, 64, 64),
        (40, 40, 64, 64),
        (3, 3, 100, 37),
    ];
    for &(na, nb, la, lb) in cases {
        let x = make_series(na, la, 1);
        let y = make_series(nb, lb, 2);
        let (p, q) = (x.as_slice(), Some(y.as_slice()));
        let s = &format!("{na}x{nb} series, len {la}/{lb}");

        check(
            "DTW",
            s,
            &core::dtw(p, q, 1.0, false, "gpu").unwrap(),
            &core::dtw(p, q, 1.0, false, "cpu").unwrap(),
        );
        check(
            "MSM",
            s,
            &core::msm(p, q, 1.0, false, "gpu").unwrap(),
            &core::msm(p, q, 1.0, false, "cpu").unwrap(),
        );
        check(
            "LCSS",
            s,
            &core::lcss(p, q, 1.0, 1.0, false, "gpu").unwrap(),
            &core::lcss(p, q, 1.0, 1.0, false, "cpu").unwrap(),
        );
    }
}

/// Series whose lengths differ by a large factor.
///
/// The CPU backend prunes cells that exceed an upper bound estimated by walking the
/// diagonal and then the last row. That estimate used the wrong predecessor for the
/// horizontal segment, so for ERP/MSM/TWE it was not an upper bound at all -- too low, it
/// pruned the optimum, the band collapsed and the function returned `inf` from a ratio of
/// about 2.5 upwards. The GPU does no pruning, so agreeing with it here is what
/// establishes that the bound is valid rather than merely finite.
#[test]
fn gpu_matches_cpu_at_large_length_ratios() {
    for &(la, lb) in &[
        (64usize, 160usize),
        (64, 192),
        (64, 256),
        (64, 512),
        (32, 96),
        (100, 300),
        (37, 200),
    ] {
        let x = make_series(2, la, 1);
        let y = make_series(2, lb, 2);
        let (p, q) = (x.as_slice(), Some(y.as_slice()));
        let s = &format!("len {la}/{lb} ({:.1}x)", lb as f64 / la as f64);

        check(
            "DTW",
            s,
            &core::dtw(p, q, 1.0, false, "gpu").unwrap(),
            &core::dtw(p, q, 1.0, false, "cpu").unwrap(),
        );
        check(
            "ERP",
            s,
            &core::erp(p, q, 1.0, 0.0, false, "gpu").unwrap(),
            &core::erp(p, q, 1.0, 0.0, false, "cpu").unwrap(),
        );
        check(
            "MSM",
            s,
            &core::msm(p, q, 1.0, false, "gpu").unwrap(),
            &core::msm(p, q, 1.0, false, "cpu").unwrap(),
        );
        check(
            "TWE",
            s,
            &core::twe(p, q, 1.0, 0.001, 1.0, false, "gpu").unwrap(),
            &core::twe(p, q, 1.0, 0.001, 1.0, false, "cpu").unwrap(),
        );
        check(
            "WDTW",
            s,
            &core::wdtw(p, q, 1.0, 0.05, false, "gpu").unwrap(),
            &core::wdtw(p, q, 1.0, 0.05, false, "cpu").unwrap(),
        );
        check(
            "ADTW",
            s,
            &core::adtw(p, q, 1.0, 0.1, false, "gpu").unwrap(),
            &core::adtw(p, q, 1.0, 0.1, false, "cpu").unwrap(),
        );
    }
}

/// The Sakoe-Chiba band must constrain the GPU exactly as it constrains the CPU.
///
/// The GPU backend used to ignore `sakoe_chiba_band` outright -- the kernels had no notion
/// of it -- so `device="gpu"` silently returned the *unbanded* distance for any band. It
/// looked correct at moderate bands only because the optimal path happened to fall inside
/// them, which made the disagreement data-dependent rather than obvious.
///
/// Note the second assertion. Parity alone would pass vacuously if *both* backends ignored
/// the parameter, so this also requires a tight band to actually change the answer.
#[test]
fn gpu_honours_sakoe_chiba_band() {
    for &len in &[64usize, 100, 128] {
        let x = make_series(2, len, 1);
        let y = make_series(2, len, 2);
        let (p, q) = (x.as_slice(), Some(y.as_slice()));

        for &band in &[1.0f64, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05] {
            let s = &format!("len {len}, band {band}");
            check(
                "DTW",
                s,
                &core::dtw(p, q, band, false, "gpu").unwrap(),
                &core::dtw(p, q, band, false, "cpu").unwrap(),
            );
            check(
                "MSM",
                s,
                &core::msm(p, q, band, false, "gpu").unwrap(),
                &core::msm(p, q, band, false, "cpu").unwrap(),
            );
            check(
                "ERP",
                s,
                &core::erp(p, q, band, 0.0, false, "gpu").unwrap(),
                &core::erp(p, q, band, 0.0, false, "cpu").unwrap(),
            );
        }

        // The band must bite, or the parity above proves nothing.
        let wide = core::dtw(p, q, 1.0, false, "gpu").unwrap()[0][0];
        let tight = core::dtw(p, q, 0.05, false, "gpu").unwrap()[0][0];
        assert!(
            tight > wide,
            "len {len}: a 0.05 band left the GPU result unchanged ({tight} vs {wide}), so the \
             constraint is being ignored -- exactly the bug this test exists for"
        );
    }
}

/// Exactly one diamond per workgroup. This is a correctness invariant, not a tuning knob.
///
/// The wavefront barrier has Workgroup *execution* scope and sits inside a loop whose trip
/// count is `diag_count`, which depends on how much of the matrix a given diamond covers.
/// A workgroup spanning several diamonds would therefore have invocations running
/// different numbers of iterations, leaving some short of a barrier the others are waiting
/// on -- undefined behaviour. That is precisely what the kernel used to do, with a
/// workgroup of `max_compute_work_group_size[0]` (1024) covering 16 diamonds; AMD
/// tolerated it and lavapipe produced garbage at every length.
///
/// This assertion exists because **nothing else catches a regression here**. Widening the
/// workgroup again passes every numerical test in this file on both drivers available
/// locally: the divergence only bites for edge diamonds whose `diag_count` is clipped, so
/// it stays latent until some other GPU or input shape exposes it. Verified by
/// re-widening the workgroup and watching the rest of the suite stay green.
#[test]
fn kernels_run_one_diamond_per_workgroup() {
    let (device, ..) = tsdistances_gpu::utils::get_device();
    let tile = tsdistances_gpu::utils::compute_tile_width(&device);
    let workgroup = tsdistances_gpu::utils::compute_workgroup_size(&device, tile);
    assert_eq!(
        workgroup as usize, tile,
        "a workgroup must be exactly one diamond ({tile} invocations), got {workgroup}; \
         a wider workgroup puts the wavefront barrier in divergent control flow, which is \
         undefined and which the numerical tests will not catch"
    );
}

/// A self-distance matrix must be symmetric with a zero diagonal. Independent of the CPU
/// backend, so this still holds where the CPU backend's own upper-bound pruning gives up
/// (`diagonal.rs` returns `inf` for MSM/TWE once the two lengths differ by 3x or more).
#[test]
fn gpu_self_distance_is_symmetric_with_zero_diagonal() {
    for &len in &[37usize, 64, 100] {
        let x = make_series(5, len, 7);
        let p = x.as_slice();
        let d = core::dtw(p, None, 1.0, false, "gpu").unwrap();
        for (i, row) in d.iter().enumerate() {
            assert!(
                row[i].abs() < 1e-4,
                "len {len}: dtw self-distance [{i}][{i}] = {}, expected ~0",
                row[i]
            );
            for (j, &value) in row.iter().enumerate() {
                let mirrored = d[j][i];
                assert!(
                    (value - mirrored).abs() < 1e-4 * value.abs().max(1.0),
                    "len {len}: dtw matrix not symmetric at [{i}][{j}]: {value} vs {mirrored}"
                );
            }
        }
    }
}
