#![feature(test)]

extern crate test;

use test::{Bencher, black_box};
use tsdistances::{core, diagonal};

fn build_series(count: usize, len: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let mut out = Vec::with_capacity(count);

    for _ in 0..count {
        let mut row = Vec::with_capacity(len);
        for _ in 0..len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let value = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            row.push(value * 2.0 - 1.0);
        }
        out.push(row);
    }

    out
}

#[bench]
fn bench_diagonal_dtw_kernel_len512(b: &mut Bencher) {
    let a = build_series(1, 512, 1).pop().unwrap();
    let c = build_series(1, 512, 2).pop().unwrap();

    b.iter(|| {
        black_box(diagonal::diagonal_distance_dtw(
            black_box(&a),
            black_box(&c),
            black_box(1.0),
        ))
    });
}

#[bench]
fn bench_pairwise_euclidean_32x32_len256(b: &mut Bencher) {
    let x1 = build_series(32, 256, 3);
    let x2 = build_series(32, 256, 4);

    b.iter(|| {
        black_box(core::euclidean(
            black_box(x1.as_slice()),
            black_box(Some(x2.as_slice())),
            black_box(false),
        ))
        .unwrap()
    });
}

#[bench]
fn bench_pairwise_dtw_16x16_len256(b: &mut Bencher) {
    let x1 = build_series(16, 256, 5);
    let x2 = build_series(16, 256, 6);

    b.iter(|| {
        black_box(core::dtw(
            black_box(x1.as_slice()),
            black_box(Some(x2.as_slice())),
            black_box(1.0),
            black_box(false),
            black_box("cpu"),
        ))
        .unwrap()
    });
}

#[bench]
fn bench_pairwise_adtw_16x16_len256(b: &mut Bencher) {
    let x1 = build_series(16, 256, 7);
    let x2 = build_series(16, 256, 8);

    b.iter(|| {
        black_box(core::adtw(
            black_box(x1.as_slice()),
            black_box(Some(x2.as_slice())),
            black_box(1.0),
            black_box(0.05),
            black_box(false),
            black_box("cpu"),
        ))
        .unwrap()
    });
}
