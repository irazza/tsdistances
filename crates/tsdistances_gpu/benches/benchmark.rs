// Comprehensive benchmarking suite for tsdistances_gpu
// Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

use tsdistances_gpu::{
    cpu::{adtw, dtw, erp, lcss, msm, twe, wdtw},
    utils::get_device,
};

/// Generate synthetic test data for benchmarking
fn generate_test_data(num_series: usize, series_length: usize) -> Vec<Vec<f32>> {
    use std::f32::consts::PI;
    
    (0..num_series)
        .map(|i| {
            (0..series_length)
                .map(|j| {
                    let phase = (i as f32) * 0.1;
                    let freq = 1.0 + (i % 3) as f32 * 0.5;
                    let t = j as f32 / series_length as f32;
                    (2.0 * PI * freq * t + phase).sin() + 0.1 * ((i * j) % 100) as f32 / 100.0
                })
                .collect()
        })
        .collect()
}

const WEIGHT_MAX: f32 = 1.0;
fn dtw_weights(len: usize, g: f32) -> Vec<f32> {
    let mut weights = vec![0.0; len];
    let half_len = len as f32 / 2.0;
    let e = std::f64::consts::E as f32;
    for i in 0..len {
        weights[i] = WEIGHT_MAX / (1.0 + e.powf(-g * (i as f32 - half_len)));
    }
    weights
}

fn bench_dtw(c: &mut Criterion) {
    let mut group = c.benchmark_group("DTW");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    
    for (num_series, series_len) in [(10, 64), (50, 128), (100, 256)].iter() {
        let train_data = generate_test_data(*num_series, *series_len);
        let test_data = generate_test_data(*num_series, *series_len);
        let pairs = num_series * num_series;
        
        group.throughput(Throughput::Elements(pairs as u64));
        group.bench_with_input(
            BenchmarkId::new("pairs", format!("{}x{}_len{}", num_series, num_series, series_len)),
            &(&train_data, &test_data),
            |b, (train, test)| {
                b.iter(|| {
                    let (device, queue, sba, sda, ma) = get_device();
                    black_box(dtw(device, queue, sba, sda, ma, train, test))
                });
            },
        );
    }
    group.finish();
}

fn bench_erp(c: &mut Criterion) {
    let mut group = c.benchmark_group("ERP");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    
    let gap_penalty = 0.0f32;
    
    for (num_series, series_len) in [(10, 64), (50, 128), (100, 256)].iter() {
        let train_data = generate_test_data(*num_series, *series_len);
        let test_data = generate_test_data(*num_series, *series_len);
        let pairs = num_series * num_series;
        
        group.throughput(Throughput::Elements(pairs as u64));
        group.bench_with_input(
            BenchmarkId::new("pairs", format!("{}x{}_len{}", num_series, num_series, series_len)),
            &(&train_data, &test_data, gap_penalty),
            |b, (train, test, gap)| {
                b.iter(|| {
                    let (device, queue, sba, sda, ma) = get_device();
                    black_box(erp(device, queue, sba, sda, ma, train, test, *gap))
                });
            },
        );
    }
    group.finish();
}

fn bench_lcss(c: &mut Criterion) {
    let mut group = c.benchmark_group("LCSS");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    
    let epsilon = 1.0f32;
    
    for (num_series, series_len) in [(10, 64), (50, 128), (100, 256)].iter() {
        let train_data = generate_test_data(*num_series, *series_len);
        let test_data = generate_test_data(*num_series, *series_len);
        let pairs = num_series * num_series;
        
        group.throughput(Throughput::Elements(pairs as u64));
        group.bench_with_input(
            BenchmarkId::new("pairs", format!("{}x{}_len{}", num_series, num_series, series_len)),
            &(&train_data, &test_data, epsilon),
            |b, (train, test, eps)| {
                b.iter(|| {
                    let (device, queue, sba, sda, ma) = get_device();
                    black_box(lcss(device, queue, sba, sda, ma, train, test, *eps))
                });
            },
        );
    }
    group.finish();
}

fn bench_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("MSM");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    
    for (num_series, series_len) in [(10, 64), (50, 128), (100, 256)].iter() {
        let train_data = generate_test_data(*num_series, *series_len);
        let test_data = generate_test_data(*num_series, *series_len);
        let pairs = num_series * num_series;
        
        group.throughput(Throughput::Elements(pairs as u64));
        group.bench_with_input(
            BenchmarkId::new("pairs", format!("{}x{}_len{}", num_series, num_series, series_len)),
            &(&train_data, &test_data),
            |b, (train, test)| {
                b.iter(|| {
                    let (device, queue, sba, sda, ma) = get_device();
                    black_box(msm(device, queue, sba, sda, ma, train, test))
                });
            },
        );
    }
    group.finish();
}

fn bench_twe(c: &mut Criterion) {
    let mut group = c.benchmark_group("TWE");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    
    let stiffness = 0.001f32;
    let penalty = 1.0f32;
    
    for (num_series, series_len) in [(10, 64), (50, 128), (100, 256)].iter() {
        let train_data = generate_test_data(*num_series, *series_len);
        let test_data = generate_test_data(*num_series, *series_len);
        let pairs = num_series * num_series;
        
        group.throughput(Throughput::Elements(pairs as u64));
        group.bench_with_input(
            BenchmarkId::new("pairs", format!("{}x{}_len{}", num_series, num_series, series_len)),
            &(&train_data, &test_data, stiffness, penalty),
            |b, (train, test, stiff, pen)| {
                b.iter(|| {
                    let (device, queue, sba, sda, ma) = get_device();
                    black_box(twe(device, queue, sba, sda, ma, train, test, *stiff, *pen))
                });
            },
        );
    }
    group.finish();
}

fn bench_wdtw(c: &mut Criterion) {
    let mut group = c.benchmark_group("WDTW");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    
    for (num_series, series_len) in [(10, 64), (50, 128), (100, 256)].iter() {
        let train_data = generate_test_data(*num_series, *series_len);
        let test_data = generate_test_data(*num_series, *series_len);
        let weights = dtw_weights(*series_len, 0.05);
        let pairs = num_series * num_series;
        
        group.throughput(Throughput::Elements(pairs as u64));
        group.bench_with_input(
            BenchmarkId::new("pairs", format!("{}x{}_len{}", num_series, num_series, series_len)),
            &(&train_data, &test_data, &weights),
            |b, (train, test, w)| {
                b.iter(|| {
                    let (device, queue, sba, sda, ma) = get_device();
                    black_box(wdtw(device, queue, sba, sda, ma, train, test, w))
                });
            },
        );
    }
    group.finish();
}

fn bench_adtw(c: &mut Criterion) {
    let mut group = c.benchmark_group("ADTW");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    
    let w = 0.1f32;
    
    for (num_series, series_len) in [(10, 64), (50, 128), (100, 256)].iter() {
        let train_data = generate_test_data(*num_series, *series_len);
        let test_data = generate_test_data(*num_series, *series_len);
        let pairs = num_series * num_series;
        
        group.throughput(Throughput::Elements(pairs as u64));
        group.bench_with_input(
            BenchmarkId::new("pairs", format!("{}x{}_len{}", num_series, num_series, series_len)),
            &(&train_data, &test_data, w),
            |b, (train, test, weight)| {
                b.iter(|| {
                    let (device, queue, sba, sda, ma) = get_device();
                    black_box(adtw(device, queue, sba, sda, ma, train, test, *weight))
                });
            },
        );
    }
    group.finish();
}

/// Benchmark comparing all distance metrics at the same data size
fn bench_all_distances_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("AllDistances_50x50_128");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    
    let num_series = 50;
    let series_len = 128;
    let train_data = generate_test_data(num_series, series_len);
    let test_data = generate_test_data(num_series, series_len);
    let weights = dtw_weights(series_len, 0.05);
    
    group.bench_function("DTW", |b| {
        b.iter(|| {
            let (device, queue, sba, sda, ma) = get_device();
            black_box(dtw(device, queue, sba, sda, ma, &train_data, &test_data))
        });
    });
    
    group.bench_function("ERP", |b| {
        b.iter(|| {
            let (device, queue, sba, sda, ma) = get_device();
            black_box(erp(device, queue, sba, sda, ma, &train_data, &test_data, 0.0))
        });
    });
    
    group.bench_function("LCSS", |b| {
        b.iter(|| {
            let (device, queue, sba, sda, ma) = get_device();
            black_box(lcss(device, queue, sba, sda, ma, &train_data, &test_data, 1.0))
        });
    });
    
    group.bench_function("MSM", |b| {
        b.iter(|| {
            let (device, queue, sba, sda, ma) = get_device();
            black_box(msm(device, queue, sba, sda, ma, &train_data, &test_data))
        });
    });
    
    group.bench_function("TWE", |b| {
        b.iter(|| {
            let (device, queue, sba, sda, ma) = get_device();
            black_box(twe(device, queue, sba, sda, ma, &train_data, &test_data, 0.001, 1.0))
        });
    });
    
    group.bench_function("WDTW", |b| {
        b.iter(|| {
            let (device, queue, sba, sda, ma) = get_device();
            black_box(wdtw(device, queue, sba, sda, ma, &train_data, &test_data, &weights))
        });
    });
    
    group.bench_function("ADTW", |b| {
        b.iter(|| {
            let (device, queue, sba, sda, ma) = get_device();
            black_box(adtw(device, queue, sba, sda, ma, &train_data, &test_data, 0.1))
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_dtw,
    bench_erp,
    bench_lcss,
    bench_msm,
    bench_twe,
    bench_wdtw,
    bench_adtw,
    bench_all_distances_comparison,
);

criterion_main!(benches);
