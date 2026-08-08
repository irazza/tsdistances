// Correctness tests for GPU distance computations
// These tests verify basic properties rather than exact CPU/GPU matching
// since the GPU uses diamond partitioning with different boundary conditions

mod cpu_reference;

use tsdistances_gpu::{
    cpu::{adtw, dtw, erp, lcss, msm, twe, wdtw},
    utils::get_device,
};

const TOLERANCE: f32 = 1e-4;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
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

/// Generate simple test data
fn generate_simple_test_data(num_series: usize, series_length: usize) -> Vec<Vec<f32>> {
    (0..num_series)
        .map(|i| {
            (0..series_length)
                .map(|j| ((i * 7 + j * 3) % 100) as f32 / 10.0)
                .collect()
        })
        .collect()
}

/// Test that identical sequences have distance 0
#[test]
fn test_dtw_identity() {
    println!("\n=== DTW Identity Test ===");
    
    let data = generate_simple_test_data(5, 64);
    
    let (device, queue, sba, sda, ma) = get_device();
    let result = dtw(device, queue, sba, sda, ma, &data, &data);
    
    // Diagonal should be 0 (distance to self)
    for i in 0..data.len() {
        assert!(
            approx_eq(result[i][i], 0.0, TOLERANCE),
            "DTW identity failed at [{}][{}]: expected ~0, got {}",
            i, i, result[i][i]
        );
    }
    println!("✓ DTW identity test passed!");
}

#[test]
fn test_dtw_symmetry() {
    println!("\n=== DTW Symmetry Test ===");
    
    let a_data = generate_simple_test_data(4, 64);
    let b_data = generate_simple_test_data(4, 64);
    
    let (device, queue, sba, sda, ma) = get_device();
    let ab_result = dtw(device.clone(), queue.clone(), sba.clone(), sda.clone(), ma.clone(), &a_data, &b_data);
    
    let (device, queue, sba, sda, ma) = get_device();
    let ba_result = dtw(device, queue, sba, sda, ma, &b_data, &a_data);
    
    for i in 0..a_data.len() {
        for j in 0..b_data.len() {
            assert!(
                approx_eq(ab_result[i][j], ba_result[j][i], TOLERANCE),
                "DTW symmetry failed: d(a[{}],b[{}])={} != d(b[{}],a[{}])={}",
                i, j, ab_result[i][j], j, i, ba_result[j][i]
            );
        }
    }
    println!("✓ DTW symmetry test passed!");
}

#[test]
fn test_dtw_non_negative() {
    println!("\n=== DTW Non-negative Test ===");
    
    let train_data = generate_simple_test_data(10, 64);
    let test_data = generate_simple_test_data(8, 64);
    
    let (device, queue, sba, sda, ma) = get_device();
    let result = dtw(device, queue, sba, sda, ma, &train_data, &test_data);
    
    for (i, row) in result.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert!(
                val >= 0.0,
                "DTW should be non-negative at [{}, {}]: got {}",
                i, j, val
            );
        }
    }
    println!("✓ DTW non-negative test passed!");
}

#[test]
fn test_erp_identity() {
    println!("\n=== ERP Identity Test ===");
    
    let data = generate_simple_test_data(5, 64);
    
    let (device, queue, sba, sda, ma) = get_device();
    let result = erp(device, queue, sba, sda, ma, &data, &data, 0.0);
    
    for i in 0..data.len() {
        assert!(
            approx_eq(result[i][i], 0.0, TOLERANCE),
            "ERP identity failed at [{}][{}]: expected ~0, got {}",
            i, i, result[i][i]
        );
    }
    println!("✓ ERP identity test passed!");
}

#[test]
fn test_lcss_identity() {
    println!("\n=== LCSS Identity Test ===");
    
    let data = generate_simple_test_data(5, 64);
    
    let (device, queue, sba, sda, ma) = get_device();
    let result = lcss(device, queue, sba, sda, ma, &data, &data, 1.0);
    
    // For identical sequences with epsilon=1.0, distance should be 0
    for i in 0..data.len() {
        assert!(
            approx_eq(result[i][i], 0.0, TOLERANCE),
            "LCSS identity failed at [{}][{}]: expected ~0, got {}",
            i, i, result[i][i]
        );
    }
    println!("✓ LCSS identity test passed!");
}

#[test]
fn test_msm_identity() {
    println!("\n=== MSM Identity Test ===");
    
    let data = generate_simple_test_data(5, 64);
    
    let (device, queue, sba, sda, ma) = get_device();
    let result = msm(device, queue, sba, sda, ma, &data, &data);
    
    for i in 0..data.len() {
        assert!(
            approx_eq(result[i][i], 0.0, TOLERANCE),
            "MSM identity failed at [{}][{}]: expected ~0, got {}",
            i, i, result[i][i]
        );
    }
    println!("✓ MSM identity test passed!");
}

#[test]
fn test_twe_identity() {
    println!("\n=== TWE Identity Test ===");
    
    let data = generate_simple_test_data(5, 64);
    
    let (device, queue, sba, sda, ma) = get_device();
    let result = twe(device, queue, sba, sda, ma, &data, &data, 0.001, 1.0);
    
    for i in 0..data.len() {
        assert!(
            approx_eq(result[i][i], 0.0, TOLERANCE),
            "TWE identity failed at [{}][{}]: expected ~0, got {}",
            i, i, result[i][i]
        );
    }
    println!("✓ TWE identity test passed!");
}

#[test]
fn test_wdtw_identity() {
    println!("\n=== WDTW Identity Test ===");
    
    let data = generate_simple_test_data(5, 64);
    let weights = dtw_weights(64, 0.05);
    
    let (device, queue, sba, sda, ma) = get_device();
    let result = wdtw(device, queue, sba, sda, ma, &data, &data, &weights);
    
    for i in 0..data.len() {
        assert!(
            approx_eq(result[i][i], 0.0, TOLERANCE),
            "WDTW identity failed at [{}][{}]: expected ~0, got {}",
            i, i, result[i][i]
        );
    }
    println!("✓ WDTW identity test passed!");
}

#[test]
fn test_adtw_identity() {
    println!("\n=== ADTW Identity Test ===");
    
    let data = generate_simple_test_data(5, 64);
    
    let (device, queue, sba, sda, ma) = get_device();
    let result = adtw(device, queue, sba, sda, ma, &data, &data, 0.1);
    
    for i in 0..data.len() {
        assert!(
            approx_eq(result[i][i], 0.0, TOLERANCE),
            "ADTW identity failed at [{}][{}]: expected ~0, got {}",
            i, i, result[i][i]
        );
    }
    println!("✓ ADTW identity test passed!");
}

/// Test consistency: multiple runs produce same results
#[test]
fn test_dtw_consistency() {
    println!("\n=== DTW Consistency Test ===");
    
    let train_data = generate_simple_test_data(10, 64);
    let test_data = generate_simple_test_data(10, 64);  // Same size to avoid swap issues
    
    let (device, queue, sba, sda, ma) = get_device();
    let result1 = dtw(device, queue, sba, sda, ma, &train_data, &test_data);
    
    let (device, queue, sba, sda, ma) = get_device();
    let result2 = dtw(device, queue, sba, sda, ma, &train_data, &test_data);
    
    for i in 0..result1.len() {
        for j in 0..result1[i].len() {
            assert!(
                approx_eq(result1[i][j], result2[i][j], TOLERANCE),
                "DTW consistency failed at [{}, {}]: {} vs {}",
                i, j, result1[i][j], result2[i][j]
            );
        }
    }
    println!("✓ DTW consistency test passed!");
}

/// Test various input sizes
#[test]
fn test_various_sizes() {
    println!("\n=== Various Sizes Test ===");
    
    let sizes = [(2, 32), (5, 64), (10, 128)];
    
    for (num_series, series_len) in sizes.iter() {
        println!("Testing {}x{} series of length {}...", num_series, num_series, series_len);
        
        let data = generate_simple_test_data(*num_series, *series_len);
        
        let (device, queue, sba, sda, ma) = get_device();
        let result = dtw(device, queue, sba, sda, ma, &data, &data);
        
        // Check diagonal is zero
        for i in 0..data.len() {
            assert!(
                approx_eq(result[i][i], 0.0, TOLERANCE),
                "Size {}x{}: diagonal not zero at {}: {}",
                num_series, series_len, i, result[i][i]
            );
        }
        println!("  ✓ {}x{} passed", num_series, series_len);
    }
    println!("✓ All size tests passed!");
}

/// Test that results are finite (no NaN/Inf)
#[test]
fn test_results_finite() {
    println!("\n=== Results Finite Test ===");
    
    let train_data = generate_simple_test_data(10, 64);
    let test_data = generate_simple_test_data(8, 64);
    let weights = dtw_weights(64, 0.05);
    
    // Test all distance functions
    let (d, q, s, ds, m) = get_device();
    let result = dtw(d, q, s, ds, m, &train_data, &test_data);
    for row in &result {
        for &val in row {
            assert!(val.is_finite(), "DTW produced non-finite value: {}", val);
        }
    }
    
    let (d, q, s, ds, m) = get_device();
    let result = erp(d, q, s, ds, m, &train_data, &test_data, 0.0);
    for row in &result {
        for &val in row {
            assert!(val.is_finite(), "ERP produced non-finite value: {}", val);
        }
    }
    
    let (d, q, s, ds, m) = get_device();
    let result = msm(d, q, s, ds, m, &train_data, &test_data);
    for row in &result {
        for &val in row {
            assert!(val.is_finite(), "MSM produced non-finite value: {}", val);
        }
    }
    
    let (d, q, s, ds, m) = get_device();
    let result = wdtw(d, q, s, ds, m, &train_data, &test_data, &weights);
    for row in &result {
        for &val in row {
            assert!(val.is_finite(), "WDTW produced non-finite value: {}", val);
        }
    }
    
    let (d, q, s, ds, m) = get_device();
    let result = adtw(d, q, s, ds, m, &train_data, &test_data, 0.1);
    for row in &result {
        for &val in row {
            assert!(val.is_finite(), "ADTW produced non-finite value: {}", val);
        }
    }
    
    println!("✓ All results are finite!");
}
