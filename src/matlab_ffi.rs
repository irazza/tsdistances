//! C FFI bindings for MATLAB integration
//!
//! This module provides C-compatible functions that can be called from MATLAB
//! through MEX files.

use std::slice;

use crate::core;

/// Result structure for returning distance matrices to MATLAB
#[repr(C)]
pub struct DistanceResult {
    /// Pointer to the distance matrix data (row-major order)
    pub data: *mut f64,
    /// Number of rows in the result matrix
    pub rows: usize,
    /// Number of columns in the result matrix
    pub cols: usize,
    /// Error code: 0 = success, non-zero = error
    pub error_code: i32,
}

impl DistanceResult {
    fn success(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };

        // Flatten the 2D vector to 1D (column-major for MATLAB)
        let mut flat_data: Vec<f64> = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            for row in 0..rows {
                flat_data.push(data[row][col]);
            }
        }

        let mut boxed = flat_data.into_boxed_slice();
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);

        DistanceResult {
            data: ptr,
            rows,
            cols,
            error_code: 0,
        }
    }

    fn error(code: i32) -> Self {
        DistanceResult {
            data: std::ptr::null_mut(),
            rows: 0,
            cols: 0,
            error_code: code,
        }
    }
}

/// Free memory allocated for a DistanceResult
///
/// # Safety
/// The caller must ensure that the pointer was allocated by this library
/// and has not been freed before.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_free_result(result: *mut DistanceResult) {
    if !result.is_null() {
        unsafe {
            let res = &*result;
            if !res.data.is_null() && res.rows > 0 && res.cols > 0 {
                let size = res.rows * res.cols;
                let _ = Vec::from_raw_parts(res.data, size, size);
            }
        }
    }
}

/// Helper function to convert C arrays to Rust vectors
///
/// # Safety
/// The caller must ensure that the pointers are valid and the sizes are correct.
unsafe fn c_arrays_to_vecs(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
) -> (Vec<Vec<f64>>, Option<Vec<Vec<f64>>>) {
    // Convert x1 from column-major (MATLAB) to row-major (Rust)
    let x1_slice = unsafe { slice::from_raw_parts(x1_data, x1_rows * x1_cols) };
    let mut x1: Vec<Vec<f64>> = Vec::with_capacity(x1_rows);
    for i in 0..x1_rows {
        let mut row = Vec::with_capacity(x1_cols);
        for j in 0..x1_cols {
            row.push(x1_slice[j * x1_rows + i]);
        }
        x1.push(row);
    }

    // Convert x2 if provided
    let x2 = if !x2_data.is_null() && x2_rows > 0 && x2_cols > 0 {
        let x2_slice = unsafe { slice::from_raw_parts(x2_data, x2_rows * x2_cols) };
        let mut x2_vec: Vec<Vec<f64>> = Vec::with_capacity(x2_rows);
        for i in 0..x2_rows {
            let mut row = Vec::with_capacity(x2_cols);
            for j in 0..x2_cols {
                row.push(x2_slice[j * x2_rows + i]);
            }
            x2_vec.push(row);
        }
        Some(x2_vec)
    } else {
        None
    };

    (x1, x2)
}

/// Compute Euclidean distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_euclidean(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::euclidean(x1, x2, parallel) {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute Catch22-Euclidean distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_catch_euclidean(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::catch_euclidean(x1, x2, parallel) {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute ERP (Edit Distance with Real Penalty) distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_erp(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    sakoe_chiba_band: f64,
    gap_penalty: f64,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::erp(x1, x2, sakoe_chiba_band, gap_penalty, parallel, "cpu") {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute LCSS (Longest Common Subsequence) distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_lcss(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    sakoe_chiba_band: f64,
    epsilon: f64,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::lcss(x1, x2, sakoe_chiba_band, epsilon, parallel, "cpu") {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute DTW (Dynamic Time Warping) distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_dtw(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    sakoe_chiba_band: f64,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::dtw(x1, x2, sakoe_chiba_band, parallel, "cpu") {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute DDTW (Derivative Dynamic Time Warping) distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_ddtw(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    sakoe_chiba_band: f64,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::ddtw(x1, x2, sakoe_chiba_band, parallel, "cpu") {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute WDTW (Weighted Dynamic Time Warping) distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_wdtw(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    sakoe_chiba_band: f64,
    g: f64,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::wdtw(x1, x2, sakoe_chiba_band, g, parallel, "cpu") {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute WDDTW (Weighted Derivative Dynamic Time Warping) distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_wddtw(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    sakoe_chiba_band: f64,
    g: f64,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::wddtw(x1, x2, sakoe_chiba_band, g, parallel, "cpu") {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute ADTW (Amerced Dynamic Time Warping) distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_adtw(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    sakoe_chiba_band: f64,
    warp_penalty: f64,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::adtw(x1, x2, sakoe_chiba_band, warp_penalty, parallel, "cpu") {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute MSM (Move-Split-Merge) distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_msm(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    cost: f64,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::msm(x1, x2, cost, parallel, "cpu") {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute TWE (Time Warp Edit) distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_twe(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    stiffness: f64,
    penalty: f64,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::twe(x1, x2, 1.0, stiffness, penalty, parallel, "cpu") {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute SBD (Shape-Based Distance) distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_sbd(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::sbd(x1, x2, parallel) {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}

/// Compute MP (Matrix Profile) distance matrix
///
/// # Safety
/// All pointers must be valid. x2_data can be null for pairwise distance within x1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsd_mp(
    x1_data: *const f64,
    x1_rows: usize,
    x1_cols: usize,
    x2_data: *const f64,
    x2_rows: usize,
    x2_cols: usize,
    window_size: usize,
    parallel: bool,
) -> DistanceResult {
    let (x1, x2) =
        unsafe { c_arrays_to_vecs(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols) };

    match core::mp(x1, x2, window_size as i32, parallel) {
        Ok(result) => DistanceResult::success(result),
        Err(_) => DistanceResult::error(1),
    }
}
