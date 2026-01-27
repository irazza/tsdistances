/*
 * tsdistances.h - C API header for tsdistances library
 * 
 * This header defines the C interface for the tsdistances Rust library,
 * enabling integration with MATLAB, Python ctypes, and other languages.
 */

#ifndef TSDISTANCES_H
#define TSDISTANCES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Result structure returned by all distance functions.
 * 
 * The data pointer contains the distance matrix in column-major order
 * (suitable for MATLAB/Fortran). Memory must be freed using tsd_free_result().
 */
typedef struct {
    /** Pointer to distance matrix data (column-major order) */
    double* data;
    /** Number of rows in the result matrix */
    size_t rows;
    /** Number of columns in the result matrix */
    size_t cols;
    /** Error code: 0 = success, non-zero = error */
    int32_t error_code;
} DistanceResult;

/**
 * Free memory allocated for a DistanceResult.
 * 
 * @param result Pointer to the result to free (can be NULL)
 */
void tsd_free_result(DistanceResult* result);

/**
 * Compute Euclidean distance matrix between time series sets.
 * 
 * @param x1_data Pointer to first dataset (column-major, M x N)
 * @param x1_rows Number of time series in x1
 * @param x1_cols Length of each time series in x1
 * @param x2_data Pointer to second dataset (NULL for pairwise within x1)
 * @param x2_rows Number of time series in x2
 * @param x2_cols Length of each time series in x2
 * @param parallel Enable parallel computation
 * @return DistanceResult containing the distance matrix
 */
DistanceResult tsd_euclidean(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    bool parallel
);

/**
 * Compute Catch22-Euclidean distance matrix.
 * 
 * Time series are first transformed using the Catch22 feature set,
 * then Euclidean distance is computed in the feature space.
 */
DistanceResult tsd_catch_euclidean(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    bool parallel
);

/**
 * Compute Edit Distance with Real Penalty (ERP) matrix.
 * 
 * @param sakoe_chiba_band Band constraint [0-1], 1.0 = no constraint
 * @param gap_penalty Penalty for gap insertion/deletion
 */
DistanceResult tsd_erp(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    double sakoe_chiba_band, double gap_penalty, bool parallel
);

/**
 * Compute Longest Common Subsequence (LCSS) distance matrix.
 * 
 * @param sakoe_chiba_band Band constraint [0-1]
 * @param epsilon Matching threshold for element similarity
 */
DistanceResult tsd_lcss(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    double sakoe_chiba_band, double epsilon, bool parallel
);

/**
 * Compute Dynamic Time Warping (DTW) distance matrix.
 * 
 * @param sakoe_chiba_band Band constraint [0-1], 1.0 = no constraint
 */
DistanceResult tsd_dtw(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    double sakoe_chiba_band, bool parallel
);

/**
 * Compute Derivative Dynamic Time Warping (DDTW) distance matrix.
 * 
 * DTW applied to the derivatives of the time series.
 */
DistanceResult tsd_ddtw(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    double sakoe_chiba_band, bool parallel
);

/**
 * Compute Weighted Dynamic Time Warping (WDTW) distance matrix.
 * 
 * @param g Weight parameter controlling warping penalty (typically 0.05)
 */
DistanceResult tsd_wdtw(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    double sakoe_chiba_band, double g, bool parallel
);

/**
 * Compute Weighted Derivative DTW (WDDTW) distance matrix.
 */
DistanceResult tsd_wddtw(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    double sakoe_chiba_band, double g, bool parallel
);

/**
 * Compute Amerced Dynamic Time Warping (ADTW) distance matrix.
 * 
 * @param warp_penalty Penalty added for each warping step
 */
DistanceResult tsd_adtw(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    double sakoe_chiba_band, double warp_penalty, bool parallel
);

/**
 * Compute Move-Split-Merge (MSM) distance matrix.
 * 
 * @param cost Cost for split/merge operations
 */
DistanceResult tsd_msm(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    double cost, bool parallel
);

/**
 * Compute Time Warp Edit (TWE) distance matrix.
 * 
 * @param stiffness Stiffness parameter (nu)
 * @param penalty Edit penalty (lambda)
 */
DistanceResult tsd_twe(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    double sakoe_chiba_band,
    double stiffness, double penalty, bool parallel
);

/**
 * Compute Shape-Based Distance (SBD) matrix.
 * 
 * SBD is based on normalized cross-correlation and is shift-invariant.
 */
DistanceResult tsd_sbd(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    bool parallel
);

/**
 * Compute Matrix Profile distance matrix.
 * 
 * @param window_size Size of the subsequence window
 */
DistanceResult tsd_mp(
    const double* x1_data, size_t x1_rows, size_t x1_cols,
    const double* x2_data, size_t x2_rows, size_t x2_cols,
    size_t window_size, bool parallel
);

#ifdef __cplusplus
}
#endif

#endif /* TSDISTANCES_H */
