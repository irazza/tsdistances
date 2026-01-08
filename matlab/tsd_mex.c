/*
 * tsd_mex.c - MEX gateway for tsdistances Rust library
 * 
 * This file provides the interface between MATLAB and the tsdistances
 * Rust library for computing various time series distance measures.
 *
 * Compile with: mex tsd_mex.c -L../target/release -ltsdistances_matlab
 */

#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

/* Result structure matching the Rust FFI */
typedef struct {
    double* data;
    size_t rows;
    size_t cols;
    int32_t error_code;
} DistanceResult;

/* Function declarations from the Rust library */
extern void tsd_free_result(DistanceResult* result);
extern DistanceResult tsd_euclidean(const double* x1_data, size_t x1_rows, size_t x1_cols,
                                     const double* x2_data, size_t x2_rows, size_t x2_cols,
                                     bool parallel);
extern DistanceResult tsd_catch_euclidean(const double* x1_data, size_t x1_rows, size_t x1_cols,
                                          const double* x2_data, size_t x2_rows, size_t x2_cols,
                                          bool parallel);
extern DistanceResult tsd_erp(const double* x1_data, size_t x1_rows, size_t x1_cols,
                              const double* x2_data, size_t x2_rows, size_t x2_cols,
                              double sakoe_chiba_band, double gap_penalty, bool parallel);
extern DistanceResult tsd_lcss(const double* x1_data, size_t x1_rows, size_t x1_cols,
                               const double* x2_data, size_t x2_rows, size_t x2_cols,
                               double sakoe_chiba_band, double epsilon, bool parallel);
extern DistanceResult tsd_dtw(const double* x1_data, size_t x1_rows, size_t x1_cols,
                              const double* x2_data, size_t x2_rows, size_t x2_cols,
                              double sakoe_chiba_band, bool parallel);
extern DistanceResult tsd_ddtw(const double* x1_data, size_t x1_rows, size_t x1_cols,
                               const double* x2_data, size_t x2_rows, size_t x2_cols,
                               double sakoe_chiba_band, bool parallel);
extern DistanceResult tsd_wdtw(const double* x1_data, size_t x1_rows, size_t x1_cols,
                               const double* x2_data, size_t x2_rows, size_t x2_cols,
                               double sakoe_chiba_band, double g, bool parallel);
extern DistanceResult tsd_wddtw(const double* x1_data, size_t x1_rows, size_t x1_cols,
                                const double* x2_data, size_t x2_rows, size_t x2_cols,
                                double sakoe_chiba_band, double g, bool parallel);
extern DistanceResult tsd_adtw(const double* x1_data, size_t x1_rows, size_t x1_cols,
                               const double* x2_data, size_t x2_rows, size_t x2_cols,
                               double sakoe_chiba_band, double warp_penalty, bool parallel);
extern DistanceResult tsd_msm(const double* x1_data, size_t x1_rows, size_t x1_cols,
                              const double* x2_data, size_t x2_rows, size_t x2_cols,
                              double cost, bool parallel);
extern DistanceResult tsd_twe(const double* x1_data, size_t x1_rows, size_t x1_cols,
                              const double* x2_data, size_t x2_rows, size_t x2_cols,
                              double stiffness, double penalty, bool parallel);
extern DistanceResult tsd_sbd(const double* x1_data, size_t x1_rows, size_t x1_cols,
                              const double* x2_data, size_t x2_rows, size_t x2_cols,
                              bool parallel);
extern DistanceResult tsd_mp(const double* x1_data, size_t x1_rows, size_t x1_cols,
                             const double* x2_data, size_t x2_rows, size_t x2_cols,
                             size_t window_size, bool parallel);

/* Helper function to extract input matrices */
void get_input_data(const mxArray* prhs[], int nrhs,
                    const double** x1_data, size_t* x1_rows, size_t* x1_cols,
                    const double** x2_data, size_t* x2_rows, size_t* x2_cols,
                    bool* parallel) {
    
    /* X1 is required (input 1, index 1 since 0 is function name) */
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("tsdistances:invalidInput", "X1 must be a real double matrix.");
    }
    *x1_data = mxGetPr(prhs[1]);
    *x1_rows = mxGetM(prhs[1]);
    *x1_cols = mxGetN(prhs[1]);
    
    /* X2 is optional (input 2) */
    if (nrhs > 2 && !mxIsEmpty(prhs[2])) {
        if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2])) {
            mexErrMsgIdAndTxt("tsdistances:invalidInput", "X2 must be a real double matrix.");
        }
        *x2_data = mxGetPr(prhs[2]);
        *x2_rows = mxGetM(prhs[2]);
        *x2_cols = mxGetN(prhs[2]);
    } else {
        *x2_data = NULL;
        *x2_rows = 0;
        *x2_cols = 0;
    }
    
    /* Parallel flag (input 3) */
    if (nrhs > 3 && !mxIsEmpty(prhs[3])) {
        *parallel = mxIsLogicalScalarTrue(prhs[3]) || 
                    (mxIsDouble(prhs[3]) && mxGetScalar(prhs[3]) != 0);
    } else {
        *parallel = true;
    }
}

/* Helper function to create output matrix from result */
mxArray* create_output(DistanceResult* result) {
    if (result->error_code != 0) {
        mexErrMsgIdAndTxt("tsdistances:computationError", 
                          "Error computing distance (code: %d).", result->error_code);
    }
    
    mxArray* output = mxCreateDoubleMatrix(result->rows, result->cols, mxREAL);
    double* out_data = mxGetPr(output);
    
    /* Data is already in column-major order from Rust */
    memcpy(out_data, result->data, result->rows * result->cols * sizeof(double));
    
    /* Free the Rust-allocated memory */
    tsd_free_result(result);
    
    return output;
}

/* Main MEX gateway function */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    char func_name[64];
    const double *x1_data, *x2_data;
    size_t x1_rows, x1_cols, x2_rows, x2_cols;
    bool parallel;
    DistanceResult result;
    
    /* Check minimum arguments */
    if (nrhs < 2) {
        mexErrMsgIdAndTxt("tsdistances:invalidInput",
                          "Usage: D = tsd_mex(function_name, X1, [X2], [parallel], ...)");
    }
    
    /* Get function name */
    if (!mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("tsdistances:invalidInput", "First argument must be function name string.");
    }
    mxGetString(prhs[0], func_name, sizeof(func_name));
    
    /* Get input data */
    get_input_data(prhs, nrhs, &x1_data, &x1_rows, &x1_cols, &x2_data, &x2_rows, &x2_cols, &parallel);
    
    /* Call appropriate function */
    if (strcmp(func_name, "euclidean") == 0) {
        result = tsd_euclidean(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, parallel);
    }
    else if (strcmp(func_name, "catch_euclidean") == 0) {
        result = tsd_catch_euclidean(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, parallel);
    }
    else if (strcmp(func_name, "erp") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double gap_penalty = (nrhs > 5) ? mxGetScalar(prhs[5]) : 0.0;
        result = tsd_erp(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, gap_penalty, parallel);
    }
    else if (strcmp(func_name, "lcss") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double epsilon = (nrhs > 5) ? mxGetScalar(prhs[5]) : 1.0;
        result = tsd_lcss(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, epsilon, parallel);
    }
    else if (strcmp(func_name, "dtw") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        result = tsd_dtw(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, parallel);
    }
    else if (strcmp(func_name, "ddtw") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        result = tsd_ddtw(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, parallel);
    }
    else if (strcmp(func_name, "wdtw") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double g = (nrhs > 5) ? mxGetScalar(prhs[5]) : 0.05;
        result = tsd_wdtw(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, g, parallel);
    }
    else if (strcmp(func_name, "wddtw") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double g = (nrhs > 5) ? mxGetScalar(prhs[5]) : 0.05;
        result = tsd_wddtw(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, g, parallel);
    }
    else if (strcmp(func_name, "adtw") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double warp_penalty = (nrhs > 5) ? mxGetScalar(prhs[5]) : 1.0;
        result = tsd_adtw(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, warp_penalty, parallel);
    }
    else if (strcmp(func_name, "msm") == 0) {
        double cost = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        result = tsd_msm(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, cost, parallel);
    }
    else if (strcmp(func_name, "twe") == 0) {
        double stiffness = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double penalty = (nrhs > 5) ? mxGetScalar(prhs[5]) : 1.0;
        result = tsd_twe(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, stiffness, penalty, parallel);
    }
    else if (strcmp(func_name, "sbd") == 0) {
        result = tsd_sbd(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, parallel);
    }
    else if (strcmp(func_name, "mp") == 0) {
        size_t window_size = (nrhs > 4) ? (size_t)mxGetScalar(prhs[4]) : x1_cols / 4;
        result = tsd_mp(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, window_size, parallel);
    }
    else {
        mexErrMsgIdAndTxt("tsdistances:unknownFunction", "Unknown distance function: %s", func_name);
    }
    
    /* Create output */
    plhs[0] = create_output(&result);
}
