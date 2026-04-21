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

typedef struct {
    const double* data;
    const size_t* lengths;
    size_t rows;
    size_t total_values;
} RaggedInput;

typedef struct {
    RaggedInput input;
    bool owns_data;
    bool owns_lengths;
} RaggedInputOwned;

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
                              double sakoe_chiba_band, double stiffness, double penalty, bool parallel);
extern DistanceResult tsd_sbd(const double* x1_data, size_t x1_rows, size_t x1_cols,
                              const double* x2_data, size_t x2_rows, size_t x2_cols,
                              bool parallel);
extern DistanceResult tsd_mp(const double* x1_data, size_t x1_rows, size_t x1_cols,
                             const double* x2_data, size_t x2_rows, size_t x2_cols,
                             size_t window_size, bool parallel);
extern DistanceResult tsd_euclidean_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                           bool parallel);
extern DistanceResult tsd_catch_euclidean_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                                 bool parallel);
extern DistanceResult tsd_erp_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                     double sakoe_chiba_band, double gap_penalty, bool parallel);
extern DistanceResult tsd_lcss_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                      double sakoe_chiba_band, double epsilon, bool parallel);
extern DistanceResult tsd_dtw_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                     double sakoe_chiba_band, bool parallel);
extern DistanceResult tsd_ddtw_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                      double sakoe_chiba_band, bool parallel);
extern DistanceResult tsd_wdtw_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                      double sakoe_chiba_band, double g, bool parallel);
extern DistanceResult tsd_wddtw_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                       double sakoe_chiba_band, double g, bool parallel);
extern DistanceResult tsd_adtw_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                      double sakoe_chiba_band, double warp_penalty, bool parallel);
extern DistanceResult tsd_msm_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                     double cost, bool parallel);
extern DistanceResult tsd_twe_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                     double sakoe_chiba_band, double stiffness, double penalty,
                                     bool parallel);
extern DistanceResult tsd_sbd_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                     bool parallel);
extern DistanceResult tsd_mp_ragged(const RaggedInput* x1, const RaggedInput* x2,
                                    size_t window_size, bool parallel);

static void init_ragged_owned(RaggedInputOwned* owned) {
    owned->input.data = NULL;
    owned->input.lengths = NULL;
    owned->input.rows = 0;
    owned->input.total_values = 0;
    owned->owns_data = false;
    owned->owns_lengths = false;
}

static void free_ragged_owned(RaggedInputOwned* owned) {
    if (owned->owns_data && owned->input.data != NULL) {
        mxFree((void*)owned->input.data);
    }
    if (owned->owns_lengths && owned->input.lengths != NULL) {
        mxFree((void*)owned->input.lengths);
    }
    init_ragged_owned(owned);
}

static bool get_parallel_flag(const mxArray* prhs[], int nrhs) {
    if (nrhs > 3 && !mxIsEmpty(prhs[3])) {
        return mxIsLogicalScalarTrue(prhs[3]) ||
               (mxIsDouble(prhs[3]) && mxGetScalar(prhs[3]) != 0);
    }
    return true;
}

static void parse_matrix_input(
    const mxArray* array,
    const double** data,
    size_t* rows,
    size_t* cols,
    const char* arg_name
) {
    if (!mxIsDouble(array) || mxIsComplex(array)) {
        mexErrMsgIdAndTxt("tsdistances:invalidInput", "%s must be a real double matrix.", arg_name);
    }
    *data = mxGetPr(array);
    *rows = mxGetM(array);
    *cols = mxGetN(array);
}

static void parse_matrix_as_ragged(const mxArray* array, RaggedInputOwned* out, const char* arg_name) {
    size_t rows = 0;
    size_t cols = 0;
    const double* data = NULL;
    parse_matrix_input(array, &data, &rows, &cols, arg_name);

    out->input.rows = rows;
    out->input.total_values = rows * cols;
    out->input.data = data;
    out->input.lengths = (size_t*)mxMalloc(rows * sizeof(size_t));
    out->owns_data = false;
    out->owns_lengths = true;

    for (size_t i = 0; i < rows; ++i) {
        ((size_t*)out->input.lengths)[i] = cols;
    }
}

static void parse_cell_as_ragged(const mxArray* array, RaggedInputOwned* out, const char* arg_name) {
    if (!mxIsCell(array)) {
        mexErrMsgIdAndTxt("tsdistances:invalidInput", "%s must be a cell array.", arg_name);
    }

    size_t rows = mxGetNumberOfElements(array);
    size_t* lengths = (size_t*)mxMalloc(rows * sizeof(size_t));
    size_t total = 0;

    for (size_t i = 0; i < rows; ++i) {
        const mxArray* cell = mxGetCell(array, i);
        if (cell == NULL || !mxIsDouble(cell) || mxIsComplex(cell)) {
            mxFree(lengths);
            mexErrMsgIdAndTxt("tsdistances:invalidInput",
                              "%s cells must contain real double vectors.", arg_name);
        }
        if (mxGetNumberOfDimensions(cell) > 2 || !(mxGetM(cell) == 1 || mxGetN(cell) == 1)) {
            mxFree(lengths);
            mexErrMsgIdAndTxt("tsdistances:invalidInput",
                              "%s cells must be row or column vectors.", arg_name);
        }
        lengths[i] = mxGetNumberOfElements(cell);
        total += lengths[i];
    }

    double* data = NULL;
    if (total > 0) {
        data = (double*)mxMalloc(total * sizeof(double));
    }

    size_t offset = 0;
    for (size_t i = 0; i < rows; ++i) {
        const mxArray* cell = mxGetCell(array, i);
        size_t len = lengths[i];
        if (len > 0) {
            const double* cell_data = mxGetPr(cell);
            memcpy(data + offset, cell_data, len * sizeof(double));
            offset += len;
        }
    }

    out->input.data = data;
    out->input.lengths = lengths;
    out->input.rows = rows;
    out->input.total_values = total;
    out->owns_data = true;
    out->owns_lengths = true;
}

static void parse_any_as_ragged(const mxArray* array, RaggedInputOwned* out, const char* arg_name) {
    if (mxIsCell(array)) {
        parse_cell_as_ragged(array, out, arg_name);
        return;
    }
    parse_matrix_as_ragged(array, out, arg_name);
}

/* Helper function to create output matrix from result */
mxArray* create_output(DistanceResult* result) {
    if (result->error_code != 0) {
        if (result->error_code == 2) {
            mexErrMsgIdAndTxt("tsdistances:invalidLength",
                              "Euclidean and Catch-Euclidean require equal length series.");
        }
        if (result->error_code == 3) {
            mexErrMsgIdAndTxt("tsdistances:invalidRaggedInput",
                              "Invalid ragged cell-array metadata passed to backend.");
        }
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
    bool parallel, use_ragged;
    DistanceResult result;
    RaggedInputOwned x1_ragged;
    RaggedInputOwned x2_ragged;
    const RaggedInput* x2_input = NULL;

    init_ragged_owned(&x1_ragged);
    init_ragged_owned(&x2_ragged);
    
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
    
    parallel = get_parallel_flag(prhs, nrhs);
    use_ragged = mxIsCell(prhs[1]) || (nrhs > 2 && !mxIsEmpty(prhs[2]) && mxIsCell(prhs[2]));

    if (use_ragged) {
        parse_any_as_ragged(prhs[1], &x1_ragged, "X1");
        if (nrhs > 2 && !mxIsEmpty(prhs[2])) {
            parse_any_as_ragged(prhs[2], &x2_ragged, "X2");
            x2_input = &x2_ragged.input;
        }
    } else {
        parse_matrix_input(prhs[1], &x1_data, &x1_rows, &x1_cols, "X1");
        if (nrhs > 2 && !mxIsEmpty(prhs[2])) {
            parse_matrix_input(prhs[2], &x2_data, &x2_rows, &x2_cols, "X2");
        } else {
            x2_data = NULL;
            x2_rows = 0;
            x2_cols = 0;
        }
    }
    
    /* Call appropriate function */
    if (strcmp(func_name, "euclidean") == 0) {
        if (use_ragged) {
            result = tsd_euclidean_ragged(&x1_ragged.input, x2_input, parallel);
        } else {
            result = tsd_euclidean(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, parallel);
        }
    }
    else if (strcmp(func_name, "catch_euclidean") == 0) {
        if (use_ragged) {
            result = tsd_catch_euclidean_ragged(&x1_ragged.input, x2_input, parallel);
        } else {
            result = tsd_catch_euclidean(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, parallel);
        }
    }
    else if (strcmp(func_name, "erp") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double gap_penalty = (nrhs > 5) ? mxGetScalar(prhs[5]) : 0.0;
        if (use_ragged) {
            result = tsd_erp_ragged(&x1_ragged.input, x2_input, band, gap_penalty, parallel);
        } else {
            result = tsd_erp(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, gap_penalty, parallel);
        }
    }
    else if (strcmp(func_name, "lcss") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double epsilon = (nrhs > 5) ? mxGetScalar(prhs[5]) : 1.0;
        if (use_ragged) {
            result = tsd_lcss_ragged(&x1_ragged.input, x2_input, band, epsilon, parallel);
        } else {
            result = tsd_lcss(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, epsilon, parallel);
        }
    }
    else if (strcmp(func_name, "dtw") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        if (use_ragged) {
            result = tsd_dtw_ragged(&x1_ragged.input, x2_input, band, parallel);
        } else {
            result = tsd_dtw(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, parallel);
        }
    }
    else if (strcmp(func_name, "ddtw") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        if (use_ragged) {
            result = tsd_ddtw_ragged(&x1_ragged.input, x2_input, band, parallel);
        } else {
            result = tsd_ddtw(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, parallel);
        }
    }
    else if (strcmp(func_name, "wdtw") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double g = (nrhs > 5) ? mxGetScalar(prhs[5]) : 0.05;
        if (use_ragged) {
            result = tsd_wdtw_ragged(&x1_ragged.input, x2_input, band, g, parallel);
        } else {
            result = tsd_wdtw(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, g, parallel);
        }
    }
    else if (strcmp(func_name, "wddtw") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double g = (nrhs > 5) ? mxGetScalar(prhs[5]) : 0.05;
        if (use_ragged) {
            result = tsd_wddtw_ragged(&x1_ragged.input, x2_input, band, g, parallel);
        } else {
            result = tsd_wddtw(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, g, parallel);
        }
    }
    else if (strcmp(func_name, "adtw") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double warp_penalty = (nrhs > 5) ? mxGetScalar(prhs[5]) : 1.0;
        if (use_ragged) {
            result = tsd_adtw_ragged(&x1_ragged.input, x2_input, band, warp_penalty, parallel);
        } else {
            result = tsd_adtw(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, warp_penalty, parallel);
        }
    }
    else if (strcmp(func_name, "msm") == 0) {
        double cost = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        if (use_ragged) {
            result = tsd_msm_ragged(&x1_ragged.input, x2_input, cost, parallel);
        } else {
            result = tsd_msm(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, cost, parallel);
        }
    }
    else if (strcmp(func_name, "twe") == 0) {
        double band = (nrhs > 4) ? mxGetScalar(prhs[4]) : 1.0;
        double stiffness = (nrhs > 5) ? mxGetScalar(prhs[5]) : 1.0;
        double penalty = (nrhs > 6) ? mxGetScalar(prhs[6]) : 1.0;
        if (use_ragged) {
            result = tsd_twe_ragged(&x1_ragged.input, x2_input, band, stiffness, penalty, parallel);
        } else {
            result = tsd_twe(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, band, stiffness, penalty, parallel);
        }
    }
    else if (strcmp(func_name, "sbd") == 0) {
        if (use_ragged) {
            result = tsd_sbd_ragged(&x1_ragged.input, x2_input, parallel);
        } else {
            result = tsd_sbd(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, parallel);
        }
    }
    else if (strcmp(func_name, "mp") == 0) {
        size_t window_size = 0;
        if (nrhs > 4) {
            window_size = (size_t)mxGetScalar(prhs[4]);
        } else if (use_ragged) {
            size_t first_len = x1_ragged.input.rows > 0 ? x1_ragged.input.lengths[0] : 0;
            window_size = first_len / 4;
        } else {
            window_size = x1_cols / 4;
        }

        if (use_ragged) {
            result = tsd_mp_ragged(&x1_ragged.input, x2_input, window_size, parallel);
        } else {
            result = tsd_mp(x1_data, x1_rows, x1_cols, x2_data, x2_rows, x2_cols, window_size, parallel);
        }
    }
    else {
        free_ragged_owned(&x1_ragged);
        free_ragged_owned(&x2_ragged);
        mexErrMsgIdAndTxt("tsdistances:unknownFunction", "Unknown distance function: %s", func_name);
    }
    
    /* Create output */
    plhs[0] = create_output(&result);
    free_ragged_owned(&x1_ragged);
    free_ragged_owned(&x2_ragged);
}
