//! PyO3 bindings for the time-series distance functions.
//!
//! These are thin wrappers around the binding-agnostic implementations in
//! [`crate::core`]. Keeping the actual algorithms in `core` (which has no PyO3
//! dependency) lets the MATLAB C FFI reuse the exact same code, so the two
//! bindings can never drift apart.
//!
//! Inputs arrive as numpy `PyReadonlyArray2<f64>` and are borrowed row-by-row
//! straight into `core` (zero-copy for the common C-contiguous float64 case).
//! Outputs are built directly as a numpy `PyArray2`, so no Python
//! list-of-lists is ever materialized.

use crate::core;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Bridge `core::DistanceError` into a Python exception.
///
/// Defined here (rather than in `core`) so that `core` stays free of any PyO3
/// dependency for the MATLAB-only build. The orphan rule is satisfied because
/// `DistanceError` is a local type. `InvalidParameter` maps to `ValueError` to
/// preserve the exception type the Python API has always raised.
impl From<core::DistanceError> for PyErr {
    fn from(err: core::DistanceError) -> PyErr {
        match err {
            core::DistanceError::InvalidParameter(msg) => {
                pyo3::exceptions::PyValueError::new_err(msg)
            }
            core::DistanceError::ComputationError(msg) => {
                pyo3::exceptions::PyRuntimeError::new_err(msg)
            }
        }
    }
}

/// Borrow the rows of a 2-D numpy array as `&[f64]` slices.
///
/// The returned slices point directly into the numpy buffer (zero-copy). The
/// array must be C-contiguous; the Python wrapper guarantees this by calling
/// `np.ascontiguousarray`, but a non-contiguous buffer that reached us anyway
/// is reported as a `ValueError` rather than panicking.
fn rows_of<'a>(arr: &'a PyReadonlyArray2<'_, f64>) -> PyResult<Vec<&'a [f64]>> {
    let shape = arr.shape();
    let (nrows, ncols) = (shape[0], shape[1]);
    let flat = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("input array must be C-contiguous float64"))?;
    // `chunks_exact(0)` panics, so handle the zero-column case explicitly: each
    // of the `nrows` series is an empty slice.
    if ncols == 0 {
        return Ok(vec![&[][..]; nrows]);
    }
    Ok(flat.chunks_exact(ncols).collect())
}

/// Build a numpy `PyArray2` from the rectangular matrix `core` returns.
fn to_pyarray2(py: Python<'_>, matrix: Vec<Vec<f64>>) -> PyResult<Bound<'_, PyArray2<f64>>> {
    PyArray2::from_vec2(py, &matrix).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, par=true))]
pub fn euclidean<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    par: Option<bool>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::euclidean(&x1_rows, x2_rows.as_deref(), par.unwrap_or(true))?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, par=true))]
pub fn catch_euclidean<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    par: Option<bool>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::catch_euclidean(&x1_rows, x2_rows.as_deref(), par.unwrap_or(true))?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, gap_penalty=0.0, par=true, device="cpu"))]
pub fn erp<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    band: Option<f64>,
    gap_penalty: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::erp(
        &x1_rows,
        x2_rows.as_deref(),
        band.unwrap_or(1.0),
        gap_penalty.unwrap_or(0.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, epsilon=1.0, par=true, device="cpu"))]
pub fn lcss<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    band: Option<f64>,
    epsilon: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::lcss(
        &x1_rows,
        x2_rows.as_deref(),
        band.unwrap_or(1.0),
        epsilon.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, par=true, device="cpu"))]
pub fn dtw<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    band: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::dtw(
        &x1_rows,
        x2_rows.as_deref(),
        band.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, par=true, device="cpu"))]
pub fn ddtw<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    band: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::ddtw(
        &x1_rows,
        x2_rows.as_deref(),
        band.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, g=0.05, par=true, device="cpu"))]
pub fn wdtw<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    band: Option<f64>,
    g: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::wdtw(
        &x1_rows,
        x2_rows.as_deref(),
        band.unwrap_or(1.0),
        g.unwrap_or(0.05),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, g=0.05, par=true, device="cpu"))]
pub fn wddtw<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    band: Option<f64>,
    g: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::wddtw(
        &x1_rows,
        x2_rows.as_deref(),
        band.unwrap_or(1.0),
        g.unwrap_or(0.05),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, warp_penalty=1.0, par=true, device="cpu"))]
pub fn adtw<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    band: Option<f64>,
    warp_penalty: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::adtw(
        &x1_rows,
        x2_rows.as_deref(),
        band.unwrap_or(1.0),
        warp_penalty.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, par=true, device="cpu"))]
pub fn msm<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    band: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::msm(
        &x1_rows,
        x2_rows.as_deref(),
        band.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, stiffness=0.001, penalty=1.0, par=true, device="cpu"))]
#[allow(clippy::too_many_arguments)]
pub fn twe<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    band: Option<f64>,
    stiffness: Option<f64>,
    penalty: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::twe(
        &x1_rows,
        x2_rows.as_deref(),
        band.unwrap_or(1.0),
        stiffness.unwrap_or(0.001),
        penalty.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, par=true))]
pub fn sb<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    par: Option<bool>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::sbd(&x1_rows, x2_rows.as_deref(), par.unwrap_or(true))?;
    to_pyarray2(py, result)
}

#[pyfunction]
#[pyo3(signature = (x1, window=20, x2=None, par=true))]
pub fn mp<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray2<'py, f64>,
    window: Option<i32>,
    x2: Option<PyReadonlyArray2<'py, f64>>,
    par: Option<bool>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x1_rows = rows_of(&x1)?;
    let x2_rows = x2.as_ref().map(rows_of).transpose()?;
    let result = core::mp(
        &x1_rows,
        x2_rows.as_deref(),
        window.unwrap_or(20),
        par.unwrap_or(true),
    )?;
    to_pyarray2(py, result)
}
