//! PyO3 bindings for the time-series distance functions.
//!
//! These are thin wrappers around the binding-agnostic implementations in
//! [`crate::core`]. Keeping the actual algorithms in `core` (which has no PyO3
//! dependency) lets the MATLAB C FFI reuse the exact same code, so the two
//! bindings can never drift apart.

use crate::core;
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

#[pyfunction]
#[pyo3(signature = (x1, x2=None, par=true))]
pub fn euclidean(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    par: Option<bool>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::euclidean(x1, x2, par.unwrap_or(true))?)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, par=true))]
pub fn catch_euclidean(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    par: Option<bool>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::catch_euclidean(x1, x2, par.unwrap_or(true))?)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, gap_penalty=0.0, par=true, device="cpu"))]
pub fn erp(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    band: Option<f64>,
    gap_penalty: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::erp(
        x1,
        x2,
        band.unwrap_or(1.0),
        gap_penalty.unwrap_or(0.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, epsilon=1.0, par=true, device="cpu"))]
pub fn lcss(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    band: Option<f64>,
    epsilon: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::lcss(
        x1,
        x2,
        band.unwrap_or(1.0),
        epsilon.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, par=true, device="cpu"))]
pub fn dtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    band: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::dtw(
        x1,
        x2,
        band.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, par=true, device="cpu"))]
pub fn ddtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    band: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::ddtw(
        x1,
        x2,
        band.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, g=0.05, par=true, device="cpu"))]
pub fn wdtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    band: Option<f64>,
    g: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::wdtw(
        x1,
        x2,
        band.unwrap_or(1.0),
        g.unwrap_or(0.05),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, g=0.05, par=true, device="cpu"))]
pub fn wddtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    band: Option<f64>,
    g: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::wddtw(
        x1,
        x2,
        band.unwrap_or(1.0),
        g.unwrap_or(0.05),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, warp_penalty=1.0, par=true, device="cpu"))]
pub fn adtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    band: Option<f64>,
    warp_penalty: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::adtw(
        x1,
        x2,
        band.unwrap_or(1.0),
        warp_penalty.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, par=true, device="cpu"))]
pub fn msm(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    band: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::msm(
        x1,
        x2,
        band.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, band=1.0, stiffness=0.001, penalty=1.0, par=true, device="cpu"))]
pub fn twe(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    band: Option<f64>,
    stiffness: Option<f64>,
    penalty: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::twe(
        x1,
        x2,
        band.unwrap_or(1.0),
        stiffness.unwrap_or(0.001),
        penalty.unwrap_or(1.0),
        par.unwrap_or(true),
        device.unwrap_or("cpu"),
    )?)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, par=true))]
pub fn sb(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    par: Option<bool>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::sbd(x1, x2, par.unwrap_or(true))?)
}

#[pyfunction]
#[pyo3(signature = (x1, window=20, x2=None, par=true))]
pub fn mp(
    x1: Vec<Vec<f64>>,
    window: Option<i32>,
    x2: Option<Vec<Vec<f64>>>,
    par: Option<bool>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(core::mp(x1, x2, window.unwrap_or(20), par.unwrap_or(true))?)
}
