pub mod diagonal;
pub mod matrix;
mod utils;

// Core module with pure Rust implementations (no PyO3 dependency)
pub mod core;

// Python bindings (only when "python" feature is enabled)
#[cfg(feature = "python")]
pub mod distances;

// MATLAB FFI bindings (only when "matlab" feature is enabled)
#[cfg(feature = "matlab")]
pub mod matlab_ffi;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "tsdistances")]
fn py_module(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    let _ = ctrlc::set_handler(move || {
        println!("\nraise KeyboardInterrupt (Ctrl+C pressed)");
        std::process::exit(1);
    });

    m.add_function(wrap_pyfunction!(distances::euclidean, m)?)?;
    m.add_function(wrap_pyfunction!(distances::catch_euclidean, m)?)?;
    m.add_function(wrap_pyfunction!(distances::erp, m)?)?;
    m.add_function(wrap_pyfunction!(distances::lcss, m)?)?;
    m.add_function(wrap_pyfunction!(distances::dtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::ddtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::wdtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::wddtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::adtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::msm, m)?)?;
    m.add_function(wrap_pyfunction!(distances::twe, m)?)?;
    m.add_function(wrap_pyfunction!(distances::sb, m)?)?;
    m.add_function(wrap_pyfunction!(distances::mp, m)?)?;
    Ok(())
}
