use crate::{
    diagonal,
    matrix::WavefrontMatrix,
    utils::{
        cross_correlation, derivate, dtw_weights, l2_norm, max, min, msm_cost_function, zscore,
    },
};
use pyo3::prelude::*;
use rayon::prelude::*;
use tsdistances_gpu::utils::get_device;

fn compute_distance_gpu(
    distance: impl (Fn(&Vec<Vec<f32>>, &Vec<Vec<f32>>) -> Vec<Vec<f32>>) + Sync + Send,
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
) -> Vec<Vec<f64>> {
    let x1 = x1
        .into_iter()
        .map(|v| v.into_iter().map(|f| f as f32).collect())
        .collect::<Vec<_>>();
    let x2 = x2.map(|x2| {
        x2.into_iter()
            .map(|v| v.into_iter().map(|f| f as f32).collect())
            .collect::<Vec<_>>()
    });

    let result = distance(&x1, x2.as_ref().unwrap_or(&x1));

    result
        .into_iter()
        .map(|v| v.into_iter().map(|f| f as f64).collect())
        .collect()
}

/// Computes the pairwise distance between two sets of timeseries.
///
/// This function computes the distance between each pair of timeseries (one from each set) using the
/// provided distance function. The computation is parallelized across multiple threads to improve
/// performance. The number of threads used can be controlled via the `par` parameter.
///
fn compute_distance(
    distance: impl (Fn(&[f64], &[f64]) -> f64) + Sync + Send,
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    par: bool,
) -> Vec<Vec<f64>> {
    let x1 = x1.into_iter().enumerate().collect::<Vec<_>>();
    let distance_matrix = if par {
        x1.par_iter()
            .map(|(i, a)| {
                if let Some(x2) = &x2 {
                    x2.iter()
                        .map(|b| {
                            let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
                            distance(a, b)
                        })
                        .collect::<Vec<_>>()
                } else {
                    x1.iter()
                        .take(*i)
                        .map(|(_, b)| {
                            let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
                            distance(a, b)
                        })
                        .collect::<Vec<_>>()
                }
            })
            .collect::<Vec<_>>()
    } else {
        x1.iter()
            .map(|(i, a)| {
                if let Some(x2) = &x2 {
                    x2.iter()
                        .map(|b| {
                            let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
                            distance(a, b)
                        })
                        .collect::<Vec<_>>()
                } else {
                    x1.iter()
                        .take(*i)
                        .map(|(_, b)| {
                            let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
                            distance(a, b)
                        })
                        .collect::<Vec<_>>()
                }
            })
            .collect::<Vec<_>>()
    };

    if x2.is_none() {
        let mut distance_matrix = distance_matrix;
        for i in 0..distance_matrix.len() {
            let row_len = distance_matrix.len();
            distance_matrix[i].reserve(row_len - i);
            distance_matrix[i].push(0.0);
            for j in i + 1..distance_matrix.len() {
                let d = distance_matrix[j][i];
                distance_matrix[i].push(d);
            }
        }
        distance_matrix
    } else {
        distance_matrix
    }
}

fn next_multiple_of_n(x: usize, n: usize) -> usize {
    (x + n - 1) / n * n
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, par=true))]
pub fn euclidean(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    par: Option<bool>,
) -> PyResult<Vec<Vec<f64>>> {
    let distance_matrix = compute_distance(
        |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        },
        x1,
        x2,
        par.unwrap_or(true),
    );
    Ok(distance_matrix)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, par=true))]
pub fn catch_euclidean(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    par: Option<bool>,
) -> PyResult<Vec<Vec<f64>>> {
    let x1 = x1
        .iter()
        .map(|x| {
            let mut transformed_x = Vec::with_capacity(catch22::N_CATCH22);
            for i in 0..catch22::N_CATCH22 {
                let value = catch22::compute(&x, i);
                if value.is_nan() {
                    transformed_x.push(0.0);
                } else {
                    transformed_x.push(value);
                }
            }
            return transformed_x;
        })
        .collect::<Vec<Vec<_>>>();
    let x2 = if let Some(x2) = x2 {
        Some(
            x2.iter()
                .map(|x| {
                    let mut transformed_x = Vec::with_capacity(catch22::N_CATCH22);
                    for i in 0..catch22::N_CATCH22 {
                        let value = catch22::compute(&x, i);
                        if value.is_finite() {
                            transformed_x.push(value);
                        } else {
                            transformed_x.push(0.0);
                        }
                    }
                    return transformed_x;
                })
                .collect::<Vec<Vec<_>>>(),
        )
    } else {
        None
    };
    // Z-Normalize on the column-wise
    let mean_x1 = (0..catch22::N_CATCH22)
        .map(|i| {
            let sum = x1.iter().map(|x| x[i]).sum::<f64>();
            sum / x1.len() as f64
        })
        .collect::<Vec<f64>>();
    let std_x1 = (0..catch22::N_CATCH22)
        .map(|i| {
            let sum = x1.iter().map(|x| (x[i] - mean_x1[i]).powi(2)).sum::<f64>();
            (sum / x1.len() as f64).sqrt()
        })
        .collect::<Vec<f64>>();
    let x1 = x1
        .iter()
        .map(|x| {
            x.iter()
                .enumerate()
                .map(|(i, val)| {
                    (val - mean_x1[i])
                        / if std_x1[i].abs() < f64::EPSILON {
                            1.0
                        } else {
                            std_x1[i]
                        }
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let x2 = if let Some(x2) = x2 {
        let mean_x2 = (0..catch22::N_CATCH22)
            .map(|i| {
                let sum = x2.iter().map(|x| x[i]).sum::<f64>();
                sum / x2.len() as f64
            })
            .collect::<Vec<f64>>();
        let std_x2 = (0..catch22::N_CATCH22)
            .map(|i| {
                let sum = x2.iter().map(|x| (x[i] - mean_x2[i]).powi(2)).sum::<f64>();
                (sum / x2.len() as f64).sqrt()
            })
            .collect::<Vec<f64>>();
        Some(
            x2.iter()
                .map(|x| {
                    x.iter()
                        .enumerate()
                        .map(|(i, val)| {
                            (val - mean_x2[i])
                                / if std_x2[i].abs() < f64::EPSILON {
                                    1.0
                                } else {
                                    std_x2[i]
                                }
                        })
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>(),
        )
    } else {
        None
    };
    euclidean(x1, x2, par)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, gap_penalty=0.0, par=true, device="cpu"))]
pub fn erp(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: Option<f64>,
    gap_penalty: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let sakoe_chiba_band = sakoe_chiba_band.unwrap_or(1.0);
    let gap_penalty = gap_penalty.unwrap_or(0.0);
    if gap_penalty < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Gap penalty must be non-negative",
        ));
    }
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;
    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        let erp_cost_func =
                            |a: &[f64], b: &[f64], i: usize, j: usize, x: f64, y: f64, z: f64| {
                                min(
                                    min(y + (a[i] - b[j]).abs(), z + (a[i] - gap_penalty).abs()),
                                    x + (b[j] - gap_penalty).abs(),
                                )
                            };
                        diagonal::diagonal_distance::<WavefrontMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            erp_cost_func,
                            erp_cost_func,
                            true,
                        )
                    },
                    x1,
                    x2,
                    par.unwrap_or(true),
                ));
            }
            "gpu" => {
                distance_matrix = Some(compute_distance_gpu(
                    |a, b| {
                        let (gpu_device, queue, sba, sda, ma) = get_device();
                        tsdistances_gpu::cpu::erp(
                            gpu_device.clone(),
                            queue.clone(),
                            sba.clone(),
                            sda.clone(),
                            ma.clone(),
                            a,
                            b,
                            gap_penalty as f32,
                        )
                    },
                    x1,
                    x2,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }
    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing ERP distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, epsilon=1.0, par=true, device="cpu"))]
pub fn lcss(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: Option<f64>,
    epsilon: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let sakoe_chiba_band = sakoe_chiba_band.unwrap_or(1.0);
    let epsilon = epsilon.unwrap_or(1.0);
    if epsilon < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Epsilon must be non-negative",
        ));
    }
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;

    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        let lcss_cost_func =
                            |a: &[f64], b: &[f64], i: usize, j: usize, x: f64, y: f64, z: f64| {
                                let dist = (a[i] - b[j]).abs();
                                (dist <= epsilon) as i32 as f64 * (y + 1.0)
                                    + (dist > epsilon) as i32 as f64 * max(x, z)
                            };
                        let max_len = max(a.len(), b.len()) as f64;
                        let similarity = diagonal::diagonal_distance::<WavefrontMatrix>(
                            a,
                            b,
                            0.0,
                            sakoe_chiba_band,
                            lcss_cost_func,
                            lcss_cost_func,
                            false,
                        );
                        1.0 - similarity / max_len
                    },
                    x1,
                    x2,
                    par.unwrap_or(true),
                ));
            }
            "gpu" => {
                distance_matrix = Some(compute_distance_gpu(
                    |a, b| {
                        let (gpu_device, queue, sba, sda, ma) = get_device();
                        tsdistances_gpu::cpu::lcss(
                            gpu_device.clone(),
                            queue.clone(),
                            sba.clone(),
                            sda.clone(),
                            ma.clone(),
                            a,
                            b,
                            epsilon as f32,
                        )
                    },
                    x1,
                    x2,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }
    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing LCSS distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, par=true, device="cpu"))]
pub fn dtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let sakoe_chiba_band = sakoe_chiba_band.unwrap_or(1.0);
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;

    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        let dtw_cost_func =
                            |a: &[f64], b: &[f64], i: usize, j: usize, x: f64, y: f64, z: f64| {
                                let dist = (a[i] - b[j]).powi(2);
                                dist + min(min(z, x), y)
                            };
                        diagonal::diagonal_distance::<WavefrontMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            dtw_cost_func,
                            dtw_cost_func,
                            true,
                        )
                    },
                    x1,
                    x2,
                    par.unwrap_or(true),
                ));
            }
            "gpu" => {
                distance_matrix = Some(compute_distance_gpu(
                    |a, b| {
                        let (gpu_device, queue, sba, sda, ma) = get_device();
                        tsdistances_gpu::cpu::dtw(
                            gpu_device.clone(),
                            queue.clone(),
                            sba.clone(),
                            sda.clone(),
                            ma.clone(),
                            a,
                            b,
                        )
                    },
                    x1,
                    x2,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }

    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing DTW distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, par=true, device="cpu"))]
pub fn ddtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let x1_d = derivate(&x1);
    let x2_d = if let Some(x2) = &x2 {
        Some(derivate(&x2))
    } else {
        None
    };
    dtw(x1_d, x2_d, sakoe_chiba_band, par, device)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, g=0.05, par=true, device="cpu"))]
pub fn wdtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: Option<f64>,
    g: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let sakoe_chiba_band = sakoe_chiba_band.unwrap_or(1.0);
    let g = g.unwrap_or(0.05);
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;

    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        let weights = dtw_weights(a.len().max(b.len()), g);

                        let wdtw_cost_func =
                            |a: &[f64], b: &[f64], i: usize, j: usize, x: f64, y: f64, z: f64| {
                                let dist = (a[i] - b[j]).powi(2)
                                    * weights[(i as i32 - j as i32).abs() as usize];
                                dist + min(min(z, x), y)
                            };

                        diagonal::diagonal_distance::<WavefrontMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            wdtw_cost_func,
                            wdtw_cost_func,
                            true,
                        )
                    },
                    x1,
                    x2,
                    par.unwrap_or(true),
                ));
            }
            "gpu" => {
                distance_matrix = Some(compute_distance_gpu(
                    |a, b| {
                        let weights =
                            dtw_weights(max(a.first().unwrap().len(), b.first().unwrap().len()), g);
                        let (gpu_device, queue, sba, sda, ma) = get_device();
                        tsdistances_gpu::cpu::wdtw(
                            gpu_device.clone(),
                            queue.clone(),
                            sba.clone(),
                            sda.clone(),
                            ma.clone(),
                            a,
                            b,
                            &weights.iter().map(|x| *x as f32).collect::<Vec<_>>(),
                        )
                    },
                    x1,
                    x2,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }

    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing WDTW distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, g=0.05, par=true, device="cpu"))]
pub fn wddtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: Option<f64>,
    g: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let x1_d = derivate(&x1);
    let x2_d = if let Some(x2) = &x2 {
        Some(derivate(&x2))
    } else {
        None
    };
    wdtw(x1_d, x2_d, sakoe_chiba_band, g, par, device)
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, par=true, device="cpu"))]
pub fn msm(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let sakoe_chiba_band = sakoe_chiba_band.unwrap_or(1.0);
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;

    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        let msm_cost_func =
                            |a: &[f64], b: &[f64], i: usize, j: usize, x: f64, y: f64, z: f64| {
                                let a_i = a[i];
                                let b_j = b[j];
                                let a_prev = if likely(i != 0) { a[i - 1] } else { 0.0 };
                                let b_prev = if likely(j != 0) { b[j - 1] } else { 0.0 };

                                min(
                                    min(
                                        y + (a_i - b_j).abs(),
                                        z + msm_cost_function(a_i, a_prev, b_j),
                                    ),
                                    x + msm_cost_function(b_j, a_i, b_prev),
                                )
                            };

                        diagonal::diagonal_distance::<WavefrontMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            msm_cost_func,
                            msm_cost_func,
                            true,
                        )
                    },
                    x1,
                    x2,
                    par.unwrap_or(true),
                ));
            }
            "gpu" => {
                distance_matrix = Some(compute_distance_gpu(
                    |a, b| {
                        let (gpu_device, queue, sba, sda, ma) = get_device();
                        tsdistances_gpu::cpu::msm(
                            gpu_device.clone(),
                            queue.clone(),
                            sba.clone(),
                            sda.clone(),
                            ma.clone(),
                            a,
                            b,
                        )
                    },
                    x1,
                    x2,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }

    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing MSM distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, stiffness=0.001, penalty=1.0, par=true, device="cpu"))]
pub fn twe(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: Option<f64>,
    stiffness: Option<f64>,
    penalty: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let sakoe_chiba_band = sakoe_chiba_band.unwrap_or(1.0);
    let stiffness = stiffness.unwrap_or(0.001);
    let penalty = penalty.unwrap_or(1.0);
    if stiffness < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Stiffness (nu) must be non-negative",
        ));
    }
    if penalty < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Penalty (lambda) must be non-negative",
        ));
    }
    let delete_addition = stiffness + penalty;

    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;

    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        let twe_cost_func =
                            |a: &[f64], b: &[f64], i: usize, j: usize, x: f64, y: f64, z: f64| {
                                let a_i = a[i];
                                let b_j = b[j];
                                let a_prev = if likely(i != 0) { a[i - 1] } else { 0.0 };
                                let b_prev = if likely(j != 0) { b[j - 1] } else { 0.0 };
                                // deletion in a
                                let del_a: f64 = z + (a_prev - a_i).abs() + delete_addition;

                                // deletion in b
                                let del_b = x + (b_prev - b_j).abs() + delete_addition;

                                // match
                                let match_current = (a_i - b_j).abs();
                                let match_previous = (a_prev - b_prev).abs();
                                let match_a_b = y
                                    + match_current
                                    + match_previous
                                    + stiffness * (2.0 * (i as isize - j as isize).abs() as f64);

                                min(min(del_a, del_b), match_a_b)
                            };

                        diagonal::diagonal_distance::<WavefrontMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            twe_cost_func,
                            twe_cost_func,
                            true,
                        )
                    },
                    x1,
                    x2,
                    par.unwrap_or(true),
                ));
            }
            "gpu" => {
                distance_matrix = Some(compute_distance_gpu(
                    |a, b| {
                        let (gpu_device, queue, sba, sda, ma) = get_device();
                        tsdistances_gpu::cpu::twe(
                            gpu_device.clone(),
                            queue.clone(),
                            sba.clone(),
                            sda.clone(),
                            ma.clone(),
                            a,
                            b,
                            stiffness as f32,
                            penalty as f32,
                        )
                    },
                    x1,
                    x2,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }

    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing TWE distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, sakoe_chiba_band=1.0, warp_penalty=1.0, par=true, device="cpu"))]
pub fn adtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: Option<f64>,
    warp_penalty: Option<f64>,
    par: Option<bool>,
    device: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let sakoe_chiba_band = sakoe_chiba_band.unwrap_or(1.0);
    let warp_penalty = warp_penalty.unwrap_or(0.1);
    if warp_penalty < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Weight must be non-negative",
        ));
    }
    if sakoe_chiba_band < 0.0 || sakoe_chiba_band > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sakoe-Chiba band must be non-negative and less than 1.0",
        ));
    }
    let mut distance_matrix = None;
    if let Some(device) = device {
        match device {
            "cpu" => {
                distance_matrix = Some(compute_distance(
                    |a, b| {
                        let adtw_cost_func =
                            |a: &[f64], b: &[f64], i: usize, j: usize, x: f64, y: f64, z: f64| {
                                let dist = (a[i] - b[j]).powi(2);
                                dist + min(min(z + warp_penalty, x + warp_penalty), y)
                            };

                        diagonal::diagonal_distance::<WavefrontMatrix>(
                            a,
                            b,
                            f64::INFINITY,
                            sakoe_chiba_band,
                            adtw_cost_func,
                            adtw_cost_func,
                            true,
                        )
                    },
                    x1,
                    x2,
                    par.unwrap_or(true),
                ));
            }
            "gpu" => {
                distance_matrix = Some(compute_distance_gpu(
                    |a, b| {
                        let (gpu_device, queue, sba, sda, ma) = get_device();
                        tsdistances_gpu::cpu::adtw(
                            gpu_device.clone(),
                            queue.clone(),
                            sba.clone(),
                            sda.clone(),
                            ma.clone(),
                            a,
                            b,
                            warp_penalty as f32,
                        )
                    },
                    x1,
                    x2,
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device must be either 'cpu' or 'gpu'",
                ));
            }
        }
    }

    if let Some(distance_matrix) = distance_matrix {
        return Ok(distance_matrix);
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Error computing ADTW distance",
        ));
    }
}

#[pyfunction]
#[pyo3(signature = (x1, x2=None, par=true))]
pub fn sb(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    par: Option<bool>,
) -> PyResult<Vec<Vec<f64>>> {
    let distance_matrix = compute_distance(
        |a, b| {
            let a = zscore(&a);
            let b = zscore(&b);
            let cc = cross_correlation(&a, &b);
            1.0 - cc.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
                / (l2_norm(&a) * l2_norm(&b))
        },
        x1,
        x2,
        par.unwrap_or(true),
    );
    Ok(distance_matrix)
}

#[pyfunction]
#[pyo3(signature = (x1, window=20, x2=None, par=true))]
pub fn mp(
    x1: Vec<Vec<f64>>,
    window: Option<i32>,
    x2: Option<Vec<Vec<f64>>>,
    par: Option<bool>,
) -> PyResult<Vec<Vec<f64>>> {
    let threshold = 0.05;
    let window = window.unwrap_or(20) as usize;
    let distance_matrix = compute_distance(
        |a, b| {
            let n_a = a.len();
            let n_b = b.len();
            let mut p_abba = mp_(&a, &b, window as usize);
            let n = min(
                (threshold * (n_a + n_b) as f64).ceil() as usize,
                n_a - window + 1 + n_b - window + 1 - 1,
            );
            *p_abba
                .select_nth_unstable_by(n, |x, y| x.partial_cmp(y).unwrap())
                .1
        },
        x1,
        x2,
        par.unwrap_or(true),
    );
    Ok(distance_matrix)
}

fn mp_(a: &[f64], b: &[f64], window: usize) -> Vec<f64> {
    let n_a = a.len();
    let n_b = b.len();

    let window = window.min(n_a).min(n_b);

    let mut p_ab = vec![f64::INFINITY; n_a - window + 1];
    let mut p_ba = vec![f64::INFINITY; n_b - window + 1];

    let (mean_a, std_a) = mean_std_per_windows(&a, window);
    let (mean_b, std_b) = mean_std_per_windows(&b, window);

    for (i, sw_a) in a.windows(window).enumerate() {
        for (j, sw_b) in b.windows(window).enumerate() {
            let mut dist = 0.0;
            for (x, y) in sw_a.iter().zip(sw_b.iter()) {
                dist += (((x - mean_a[i]) / std_a[i]) - ((y - mean_b[j]) / std_b[j])).powi(2);
            }
            dist = dist.sqrt();
            p_ab[i] = p_ab[i].min(dist);
            p_ba[j] = p_ba[j].min(dist);
        }
    }

    if p_ab.len() > p_ba.len() {
        p_ab.extend(p_ba);
        p_ab
    } else {
        p_ba.extend(p_ab);
        p_ba
    }
}

fn mean_std_per_windows(a: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let n = a.len();

    let mut means = Vec::with_capacity(n - window + 1);
    let mut stds = Vec::with_capacity(n - window + 1);

    let mut sum: f64 = a[0..window].iter().sum();
    let mut sum_squares: f64 = a[0..window].iter().map(|&x| x * x).sum();

    means.push(sum / window as f64);
    let var = (sum_squares / window as f64) - (means[0] * means[0]);
    stds.push(var.sqrt());

    for i in window..n {
        sum += a[i] - a[i - window];
        sum_squares += a[i] * a[i] - a[i - window] * a[i - window];

        let mean = sum / window as f64;
        means.push(mean);

        let var = (sum_squares / window as f64) - (mean * mean);
        stds.push(var.sqrt());
    }

    (means, stds)
}

#[inline]
#[cold]
fn cold() {}

#[inline]
fn likely(b: bool) -> bool {
    if !b {
        cold()
    }
    b
}
