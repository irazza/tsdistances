//! Core distance computation functions without any binding dependencies.
//! These functions are used by both Python (PyO3) and MATLAB (C FFI) bindings.

use crate::{
    diagonal,
    matrix::WavefrontMatrix,
    utils::{
        cross_correlation, derivate, dtw_weights, l2_norm, max, min, msm_cost_function, zscore,
    },
};
use rayon::prelude::*;
use tsdistances_gpu::utils::get_device;

/// Error type for distance computation
#[derive(Debug, Clone)]
pub enum DistanceError {
    InvalidParameter(String),
    ComputationError(String),
}

impl std::fmt::Display for DistanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceError::InvalidParameter(msg) => write!(f, "Invalid parameter: {msg}"),
            DistanceError::ComputationError(msg) => write!(f, "Computation error: {msg}"),
        }
    }
}

impl std::error::Error for DistanceError {}

pub type Result<T> = std::result::Result<T, DistanceError>;

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
pub fn compute_distance(
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

/// Compute Euclidean distance matrix
pub fn euclidean(x1: Vec<Vec<f64>>, x2: Option<Vec<Vec<f64>>>, par: bool) -> Result<Vec<Vec<f64>>> {
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
        par,
    );
    Ok(distance_matrix)
}

/// Compute Catch22-Euclidean distance matrix
pub fn catch_euclidean(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    par: bool,
) -> Result<Vec<Vec<f64>>> {
    let x1 = x1
        .iter()
        .map(|x| {
            let mut transformed_x = Vec::with_capacity(catch22::N_CATCH22);
            for i in 0..catch22::N_CATCH22 {
                let value = catch22::compute(x, i);
                if value.is_nan() {
                    transformed_x.push(0.0);
                } else {
                    transformed_x.push(value);
                }
            }
            transformed_x
        })
        .collect::<Vec<Vec<_>>>();

    let x2 = x2.map(|x2| {
        x2.iter()
            .map(|x| {
                let mut transformed_x = Vec::with_capacity(catch22::N_CATCH22);
                for i in 0..catch22::N_CATCH22 {
                    let value = catch22::compute(x, i);
                    if value.is_finite() {
                        transformed_x.push(value);
                    } else {
                        transformed_x.push(0.0);
                    }
                }
                transformed_x
            })
            .collect::<Vec<Vec<_>>>()
    });

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

/// Compute ERP (Edit Distance with Real Penalty) distance matrix
pub fn erp(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    gap_penalty: f64,
    par: bool,
    device: &str,
) -> Result<Vec<Vec<f64>>> {
    if gap_penalty < 0.0 {
        return Err(DistanceError::InvalidParameter(
            "Gap penalty must be non-negative".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&sakoe_chiba_band) {
        return Err(DistanceError::InvalidParameter(
            "Sakoe-Chiba band must be between 0.0 and 1.0".to_string(),
        ));
    }

    match device {
        "cpu" => {
            let distance_matrix = compute_distance(
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
                par,
            );
            Ok(distance_matrix)
        }
        "gpu" => {
            let distance_matrix = compute_distance_gpu(
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
            );
            Ok(distance_matrix)
        }
        _ => Err(DistanceError::InvalidParameter(
            "Device must be either 'cpu' or 'gpu'".to_string(),
        )),
    }
}

/// Compute LCSS (Longest Common Subsequence) distance matrix
pub fn lcss(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    epsilon: f64,
    par: bool,
    device: &str,
) -> Result<Vec<Vec<f64>>> {
    if epsilon < 0.0 {
        return Err(DistanceError::InvalidParameter(
            "Epsilon must be non-negative".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&sakoe_chiba_band) {
        return Err(DistanceError::InvalidParameter(
            "Sakoe-Chiba band must be between 0.0 and 1.0".to_string(),
        ));
    }

    match device {
        "cpu" => {
            let distance_matrix = compute_distance(
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
                par,
            );
            Ok(distance_matrix)
        }
        "gpu" => {
            let distance_matrix = compute_distance_gpu(
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
            );
            Ok(distance_matrix)
        }
        _ => Err(DistanceError::InvalidParameter(
            "Device must be either 'cpu' or 'gpu'".to_string(),
        )),
    }
}

/// Compute DTW (Dynamic Time Warping) distance matrix
pub fn dtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    par: bool,
    device: &str,
) -> Result<Vec<Vec<f64>>> {
    if !(0.0..=1.0).contains(&sakoe_chiba_band) {
        return Err(DistanceError::InvalidParameter(
            "Sakoe-Chiba band must be between 0.0 and 1.0".to_string(),
        ));
    }

    match device {
        "cpu" => {
            let distance_matrix = compute_distance(
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
                par,
            );
            Ok(distance_matrix)
        }
        "gpu" => {
            let distance_matrix = compute_distance_gpu(
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
            );
            Ok(distance_matrix)
        }
        _ => Err(DistanceError::InvalidParameter(
            "Device must be either 'cpu' or 'gpu'".to_string(),
        )),
    }
}

/// Compute DDTW (Derivative Dynamic Time Warping) distance matrix
pub fn ddtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    par: bool,
    device: &str,
) -> Result<Vec<Vec<f64>>> {
    let x1_d = derivate(&x1);
    let x2_d = x2.as_ref().map(|x| derivate(x));
    dtw(x1_d, x2_d, sakoe_chiba_band, par, device)
}

/// Compute WDTW (Weighted Dynamic Time Warping) distance matrix
pub fn wdtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    g: f64,
    par: bool,
    device: &str,
) -> Result<Vec<Vec<f64>>> {
    if !(0.0..=1.0).contains(&sakoe_chiba_band) {
        return Err(DistanceError::InvalidParameter(
            "Sakoe-Chiba band must be between 0.0 and 1.0".to_string(),
        ));
    }

    match device {
        "cpu" => {
            let distance_matrix = compute_distance(
                |a, b| {
                    let weights = dtw_weights(a.len().max(b.len()), g);

                    let wdtw_cost_func =
                        |a: &[f64], b: &[f64], i: usize, j: usize, x: f64, y: f64, z: f64| {
                            let dist = (a[i] - b[j]).powi(2)
                                * weights[(i as i32 - j as i32).unsigned_abs() as usize];
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
                par,
            );
            Ok(distance_matrix)
        }
        "gpu" => {
            let distance_matrix = compute_distance_gpu(
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
            );
            Ok(distance_matrix)
        }
        _ => Err(DistanceError::InvalidParameter(
            "Device must be either 'cpu' or 'gpu'".to_string(),
        )),
    }
}

/// Compute WDDTW (Weighted Derivative Dynamic Time Warping) distance matrix
pub fn wddtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    g: f64,
    par: bool,
    device: &str,
) -> Result<Vec<Vec<f64>>> {
    let x1_d = derivate(&x1);
    let x2_d = x2.as_ref().map(|x| derivate(x));
    wdtw(x1_d, x2_d, sakoe_chiba_band, g, par, device)
}

/// Compute ADTW (Amerced Dynamic Time Warping) distance matrix
pub fn adtw(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    warp_penalty: f64,
    par: bool,
    device: &str,
) -> Result<Vec<Vec<f64>>> {
    if warp_penalty < 0.0 {
        return Err(DistanceError::InvalidParameter(
            "Warp penalty must be non-negative".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&sakoe_chiba_band) {
        return Err(DistanceError::InvalidParameter(
            "Sakoe-Chiba band must be between 0.0 and 1.0".to_string(),
        ));
    }

    match device {
        "cpu" => {
            let distance_matrix = compute_distance(
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
                par,
            );
            Ok(distance_matrix)
        }
        "gpu" => {
            let distance_matrix = compute_distance_gpu(
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
            );
            Ok(distance_matrix)
        }
        _ => Err(DistanceError::InvalidParameter(
            "Device must be either 'cpu' or 'gpu'".to_string(),
        )),
    }
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

/// Compute MSM (Move-Split-Merge) distance matrix
pub fn msm(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    par: bool,
    device: &str,
) -> Result<Vec<Vec<f64>>> {
    if !(0.0..=1.0).contains(&sakoe_chiba_band) {
        return Err(DistanceError::InvalidParameter(
            "Sakoe-Chiba band must be between 0.0 and 1.0".to_string(),
        ));
    }

    match device {
        "cpu" => {
            let distance_matrix = compute_distance(
                |a, b| {
                    let msm_cost_func_inner =
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
                        msm_cost_func_inner,
                        msm_cost_func_inner,
                        true,
                    )
                },
                x1,
                x2,
                par,
            );
            Ok(distance_matrix)
        }
        "gpu" => {
            let distance_matrix = compute_distance_gpu(
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
            );
            Ok(distance_matrix)
        }
        _ => Err(DistanceError::InvalidParameter(
            "Device must be either 'cpu' or 'gpu'".to_string(),
        )),
    }
}

/// Compute TWE (Time Warp Edit) distance matrix
pub fn twe(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    sakoe_chiba_band: f64,
    stiffness: f64,
    penalty: f64,
    par: bool,
    device: &str,
) -> Result<Vec<Vec<f64>>> {
    if stiffness < 0.0 {
        return Err(DistanceError::InvalidParameter(
            "Stiffness (nu) must be non-negative".to_string(),
        ));
    }
    if penalty < 0.0 {
        return Err(DistanceError::InvalidParameter(
            "Penalty (lambda) must be non-negative".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&sakoe_chiba_band) {
        return Err(DistanceError::InvalidParameter(
            "Sakoe-Chiba band must be between 0.0 and 1.0".to_string(),
        ));
    }

    let delete_addition = stiffness + penalty;

    match device {
        "cpu" => {
            let distance_matrix = compute_distance(
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
                par,
            );
            Ok(distance_matrix)
        }
        "gpu" => {
            let distance_matrix = compute_distance_gpu(
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
            );
            Ok(distance_matrix)
        }
        _ => Err(DistanceError::InvalidParameter(
            "Device must be either 'cpu' or 'gpu'".to_string(),
        )),
    }
}

/// Compute SBD (Shape-Based Distance) distance matrix
pub fn sbd(x1: Vec<Vec<f64>>, x2: Option<Vec<Vec<f64>>>, par: bool) -> Result<Vec<Vec<f64>>> {
    let distance_matrix = compute_distance(
        |a, b| {
            let a = zscore(a);
            let b = zscore(b);
            let cc = cross_correlation(&a, &b);
            1.0 - cc.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
                / (l2_norm(&a) * l2_norm(&b))
        },
        x1,
        x2,
        par,
    );
    Ok(distance_matrix)
}

/// Compute MP (Matrix Profile) distance matrix
pub fn mp(
    x1: Vec<Vec<f64>>,
    x2: Option<Vec<Vec<f64>>>,
    window: i32,
    par: bool,
) -> Result<Vec<Vec<f64>>> {
    let threshold = 0.05;
    let window = window as usize;
    let distance_matrix = compute_distance(
        |a, b| {
            let n_a = a.len();
            let n_b = b.len();
            let mut p_abba = mp_inner(a, b, window);
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
        par,
    );
    Ok(distance_matrix)
}

fn mp_inner(a: &[f64], b: &[f64], window: usize) -> Vec<f64> {
    let n_a = a.len();
    let n_b = b.len();

    let window = window.min(n_a).min(n_b);

    let mut p_ab = vec![f64::INFINITY; n_a - window + 1];
    let mut p_ba = vec![f64::INFINITY; n_b - window + 1];

    let (mean_a, std_a) = mean_std_per_windows(a, window);
    let (mean_b, std_b) = mean_std_per_windows(b, window);

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
