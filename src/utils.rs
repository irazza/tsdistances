use rustfft::{Fft, FftDirection, algorithm::Radix4, num_complex::Complex};

pub fn min<T: PartialOrd>(x: T, y: T) -> T {
    if x < y { x } else { y }
}
pub fn max<T: PartialOrd>(x: T, y: T) -> T {
    if x > y { x } else { y }
}
#[allow(dead_code)]
pub fn next_multiple_of_n(x: usize, n: usize) -> usize {
    x.div_ceil(n) * n
}

pub fn derivate(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut x_d = Vec::with_capacity(x.len());
    for item in x {
        x_d.push(vec![0.0; item.len() - 2]);
    }
    for (idx, item) in x.iter().enumerate() {
        for j in 1..item.len() - 1 {
            x_d[idx][j - 1] = ((item[j] - item[j - 1]) + (item[j + 1] - item[j - 1]) / 2.0) / 2.0;
        }
    }
    x_d
}

const WEIGHT_MAX: f64 = 1.0;
pub fn dtw_weights(len: usize, g: f64) -> Vec<f64> {
    let mut weights = vec![0.0; len];
    let half_len = len as f64 / 2.0;
    for (i, weight) in weights.iter_mut().enumerate().take(len) {
        *weight = WEIGHT_MAX / (1.0 + std::f64::consts::E.powf(-g * (i as f64 - half_len)));
    }
    weights
}
// [1 / (1 + np.exp(-g * (i - max_size / 2))) for i in range(0, max_size)]

const MSM_C: f64 = 1.0;
#[inline(always)]
pub fn msm_cost_function(x: f64, y: f64, z: f64) -> f64 {
    MSM_C + (y.min(z) - x).max(x - y.max(z)).max(0.0)
}

pub fn cross_correlation(a: &[f64], b: &[f64]) -> Vec<f64> {
    // zero-pad the input signals a and b (add zeros to the end of each. The zero padding should fill the vectors until they reach a size of at least N = size(a)+size(b)-1
    let fft_len = (a.len() + b.len() - 1).next_power_of_two();
    let fft = Radix4::new(fft_len, FftDirection::Forward);

    let mut a_fft = vec![Complex::new(0.0, 0.0); fft_len];
    let mut b_fft = vec![Complex::new(0.0, 0.0); fft_len];
    for (i, val) in a.iter().enumerate() {
        a_fft[i] = Complex::new(*val, 0.0);
    }
    for (i, val) in b.iter().enumerate() {
        b_fft[i] = Complex::new(*val, 0.0);
    }

    fft.process(&mut a_fft);
    fft.process(&mut b_fft);

    let mut c_fft = vec![Complex::new(0.0, 0.0); fft_len];
    for i in 0..fft_len {
        c_fft[i] = a_fft[i].conj() * b_fft[i];
    }

    let mut c = vec![0.0; fft_len];
    let ifft = Radix4::new(fft_len, FftDirection::Inverse);
    ifft.process(&mut c_fft);
    for i in 0..fft_len {
        c[i] = c_fft[i].re / fft_len as f64;
    }
    c
}

pub fn zscore(x: &[f64]) -> Vec<f64> {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let std = (x.iter().map(|val| (val - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
    x.iter().map(|val| (val - mean) / std).collect()
}

pub fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|val| val.powi(2)).sum::<f64>().sqrt()
}
