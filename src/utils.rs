use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{cell::RefCell, collections::HashMap, sync::Arc};

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

    FFT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let (fft, ifft) = cache.get_plans(fft_len);
        cache.ensure_len(fft_len);

        cache.a_fft.fill(Complex::new(0.0, 0.0));
        cache.b_fft.fill(Complex::new(0.0, 0.0));
        for (i, val) in a.iter().enumerate() {
            cache.a_fft[i].re = *val;
        }
        for (i, val) in b.iter().enumerate() {
            cache.b_fft[i].re = *val;
        }

        fft.process(&mut cache.a_fft);
        fft.process(&mut cache.b_fft);

        for i in 0..fft_len {
            cache.c_fft[i] = cache.a_fft[i].conj() * cache.b_fft[i];
        }

        ifft.process(&mut cache.c_fft);
        for i in 0..fft_len {
            cache.c[i] = cache.c_fft[i].re / fft_len as f64;
        }
        cache.c.clone()
    })
}

struct FftCache {
    planner: FftPlanner<f64>,
    plans: HashMap<usize, (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>)>,
    a_fft: Vec<Complex<f64>>,
    b_fft: Vec<Complex<f64>>,
    c_fft: Vec<Complex<f64>>,
    c: Vec<f64>,
}

impl FftCache {
    fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
            plans: HashMap::new(),
            a_fft: Vec::new(),
            b_fft: Vec::new(),
            c_fft: Vec::new(),
            c: Vec::new(),
        }
    }

    fn get_plans(&mut self, len: usize) -> (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>) {
        if let Some((fft, ifft)) = self.plans.get(&len) {
            return (fft.clone(), ifft.clone());
        }
        let fft = self.planner.plan_fft_forward(len);
        let ifft = self.planner.plan_fft_inverse(len);
        self.plans.insert(len, (fft.clone(), ifft.clone()));
        (fft, ifft)
    }

    fn ensure_len(&mut self, len: usize) {
        if self.a_fft.len() != len {
            self.a_fft.resize(len, Complex::new(0.0, 0.0));
            self.b_fft.resize(len, Complex::new(0.0, 0.0));
            self.c_fft.resize(len, Complex::new(0.0, 0.0));
            self.c.resize(len, 0.0);
        }
    }
}

thread_local! {
    static FFT_CACHE: RefCell<FftCache> = RefCell::new(FftCache::new());
}

pub fn zscore(x: &[f64]) -> Vec<f64> {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let std = (x.iter().map(|val| (val - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
    x.iter().map(|val| (val - mean) / std).collect()
}

pub fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|val| val.powi(2)).sum::<f64>().sqrt()
}
