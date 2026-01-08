# tsdistances

## Introduction

`tsdistances` is a Python library (with Rust backend) for computing various pairwise distances between sets of time series data. 

It provides eﬀicient implementation of elastic distance measures such as Dynamic Time Warping (DTW), Longest Common Subsequence (LCSS), Time Warping Edit (TWE), and many others.

The library is designed to be fast and scalable, leveraging parallel computation and GPU support via Vulkan for improved performance.

### Features

1.  Multiple Distance Measures: Supports a wide range of time series distance measures:

    -   Euclidean

    -   CATCH22 Euclidean

    -   Edit Distance with Real Penalty (ERP) optionally with GPU support

    -   Longest Common Subsequence (LCSS) optionally with GPU support

    -   Dynamic Time Warping (DTW) optionally with GPU support

    -   Derivative Dynamic Time Warping (DDTW) optionally with GPU support

    -   Weighted Dynamic Time Warping (WDTW) optionally with GPU support

    -   Weighted Derivative Dynamic Time Warping (WDDTW) optionally with GPU support

    -   Amerced Dynamic Time Warping (ADTW) optionally with GPU support

    -   Move-Split-Merge (MSM) optionally with GPU support

    -   Time Warp Edit Distance (TWE) optionally with GPU support

    -   Shape-Based Distance (SBD)

    -   MPDist

2.  Parallel Computation: Utilizes multiple CPU cores to speed up computations.

3.  GPU Acceleration: Optional GPU support based on [Vulkan](https://www.vulkan.org/) for even faster computations with [Rust-GPU](https://rust-gpu.github.io/).

## Benchmark

To evaluate the performance of our time series distance computation library, we conducted a comparative analysis with existing libraries. 

We selected [AEON](https://github.com/aeon-toolkit/aeon) as the primary competitor due to its comprehensive implementation of distance metrics, making it the most suitable for direct comparison. Several other libraries were considered, and while we did not conduct a full benchmark on all datasets, we reported their execution times on a subset of the [UCR Archive datasets](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

|                             | sthread |    par |    gpu |
|-----------------------------|--------:|-------:|-------:|
| ACSF1                       |   31.60 |   5.74 |   1.41 |
| Adiac                       |    5.27 |   0.84 |   1.02 |
| Beef                        |    0.47 |   0.08 |   0.09 |
| CBF                         |    1.51 |   0.25 |   0.28 |
| ChlorineConcentration       |  131.53 |  22.73 |   7.36 |
| CinCECGTorso                |  524.77 |  86.92 |   7.85 |
| CricketX                    |   47.93 |   8.35 |   1.82 |
| DiatomSizeReduction         |    0.59 |   0.09 |   0.24 |
| DistalPhalanxOutlineCorrect |    2.45 |   0.37 |   0.57 |
| ECG200                      |    0.28 |   0.04 |   0.05 |
| EthanolLevel                |  925.06 | 169.63 |  38.18 |
| FreezerRegularTrain         |  100.48 |  16.94 |   4.69 |
| FreezerSmallTrain           |   18.54 |   3.18 |   1.11 |
| Ham                         |    5.43 |   0.93 |   0.45 |
| Haptics                     |  149.35 |  27.49 |   3.82 |
| HouseTwenty                 |   65.37 |  11.64 |   1.20 |
| ItalyPowerDemand            |    0.16 |   0.03 |   0.12 |
| MixedShapesSmallTrain       |  804.75 | 150.01 |  16.66 |
| NonInvasiveFetalECGThorax1  | 3398.64 | 724.58 | 107.42 |
| ShapesAll                   |  312.02 |  54.83 |   6.90 |
| Strawberry                  |   19.38 |   3.42 |   2.35 |
| UWaveGestureLibraryX        | 1016.30 | 196.54 |  21.05 |
| Wafer                       |  462.68 |  81.98 |  16.08 |

 Computation times (in seconds) of our method across 23 datasets, comparing single-threaded, parallelized, and GPU implementations.

## Installation
### PIP

If you use pip, you can install tsdistances with:
```bash
    $ pip install tsdistances
```

### From Source

This can be done by going through the following steps in sequence:

1. Install the latest [Rust compiler](https://www.rust-lang.org/tools/install)
2. Install [maturin](https://maturin.rs/): `pip install maturin`
3. Build the library:
   ```bash
   maturin develop --release
   ```

#### GPU Support

To build with GPU acceleration:

1. Install [LunarG Vulkan SDK](https://www.lunarg.com/vulkan-sdk/)
2. Either:
   - Install [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools.git), or
   - Use pre-compiled tools with `--features use-compiled-tools`

```bash
maturin develop --release --features use-compiled-tools
```

#### Feature Flags

The library uses Cargo feature flags to control what gets compiled:

| Feature | Description | Default |
|---------|-------------|---------|
| `python` | Python bindings via PyO3 | ✓ |
| `matlab` | MATLAB/C FFI bindings | ✗ |
| `use-compiled-tools` | Use pre-compiled SPIRV tools for GPU | ✓ |
| `use-installed-tools` | Use system-installed SPIRV tools | ✗ |

For Python development (default):
```bash
maturin develop --release
# or explicitly:
cargo build --release --features python,use-compiled-tools
```

For MATLAB bindings only (no Python dependency):
```bash
cargo build --release --no-default-features --features matlab,use-compiled-tools
```

See [matlab/README.md](matlab/README.md) for detailed MATLAB installation instructions. 

## Usage

### Example 1: Compute DTW Distance on CPU and GPU
```python
        
    import numpy as np
    import tsdistances

    # Generate two random time series (1-D arrays of length 100)
    np.random.seed(0)
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)

    # Compute DTW distance on CPU
    cpu_distance = tsdistances.dtw_distance(x1, x2, device='cpu')
    print(f"DTW distance (CPU): {cpu_distance}")

    gpu_distance = tsdistances.dtw_distance(x1, x2, device='gpu')

    print(f"DTW distance (GPU): {gpu_distance}")
```

### Example 2: Pairwise Distances with Multiple Time Series and Parallel Computation
```python
    import numpy as np
    import tsdistances

    # Generate a batch of 10 random time series (each of length 50)
    np.random.seed(42)
    X = np.random.rand(10, 50)

    # Pairwise DTW distances within the set X (on CPU, single thread)
    pairwise_distances = tsdistances.dtw_distance(X, par=False, device='cpu')
    print("Pairwise DTW distance matrix (CPU, 4 jobs):")
    print(pairwise_distances)

    # Compare two batches: compute distances between each element of X and each element of Y
    Y = np.random.rand(8, 50)
    batch_distances = tsdistances.dtw_distance(X, Y, par=True, device='cpu')
    print("Batch DTW distance matrix (X vs Y):")
    print(batch_distances)
```
Notes
1. `device='gpu'` enables GPU acceleration.

2. `par` controls parallelism. Set it to `True` to use all available CPU cores.

3. If `v` is not provided, the function computes pairwise distances within `u`.

Important: Results will differ between CPU and GPU due to f64ing-point precision:

    CPU computations use f64 (double precision) for higher numerical accuracy.

    GPU computations use f32 (single precision) for better performance.
    For instance, on an RTX 4090:

        FP32 performance: 82.58 TFLOPS

        FP64 performance: 1.29 TFLOPS (1:64 rate)
        Using f32 on GPU drastically improves speed but introduces small numerical differences compared to CPU results.

## Testing and Validation

All distance implementations in `tsdistances` are **tested against [AEON](https://github.com/aeon-toolkit/aeon)**, a widely-used Python library for time series analysis and distances. This ensures that the results are correct and consistent with established benchmarks in the field.

To run the correctness tests, simply use `pytest`:

```bash
pytest -v tests/test_correctness_cpu.py
```

