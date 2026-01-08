# TSDistances MATLAB Bindings

MATLAB bindings for the `tsdistances` Rust library, providing efficient computation of various time series distance measures.

## Available Distance Functions

| Function | Description | Reference |
|----------|-------------|-----------|
| `tsd_euclidean` | Euclidean distance | - |
| `tsd_catch_euclidean` | Catch22-Euclidean distance | - |
| `tsd_dtw` | Dynamic Time Warping | Berndt & Clifford, 1994 |
| `tsd_ddtw` | Derivative DTW | Keogh et al., 2001 |
| `tsd_wdtw` | Weighted DTW | Jeong et al., 2011 |
| `tsd_wddtw` | Weighted Derivative DTW | - |
| `tsd_adtw` | Amerced DTW | - |
| `tsd_erp` | Edit Distance with Real Penalty | Chen et al., 2004 |
| `tsd_lcss` | Longest Common Subsequence | Vlachos et al., 2002 |
| `tsd_msm` | Move-Split-Merge | Stefan et al., 2013 |
| `tsd_twe` | Time Warp Edit | Marteau, 2009 |
| `tsd_sbd` | Shape-Based Distance | Paparrizos & Gravano, 2015 |
| `tsd_mp` | Matrix Profile Distance | Yeh et al., 2016 |

## Requirements

- **MATLAB** R2018a or later (with MEX compiler configured)
- **Rust** toolchain (1.70+)
- **C Compiler** (Xcode Command Line Tools on macOS, GCC on Linux, MSVC on Windows)

## Installation

### Quick Install

1. Navigate to the `matlab` directory in MATLAB:
   ```matlab
   cd /path/to/tsdistances/matlab
   ```

2. Run the build script:
   ```matlab
   build_tsdistances
   ```

3. (Optional) Save the path permanently:
   ```matlab
   savepath
   ```

### Manual Installation

If the automatic build fails, you can build manually:

1. Build the Rust library (with MATLAB features, no Python dependencies):
   ```bash
   cd /path/to/tsdistances
   cargo build --release --no-default-features --features matlab,use-compiled-tools
   ```

2. Compile the MEX file (in MATLAB):
   ```matlab
   % On macOS/Linux with static library:
   mex -R2018a matlab/tsd_mex.c ./target/release/libtsdistances.a -Imatlab
   
   % On Windows:
   mex matlab/tsd_mex.c ./target/release/tsdistances.lib -Imatlab
   ```

## Usage

### Basic Usage

```matlab
% Generate random time series data
X = randn(100, 200);  % 100 time series, each of length 200

% Compute pairwise DTW distance matrix
D = tsd_dtw(X);

% Compute DTW between two sets
X1 = randn(50, 200);
X2 = randn(30, 200);
D = tsd_dtw(X1, X2);  % Returns 50x30 matrix
```

### DTW Variants

```matlab
X = randn(50, 100);

% Standard DTW
D = tsd_dtw(X);

% DTW with Sakoe-Chiba band constraint (10% of length)
D = tsd_dtw(X, [], 0.1);

% Derivative DTW
D = tsd_ddtw(X);

% Weighted DTW
D = tsd_wdtw(X, [], 1.0, 0.05);  % band=1.0, g=0.05

% Amerced DTW with warp penalty
D = tsd_adtw(X, [], 1.0, 0.5);  % band=1.0, warp_penalty=0.5
```

### Elastic Distance Measures

```matlab
X = randn(50, 100);

% Edit Distance with Real Penalty
D = tsd_erp(X, [], 1.0, 0.0);  % band=1.0, gap_penalty=0.0

% Longest Common Subsequence
D = tsd_lcss(X, [], 1.0, 0.5);  % band=1.0, epsilon=0.5

% Move-Split-Merge
D = tsd_msm(X, [], 1.0);  % cost=1.0

% Time Warp Edit Distance
D = tsd_twe(X, [], 0.5, 0.5);  % stiffness=0.5, penalty=0.5
```

### Other Distances

```matlab
X = randn(50, 100);

% Euclidean distance
D = tsd_euclidean(X);

% Catch22-based Euclidean distance
D = tsd_catch_euclidean(X);

% Shape-Based Distance (shift-invariant)
D = tsd_sbd(X);

% Matrix Profile Distance
D = tsd_mp(X, [], 25);  % window_size=25
```

### Parallel Computation

All functions support parallel computation (enabled by default):

```matlab
X = randn(100, 200);

% Parallel computation (default)
D = tsd_dtw(X, [], 1.0, true);

% Sequential computation
D = tsd_dtw(X, [], 1.0, false);
```

## Function Reference

### tsd_dtw

```matlab
D = tsd_dtw(X1, X2, band, parallel)
```

**Inputs:**
- `X1` - M×N matrix of M time series of length N
- `X2` - (optional) P×N matrix, or `[]` for pairwise within X1
- `band` - (optional) Sakoe-Chiba band size [0-1], default: 1.0
- `parallel` - (optional) enable parallel computation, default: true

**Output:**
- `D` - Distance matrix (M×M if X2 is empty, M×P otherwise)

### tsd_erp

```matlab
D = tsd_erp(X1, X2, band, gap_penalty, parallel)
```

**Inputs:**
- `X1` - M×N matrix of time series
- `X2` - (optional) P×N matrix
- `band` - (optional) Sakoe-Chiba band [0-1], default: 1.0
- `gap_penalty` - (optional) penalty for gaps, default: 0.0
- `parallel` - (optional) enable parallel, default: true

### tsd_lcss

```matlab
D = tsd_lcss(X1, X2, band, epsilon, parallel)
```

**Inputs:**
- `X1` - M×N matrix of time series
- `X2` - (optional) P×N matrix
- `band` - (optional) Sakoe-Chiba band [0-1], default: 1.0
- `epsilon` - (optional) matching threshold, default: 1.0
- `parallel` - (optional) enable parallel, default: true

## Performance Tips

1. **Use parallel computation** for large datasets (enabled by default)
2. **Use Sakoe-Chiba bands** to speed up DTW variants (e.g., `band=0.1` for 10%)
3. **Batch operations**: Compute full distance matrices rather than individual distances
4. **Data layout**: Each row should be one time series

## Troubleshooting

### MEX compilation fails

1. Ensure you have a C compiler configured:
   ```matlab
   mex -setup
   ```

2. On macOS, install Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```

### Library not found at runtime

On macOS/Linux, ensure the library path is set:
```matlab
setenv('DYLD_LIBRARY_PATH', '/path/to/tsdistances/target/release');
```

Or copy the library to the MATLAB directory (done automatically by `build_tsdistances`).

### Rust build fails

Ensure Rust is installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## License

This project is licensed under the same terms as the main tsdistances library.

## Citation

If you use this library in your research, please cite the relevant papers for each distance measure (see the table above).
