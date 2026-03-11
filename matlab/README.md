# tsdistances MATLAB Bindings

MATLAB bindings for the `tsdistances` Rust library, providing efficient computation of various time series distance measures.

**New!** This build system now includes automatic dependency checking and installation guidance.

## Quick Start

1. **Install Dependencies** (first time only):
   - **macOS/Linux**: Run `bash install_dependencies.sh` in terminal
   - **Windows**: Run `install_dependencies.bat` in Command Prompt
   - See [INSTALLATION_GUIDE.md](#documentation) for detailed instructions

2. **Build in MATLAB**:
   ```matlab
   cd /path/to/tsdistances/matlab
   build_tsdistances
   ```
   The script automatically checks dependencies and guides installation if needed.

3. **Verify**:
   ```matlab
   X = rand(50, 100);
   D = tsd_dtw(X);
   disp(size(D));  % Should be 50x50
   ```

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

### Required
- **MATLAB** R2018a or later (with MEX compiler configured)
- **Rust** toolchain (includes `cargo` and `rustc`)
- **C Compiler**:
  - macOS: Xcode Command Line Tools
  - Linux: GCC (via `build-essential`, etc.)
  - Windows: Microsoft Visual C++ (Visual Studio Build Tools)

### Optional
- **Vulkan SDK** for GPU acceleration (tsdistances_gpu)

## Documentation

This directory includes comprehensive setup documentation:

| Document | Purpose |
|----------|---------|
| [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) | Step-by-step setup for all platforms |
| [DEPENDENCIES.md](DEPENDENCIES.md) | Complete dependency reference and versions |
| [install_dependencies.sh](install_dependencies.sh) | Auto-installer for macOS/Linux |
| [install_dependencies.bat](install_dependencies.bat) | Checker script for Windows |
| `build_tsdistances.m` | Build script with auto-dependency checking |

**Start with [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for setup!**

## Installation by Platform

### macOS
```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Install Xcode Command Line Tools
xcode-select --install

# 3. Install pkg-config (recommended)
brew install pkg-config

# 4. Configure MATLAB MEX
# In MATLAB: mex -setup C

# 5. Build
cd /path/to/tsdistances/matlab
build_tsdistances
```

### Linux (Ubuntu/Debian)
```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Install build tools
sudo apt-get update
sudo apt-get install build-essential pkg-config libx11-dev libxrandr-dev

# 3. Configure MATLAB MEX
# In MATLAB: mex -setup C

# 4. Build
cd /path/to/tsdistances/matlab
build_tsdistances
```

### Windows
1. Download and install [Rust](https://www.rust-lang.org/tools/install)
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/) or Community Edition
3. In MATLAB: `mex -setup C` (select Microsoft Visual C++)
4. Run: `build_tsdistances`

## Usage Examples

### Basic DTW Distance

```matlab
% Load or generate time series data
X = randn(100, 200);  % 100 time series, each of length 200

% Compute all pairwise DTW distances
D = tsd_dtw(X);       % Returns 100×100 distance matrix
```

### Compare Different Distance Measures

```matlab
X = randn(50, 150);

% Compute various distances
D_euc = tsd_euclidean(X);      % Euclidean
D_dtw = tsd_dtw(X);            % Dynamic Time Warping
D_erp = tsd_erp(X);            % Edit Distance with Real Penalty
D_sbd = tsd_sbd(X);            % Shape-Based Distance

% Compare first 5x5 submatrix
fprintf('Euclidean:\n'); disp(D_euc(1:5, 1:5));
fprintf('DTW:\n'); disp(D_dtw(1:5, 1:5));
```

### Matrix Profile (Motif Discovery)

```matlab
% Single long time series
ts = randn(1, 1000);

% Compute matrix profile
mp = tsd_mp(ts, [], 25);  % Window size 25

% Find motif (nearest neighbor pair)
[~, idx1] = min(mp);
[~, idx2] = min(mp);  % Call again for second motif
fprintf('Motif indices: %d, %d\n', idx1, idx2);
```

### Distance with Band Constraint

```matlab
X = randn(100, 500);

% DTW with Sakoe-Chiba band (10% of diagonal)
D = tsd_dtw(X, [], 0.1);

% Faster but less accurate than full DTW
```

## Files in This Directory

| File | Description |
|------|-------------|
| `build_tsdistances.m` | Main build script (with auto-dependency checking) |
| `tsd_mex.c` | MEX gateway source code |
| `tsdistances.h` | C header file for MEX |
| `example_tsdistances.m` | Usage examples |
| `INSTALLATION_GUIDE.md` | **→ Start here for setup!** |
| `DEPENDENCIES.md` | Detailed dependency information |
| `install_dependencies.sh` | Auto-installer (macOS/Linux) |
| `install_dependencies.bat` | Dependency checker (Windows) |
| `README.md` | This file |

## Troubleshooting

### "Cargo not found"
- Install Rust: https://www.rust-lang.org/tools/install
- After install, restart your terminal and run: `source $HOME/.cargo/env`

### "MEX compiler not found"
- In MATLAB, run: `mex -setup C`
- Select your C compiler from the list

### "C compiler not found"
- **macOS**: `xcode-select --install`
- **Linux (Ubuntu)**: `sudo apt-get install build-essential`
- **Windows**: Install Visual Studio Build Tools

### Build fails with Vulkan errors
- GPU support is optional; build will complete without Vulkan
- To enable GPU: Download Vulkan SDK from https://vulkan.lunarg.com/sdk/home

### MEX compilation error on macOS
- Ensure MATLAB is configured with Xcode Command Line Tools:
  ```matlab
  mex -setup C
  ```
- Verify Xcode tools are installed: `xcode-select --install`

**For more troubleshooting, see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md#troubleshooting-reference).**

## Build System Details

### Automatic Dependency Checking

The `build_tsdistances.m` script now:
1. **Detects your platform** (macOS, Linux, Windows)
2. **Checks required tools**: Rust, MATLAB MEX, C compiler
3. **Verifies optional components**: Vulkan SDK, pkg-config
4. **Provides installation guidance** if anything is missing
5. **Builds the Rust library** in release mode
6. **Compiles MEX files** with correct linker settings
7. **Sets up MATLAB path** for easy access

### Build Optimization

- **Rust**: Compiled with LTO (Link Time Optimization), O3 optimization
- **Parallel**: Uses Rayon for CPU parallelization
- **GPU**: Optional Vulkan support for GPU acceleration
- **Static linking**: Prefers static libraries when available

## Performance Notes

- **Parallel by default**: All functions use all available CPU cores
- **Memory usage**: Distance matrix D is N×N, requiring ~8N² bytes
  - 1,000 time series: ~8 MB
  - 10,000 time series: ~800 MB
  - 100,000 time series: ~80 GB
- **GPU acceleration**: Available with Vulkan SDK (significant speedup for large datasets)

## License

This project is licensed under the same terms as the main tsdistances library.
See [LICENSE](../LICENSE) in the project root.

## Citation

If you use tsdistances in your research, please cite:

```bibtex
@software{azzari2024tsdistances,
  title={tsdistances: Time Series Distance Library (Rust backend)},
  author={Azzari, Alberto and others},
  year={2024},
  url={https://github.com/albertoazzari/tsdistances}
}
```

And the original papers for each distance measure (see table above).

## Support

- **Issues**: https://github.com/albertoazzari/tsdistances/issues
- **Documentation**: See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
- Include MATLAB version (`version` in MATLAB), OS, and full error messages


