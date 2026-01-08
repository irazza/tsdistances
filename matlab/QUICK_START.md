# TSDDistances MATLAB - Quick Start Card

## 30-Second Setup

### macOS/Linux Users:
```bash
bash install_dependencies.sh
# Then in MATLAB:
build_tsdistances
```

### Windows Users:
```cmd
install_dependencies.bat
# Then in MATLAB:
build_tsdistances
```

## First Use (in MATLAB):
```matlab
X = rand(100, 50);      % 100 time series, length 50
D = tsd_dtw(X);         % Compute DTW distances
disp(size(D));          % Verify: 100x100 matrix
```

---

## Need Help?

| Problem | Solution |
|---------|----------|
| Installation fails | See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) |
| Missing dependencies | Check [DEPENDENCIES.md](DEPENDENCIES.md) |
| Build errors | Run `install_dependencies.*` script for diagnostics |
| Runtime errors | Ensure `savepath` was run in MATLAB |
| More examples | See [README.md](README.md#usage-examples) |

---

## Available Functions (13 distance measures)

```matlab
D = tsd_dtw(X);           % Dynamic Time Warping
D = tsd_euclidean(X);     % Euclidean distance
D = tsd_erp(X);           % Edit Real Penalty
D = tsd_lcss(X);          % Longest Common Subsequence
D = tsd_msm(X);           % Move-Split-Merge
D = tsd_sbd(X);           % Shape-Based Distance
D = tsd_twe(X);           % Time Warp Edit
D = tsd_ddtw(X);          % Derivative DTW
D = tsd_wdtw(X);          % Weighted DTW
D = tsd_wddtw(X);         % Weighted Derivative DTW
D = tsd_adtw(X);          % Amerced DTW
D = tsd_catch_euclidean(X); % Catch22 + Euclidean
D = tsd_mp(X);            % Matrix Profile (motif discovery)
```

---

## Common Commands

```matlab
% Add to path permanently
savepath

% Clean rebuild (if needed)
cd /path/to/tsdistances
!cargo clean
cd matlab
build_tsdistances

% Configure MEX compiler (if needed)
mex -setup C
```

---

## Platform Requirements

### macOS
- Xcode Command Line Tools: `xcode-select --install`
- Rust: https://www.rust-lang.org/tools/install
- Configure MEX: `mex -setup C` (in MATLAB)

### Linux
- Build tools: `sudo apt-get install build-essential` (Ubuntu)
- Rust: https://www.rust-lang.org/tools/install
- Configure MEX: `mex -setup C` (in MATLAB)

### Windows
- Visual Studio Build Tools: https://visualstudio.microsoft.com/
- Rust: https://www.rust-lang.org/tools/install
- Configure MEX: `mex -setup C` (in MATLAB)

---

## Documentation Files

| File | Purpose |
|------|---------|
| **README.md** | Overview and usage examples |
| **INSTALLATION_GUIDE.md** | Detailed platform-specific setup |
| **DEPENDENCIES.md** | Complete dependency reference |
| **BUILD_SYSTEM_SUMMARY.md** | What's new in this version |
| **build_tsdistances.m** | Main build script (with auto-checking) |

**👉 Start with README.md then INSTALLATION_GUIDE.md**

---

## Performance Tips

1. **First build takes 2-5 minutes** - Rust compilation is slow
2. **Parallel by default** - Uses all CPU cores automatically
3. **Large datasets** - N×N distance matrix needs ~8N² bytes
   - 1,000 time series = 8 MB
   - 10,000 time series = 800 MB
4. **GPU acceleration** - Optional via Vulkan SDK (much faster)

---

## Example: Complete Workflow

```matlab
% 1. Generate sample data
X = randn(50, 200);  % 50 time series, length 200

% 2. Compute distances
D_dtw = tsd_dtw(X);
D_euc = tsd_euclidean(X);

% 3. Analyze
[nearest_dist, nearest_idx] = min(D_dtw(1, 2:end));
fprintf('Nearest match to series 1: series %d (distance %.3f)\n', ...
        nearest_idx+1, nearest_dist);

% 4. Visualize
figure;
imagesc(D_dtw);
colormap jet;
colorbar;
title('DTW Distance Matrix');
```

---

## Verify Installation

```matlab
% This should return a 5x5 matrix with 0s on diagonal
X = [1 2 3 4 5; 2 3 4 5 6; 3 4 5 6 7; 4 5 6 7 8; 5 6 7 8 9];
D = tsd_euclidean(X);
disp(D);

% Expected output:
%      0  1.4142  2.8284  4.2426  5.6569
%  1.4142      0  1.4142  2.8284  4.2426
%  2.8284  1.4142      0  1.4142  2.8284
%  4.2426  2.8284  1.4142      0  1.4142
%  5.6569  4.2426  2.8284  1.4142      0
```

---

## Get Help

- **Installation issues**: [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md#troubleshooting)
- **Dependency questions**: [DEPENDENCIES.md](DEPENDENCIES.md)
- **Usage questions**: [README.md](README.md#usage-examples)
- **GitHub**: https://github.com/albertoazzari/tsdistances/issues

**Happy time series analysis! 🚀**
