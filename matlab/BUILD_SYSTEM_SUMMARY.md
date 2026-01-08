# TSDDistances MATLAB Build System - Summary

## What Has Been Improved

### 1. **Enhanced Build Script** (`build_tsdistances.m`)
The build script now includes **automatic dependency checking** for all three platforms:

- **macOS**: Checks for Rust, Xcode CLI Tools, pkg-config, and Vulkan SDK
- **Linux**: Checks for Rust, GCC, pkg-config, X11 dev headers, and Vulkan SDK
- **Windows**: Checks for Rust, MSVC compiler, and Vulkan SDK

When dependencies are missing, the script provides clear instructions for installation specific to each platform.

#### Key Features:
- ✓ Automatic platform detection
- ✓ Real-time dependency validation
- ✓ Helpful error messages with installation commands
- ✓ Support for optional GPU acceleration
- ✓ Step-by-step build progress display

### 2. **Comprehensive Installation Guide** (`INSTALLATION_GUIDE.md`)
A detailed 300+ line guide covering:

- **Quick Start** - Get building in 3 steps
- **Platform-Specific Setup**:
  - macOS: Rust, Xcode, MEX configuration
  - Linux: Rust, build tools, MEX setup (Ubuntu, Fedora, Arch)
  - Windows: Rust, Visual Studio Build Tools, MEX setup
- **Dependency Details**: All Rust crates and system libraries
- **GPU Support**: Vulkan SDK installation
- **Troubleshooting**: Common issues and solutions
- **Advanced Configuration**: Custom Rust versions, clean rebuild

### 3. **Dependencies Reference** (`DEPENDENCIES.md`)
A quick-reference guide including:

- **All Required Tools**: With download links and verification commands
- **System Libraries**: Platform-specific headers and libraries
- **Cargo Dependencies**: Table of all Rust crates from Cargo.toml
- **Checklist**: Quick verification checklist for each platform
- **Build Features**: Explanation of `use-compiled-tools` vs `use-installed-tools`
- **Environment Variables**: For advanced configuration

### 4. **Automated Installers**

#### macOS/Linux (`install_dependencies.sh`)
Bash script that:
- Automatically detects Linux distribution (Ubuntu, Fedora, Arch)
- Installs Rust toolchain
- Installs platform-specific build tools
- Provides Vulkan SDK guidance
- Works on macOS, Ubuntu/Debian, Fedora/RHEL, and Arch

#### Windows (`install_dependencies.bat`)
Batch script that:
- Checks for Rust toolchain
- Verifies MSVC compiler
- Validates MATLAB MEX
- Detects Vulkan SDK
- Provides download links for missing components

### 5. **Updated README** (`README.md`)
Reorganized documentation with:

- **Quick Start** linking to installation guides
- **Available Functions**: All 13 distance measures
- **Requirements** (clear breakdown of required vs optional)
- **Documentation Index**: Links to all setup guides
- **Platform-Specific Instructions**: Quick copy-paste setup
- **Usage Examples**: Basic to advanced usage patterns
- **Troubleshooting**: Direct solutions
- **Performance Notes**: Memory and parallelization info

## Files Modified/Created

### Modified:
- `build_tsdistances.m` - Enhanced with 250+ lines of dependency checking code

### Created:
1. `INSTALLATION_GUIDE.md` - Complete setup guide (300+ lines)
2. `DEPENDENCIES.md` - Dependency reference (250+ lines)
3. `install_dependencies.sh` - Automated Linux/macOS installer
4. `install_dependencies.bat` - Windows dependency checker
5. Updated `README.md` - Comprehensive documentation with quick links

## How to Use

### For End Users:

1. **First Time Setup**:
   ```bash
   # macOS/Linux
   bash install_dependencies.sh
   
   # Windows
   install_dependencies.bat
   ```

2. **Build in MATLAB**:
   ```matlab
   cd /path/to/tsdistances/matlab
   build_tsdistances
   ```

3. **Use the Functions**:
   ```matlab
   X = rand(100, 50);
   D = tsd_dtw(X);
   ```

### For Debugging:

- If build fails: Check `INSTALLATION_GUIDE.md` troubleshooting section
- For dependency info: Refer to `DEPENDENCIES.md`
- For quick checks: Run the appropriate `install_dependencies.*` script

## Key Improvements Summary

| Aspect | Before | After |
|--------|--------|-------|
| Dependency checking | None | Automatic on all 3 platforms |
| Error messages | Generic | Specific with install commands |
| Installation guidance | Minimal | Comprehensive with links |
| Platform support | Basic | macOS, Linux (4 distros), Windows |
| GPU support | Not mentioned | Clear optional feature info |
| Troubleshooting | Limited | Detailed with solutions |
| Documentation | Minimal | 600+ lines across 4 docs |

## Dependencies Identified (from Cargo.toml)

### Rust Crates (auto-downloaded):
- `catch22` - Time series utilities
- `ctrlc` - Signal handling
- `parking_lot` - Synchronization
- `pyo3` - Python bindings (optional)
- `rand` - Random numbers
- `rayon` - Data parallelism
- `rustfft` - FFT algorithms
- `vulkano` - GPU via Vulkan
- `tsdistances_gpu` - GPU library

### System Requirements:

**macOS:**
- Rust toolchain
- Xcode Command Line Tools
- pkg-config (recommended)
- Vulkan SDK (optional)

**Linux:**
- Rust toolchain
- GCC/build-essential
- pkg-config
- libx11-dev, libxrandr-dev
- Vulkan SDK (optional)

**Windows:**
- Rust toolchain
- Visual Studio Build Tools
- MSVC compiler
- Vulkan SDK (optional)

## Build Features

The build uses:
```bash
cargo build --release --no-default-features --features matlab,use-compiled-tools
```

- **matlab** feature: Enables MEX bindings
- **use-compiled-tools**: Uses pre-compiled Vulkan (no SDK required)
- **--no-default-features**: Disables Python bindings
- **--release**: Optimization flags (O3, LTO)

## Next Steps

1. User runs `install_dependencies.*` script
2. User runs `build_tsdistances` in MATLAB
3. Script automatically checks all dependencies
4. Script builds Rust library
5. Script compiles MEX files
6. Script sets up MATLAB path

All with clear progress messages and helpful error guidance!
