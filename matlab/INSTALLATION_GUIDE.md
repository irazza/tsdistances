# TSDDistances MATLAB Installation Guide

This guide will help you set up all necessary dependencies and build the tsdistances MATLAB MEX bindings.

## Quick Start

Once all dependencies are installed (see below), simply run in MATLAB:

```matlab
build_tsdistances
```

The script will automatically:
1. Check all dependencies
2. Build the Rust library
3. Compile the MEX files
4. Set up the MATLAB path

## System Requirements

### Required:
- **Rust toolchain** (cargo, rustc)
- **MATLAB** with MEX compiler configured
- **C compiler** (platform-specific)

### Optional:
- **Vulkan SDK** (for GPU acceleration support)

---

## macOS Installation

### 1. Install Rust Toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify installation:
```bash
cargo --version
rustc --version
```

### 2. Install Xcode Command Line Tools

```bash
xcode-select --install
```

This provides the C compiler (clang) needed for compilation.

### 3. Configure MATLAB MEX Compiler

In MATLAB, run:
```matlab
mex -setup C
```

Select the Xcode C compiler from the list and confirm.

### 4. (Optional) Install pkg-config for Vulkan Support

```bash
brew install pkg-config
```

### 5. (Optional) Install Vulkan SDK for GPU Support

Download from: https://vulkan.lunarg.com/sdk/home

Or via Homebrew:
```bash
brew install vulkan-headers vulkan-loader
```

---

## Linux Installation

### For Ubuntu/Debian:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install build tools
sudo apt-get update
sudo apt-get install build-essential pkg-config
sudo apt-get install libx11-dev libxrandr-dev

# (Optional) Install Vulkan SDK for GPU support
sudo apt-get install vulkan-tools vulkan-headers libvulkan-dev
```

### For Fedora/RHEL:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install build tools
sudo yum groupinstall "Development Tools"
sudo yum install pkg-config
sudo yum install libX11-devel libXrandr-devel

# (Optional) Install Vulkan SDK for GPU support
sudo yum install vulkan-tools vulkan-devel
```

### For Arch Linux:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install build tools
sudo pacman -S base-devel
sudo pacman -S pkg-config

# (Optional) Install Vulkan SDK
sudo pacman -S vulkan-headers vulkan-loader
```

### Configure MATLAB MEX Compiler

In MATLAB, run:
```matlab
mex -setup C
```

Select GCC as your compiler and confirm.

---

## Windows Installation

### 1. Install Rust Toolchain

Download and run the installer from: https://www.rust-lang.org/tools/install

During installation, select to install the MSVC toolchain when prompted.

Verify installation by opening Command Prompt/PowerShell:
```cmd
cargo --version
rustc --version
```

### 2. Install Visual Studio Build Tools

Download from: https://visualstudio.microsoft.com/downloads/

Choose one of:
- **Visual Studio Community Edition** (full IDE)
- **Visual Studio Build Tools** (minimal build tools only)

When installing, ensure "Desktop development with C++" is selected.

### 3. Configure MATLAB MEX Compiler

In MATLAB, run:
```matlab
mex -setup C
```

Select the Microsoft Visual C++ compiler and confirm.

### 4. (Optional) Install Vulkan SDK for GPU Support

Download from: https://vulkan.lunarg.com/sdk/home

Run the installer and follow the prompts. The SDK will be installed to `C:\VulkanSDK`.

---

## Dependency Details

### Rust Dependencies (from Cargo.toml)

The project depends on:

| Package | Purpose |
|---------|---------|
| `catch22` | Time series utility functions |
| `ctrlc` | Signal handling (Python builds) |
| `parking_lot` | Synchronization primitives |
| `pyo3` | Python bindings (Python builds only) |
| `rand` | Random number generation |
| `rayon` | Parallel processing |
| `rustfft` | FFT algorithms |
| `vulkano` | GPU computation via Vulkan |
| `tsdistances_gpu` | GPU acceleration library |

When building with the `use-compiled-tools` feature, Vulkan tools are pre-compiled. If you have build issues, you can use `use-installed-tools` feature instead to use your system Vulkan SDK.

### System Libraries

- **libvulkan**: Vulkan runtime (for GPU support)
- **libx11**: X11 graphics library (Linux)
- **libxrandr**: X Resize and Rotate library (Linux)

---

## Building from MATLAB

Once dependencies are installed:

```matlab
% From MATLAB command window, navigate to the matlab directory
cd /path/to/tsdistances/matlab

% Run the build script
build_tsdistances
```

The script will:
1. ✓ Check all required tools
2. ✓ Build the Rust library (release mode)
3. ✓ Compile the MEX gateway
4. ✓ Link libraries and finalize

You should see:
```
========================================
  tsdistances MATLAB Build System
========================================

[0/4] Checking dependencies...
[1/4] Building Rust library...
[2/4] Locating compiled library...
[3/4] Compiling MEX file...
[4/4] Finalizing installation...

=== Build complete! ===
```

---

## Troubleshooting

### "Command 'cargo' not found"
- Rust is not installed or not in your PATH
- After installing Rust, restart your terminal
- On macOS/Linux: Ensure `~/.cargo/env` is sourced in your shell profile

### "MEX compiler not found"
- MATLAB is not installed or MEX is not configured
- Run `mex -setup C` in MATLAB and select your compiler

### Build fails with "Vulkan not found"
- GPU features are optional; the build will continue without GPU support
- To enable GPU support, install Vulkan SDK (see platform-specific instructions)
- Or use `-no-default-features --features matlab` flag if issues persist

### "clang/gcc/cl.exe not found" (compiler error)
- Install the appropriate compiler for your platform:
  - **macOS**: `xcode-select --install`
  - **Linux**: `sudo apt-get install build-essential` (Ubuntu)
  - **Windows**: Install Visual Studio Build Tools

### Permission denied when installing packages
- Use `sudo` if required by your system
- Or use Homebrew on macOS which doesn't require sudo for most packages

### MATLAB MEX compilation fails with linking errors
- Ensure the Rust library was built successfully (check `target/release/` folder)
- Verify that the C compiler is properly configured in MATLAB
- Run `mex -setup C` again and carefully select the correct compiler

---

## Verifying Installation

After successful build, test the MEX functions:

```matlab
% Test basic functionality
X = rand(100, 50);  % 100 time series, length 50 each

% Run a distance computation
D = tsd_dtw(X);
disp(size(D));  % Should be 100x100

% Try other functions
D_euc = tsd_euclidean(X);
D_erp = tsd_erp(X);

fprintf('Installation successful!\n');
```

---

## Advanced Configuration

### Building with GPU Support

The build script automatically handles GPU support if Vulkan SDK is available.

To force GPU features:
```bash
cargo build --release --no-default-features --features matlab,use-installed-tools
```

### Building with Specific Rust Version

If you need a specific Rust version, check `rust-toolchain.toml` in the project root:

```bash
rustup override set nightly  # or specific version
```

### Clean and Rebuild

To remove previous build artifacts and start fresh:

```matlab
cd /path/to/tsdistances
!cargo clean
build_tsdistances
```

---

## Support

For issues, please visit: https://github.com/albertoazzari/tsdistances/issues

Include:
- Your operating system and version
- MATLAB version (run `version` in MATLAB)
- Rust version (run `rustc --version`)
- Full error messages from the build script
