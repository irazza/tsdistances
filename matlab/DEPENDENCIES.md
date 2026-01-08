# TSDDistances Dependencies Reference

## Quick Reference for All Required Dependencies

### Core Dependencies (Required)

#### Rust Toolchain
- **Package**: Rust/Cargo
- **Version**: Latest stable (or as specified in `rust-toolchain.toml`)
- **Download**: https://www.rust-lang.org/tools/install
- **Install**:
  - macOS/Linux: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
  - Windows: Download installer from https://www.rust-lang.org/tools/install
- **Verify**: `cargo --version` and `rustc --version`

#### MATLAB
- **Version**: R2018a or newer
- **Download**: https://www.mathworks.com/
- **Requirements**: Must have MEX compiler support
- **Setup**: Run `mex -setup C` in MATLAB

#### C Compiler (Platform-specific)
- **macOS**: Clang (included in Xcode Command Line Tools)
  - Install: `xcode-select --install`
  - Verify: `clang --version`
  
- **Linux**: GCC
  - Ubuntu: `sudo apt-get install build-essential`
  - Fedora: `sudo yum groupinstall "Development Tools"`
  - Arch: `sudo pacman -S base-devel`
  - Verify: `gcc --version`
  
- **Windows**: Microsoft Visual C++ (MSVC)
  - Install: Visual Studio Community or Build Tools
  - Download: https://visualstudio.microsoft.com/
  - Verify: `cl.exe` in Visual Studio Command Prompt

### Supporting Tools (Required)

#### pkg-config
Used to locate system libraries and development headers.

- **macOS**:
  - Install: `brew install pkg-config`
  - Verify: `pkg-config --version`
  
- **Linux**:
  - Ubuntu: `sudo apt-get install pkg-config`
  - Fedora: `sudo yum install pkgconfig`
  - Arch: `sudo pacman -S pkg-config`
  - Verify: `pkg-config --version`
  
- **Windows**:
  - Available via MSVC compiler suite
  - May need to be added to PATH

#### Development Headers (Linux only)
- **X11 Libraries**:
  - Ubuntu: `sudo apt-get install libx11-dev libxrandr-dev`
  - Fedora: `sudo yum install libX11-devel libXrandr-devel`
  - Arch: `sudo pacman -S libx11 libxrandr`

### GPU Support (Optional)

#### Vulkan SDK
Optional but recommended for GPU acceleration via `tsdistances_gpu`.

- **Website**: https://vulkan.lunarg.com/sdk/home
- **macOS**:
  - Download: Vulkan SDK from website
  - Or via Homebrew: `brew install vulkan-headers vulkan-loader`
  - Verify: `pkg-config --cflags vulkan`
  
- **Linux**:
  - Ubuntu: `sudo apt-get install vulkan-tools vulkan-headers libvulkan-dev`
  - Fedora: `sudo yum install vulkan-tools vulkan-devel`
  - Arch: `sudo pacman -S vulkan-headers vulkan-loader`
  - Verify: `pkg-config --cflags vulkan`
  
- **Windows**:
  - Download and run installer from: https://vulkan.lunarg.com/sdk/home
  - Default install location: `C:\VulkanSDK`
  - Verify: Check `C:\VulkanSDK` folder exists

### Rust Crate Dependencies (Cargo.toml)

These are automatically downloaded by Cargo during build:

| Crate | Version | Purpose | GPU? |
|-------|---------|---------|------|
| `catch22` | git | Time series distance utilities | - |
| `ctrlc` | 3.4.7 | Signal handling | - |
| `parking_lot` | 0.12.4 | Synchronization primitives | - |
| `pyo3` | 0.27.2 | Python bindings (optional) | - |
| `rand` | 0.8.5 | Random number generation | - |
| `rayon` | 1.10.0 | Data parallelism | CPU |
| `rustfft` | 6.3.0 | FFT computations | - |
| `vulkano` | 0.35.1 | GPU abstraction layer | GPU |
| `tsdistances_gpu` | git | GPU algorithms | GPU |

---

## Installation Checklist

### macOS
- [ ] Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- [ ] Xcode CLI Tools: `xcode-select --install`
- [ ] pkg-config: `brew install pkg-config`
- [ ] MATLAB configured: Run `mex -setup C` in MATLAB
- [ ] (Optional) Vulkan: `brew install vulkan-headers vulkan-loader`

### Linux (Ubuntu/Debian)
- [ ] Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- [ ] Build tools: `sudo apt-get install build-essential pkg-config libx11-dev libxrandr-dev`
- [ ] MATLAB configured: Run `mex -setup C` in MATLAB
- [ ] (Optional) Vulkan: `sudo apt-get install vulkan-tools vulkan-headers libvulkan-dev`

### Linux (Fedora/RHEL)
- [ ] Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- [ ] Build tools: `sudo yum groupinstall "Development Tools"` and `sudo yum install pkg-config libX11-devel libXrandr-devel`
- [ ] MATLAB configured: Run `mex -setup C` in MATLAB
- [ ] (Optional) Vulkan: `sudo yum install vulkan-tools vulkan-devel`

### Windows
- [ ] Rust: Download and install from https://www.rust-lang.org/tools/install
- [ ] Visual Studio Build Tools: https://visualstudio.microsoft.com/
- [ ] MATLAB configured: Run `mex -setup C` in MATLAB
- [ ] (Optional) Vulkan: Download from https://vulkan.lunarg.com/sdk/home

---

## Build Features

The build script uses these Cargo features:

### Default Build (MATLAB)
```bash
cargo build --release --no-default-features --features matlab,use-compiled-tools
```

- `matlab`: Enable MATLAB MEX bindings
- `use-compiled-tools`: Use pre-compiled Vulkan tools (no Vulkan SDK required)

### Alternative: Use System Vulkan
```bash
cargo build --release --no-default-features --features matlab,use-installed-tools
```

Requires Vulkan SDK to be installed.

### Python Only (not needed for MATLAB)
```bash
cargo build --release --no-default-features --features python,use-compiled-tools
```

---

## Troubleshooting Reference

### Common Error: "Cargo not found"
- **Cause**: Rust not installed or not in PATH
- **Solution**: 
  1. Install: https://www.rust-lang.org/tools/install
  2. Restart terminal
  3. Verify: `cargo --version`

### Common Error: "MEX compiler not found"
- **Cause**: MATLAB MEX not configured
- **Solution**:
  1. In MATLAB: `mex -setup C`
  2. Select appropriate compiler
  3. Verify: `mex -v` shows compiler info

### Common Error: "Vulkan library not found"
- **Cause**: GPU support is optional
- **Solution**: 
  1. Build will work without Vulkan (no GPU support)
  2. To enable GPU: Install Vulkan SDK from https://vulkan.lunarg.com/sdk/home
  3. Or use `use-compiled-tools` feature (default)

### Common Error: "No C compiler found"
- **Cause**: C compiler not installed or not in PATH
- **Solution** (platform-specific):
  - **macOS**: `xcode-select --install`
  - **Ubuntu**: `sudo apt-get install build-essential`
  - **Fedora**: `sudo yum groupinstall "Development Tools"`
  - **Windows**: Install Visual Studio Build Tools

---

## Environment Variables (Advanced)

### Rust Toolchain
- `RUSTUP_HOME`: Default `$HOME/.rustup`
- `CARGO_HOME`: Default `$HOME/.cargo`

### MATLAB
- `MATLAB_ROOT`: MATLAB installation directory
- `MATLABPATH`: MATLAB search path

### Vulkan
- `VK_SDK_PATH`: Vulkan SDK installation directory (if not in default location)
- `PKG_CONFIG_PATH`: Add Vulkan lib/pkgconfig to this for pkg-config discovery

---

## Version Information

Current project configuration:
- **Rust Edition**: 2024 (from Cargo.toml)
- **Minimum Python**: 3.12+ (for Python bindings)
- **MATLAB**: R2018a or newer
- **Vulkano**: 0.35.1
- **PyO3**: 0.27.2 (Python bindings only)

Check `Cargo.toml` and `rust-toolchain.toml` for authoritative version information.
