# TSDDistances MATLAB - Complete Documentation Index

## 📚 Start Here!

### For the Impatient (< 5 minutes)
👉 **[QUICK_START.md](QUICK_START.md)** - 30-second setup guide with copy-paste commands

### For Setup & Installation
👉 **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Step-by-step guide for all platforms (macOS, Linux, Windows)

### For Dependency Questions
👉 **[DEPENDENCIES.md](DEPENDENCIES.md)** - What tools/libraries you need and how to get them

### For General Information
👉 **[README.md](README.md)** - Overview, usage examples, function reference

### What's New
👉 **[BUILD_SYSTEM_SUMMARY.md](BUILD_SYSTEM_SUMMARY.md)** - What improvements were made to the build system

---

## 📖 Complete Reading Guide

### Installation Path

1. **Start**: [QUICK_START.md](QUICK_START.md) (2 min)
   - Get a 30-second overview
   - Decide if you need more detail

2. **If you need help**: [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) (15 min)
   - Step-by-step platform-specific instructions
   - Troubleshooting section
   - Manual build instructions

3. **For dependency details**: [DEPENDENCIES.md](DEPENDENCIES.md) (10 min)
   - What each dependency does
   - How to check if it's installed
   - Platform-specific package names

### Usage Path

1. **After installation**: [README.md](README.md) (10 min)
   - Available distance functions
   - Basic and advanced examples
   - Performance notes

2. **For specific function help**:
   - In MATLAB: `help tsd_dtw` (etc.)
   - Or see [README.md](README.md#function-reference)

3. **For advanced features**: Check [README.md](README.md#advanced-configuration)

---

## 🗂️ File Organization

### Documentation Files (Read These)
```
QUICK_START.md              ← 30-second setup
INSTALLATION_GUIDE.md       ← Complete setup guide
DEPENDENCIES.md             ← What you need to install
README.md                   ← General info & usage
BUILD_SYSTEM_SUMMARY.md     ← What changed
INDEX.md                    ← This file
```

### Installer Scripts (Run These)
```
install_dependencies.sh     ← macOS/Linux auto-installer
install_dependencies.bat    ← Windows checker script
build_tsdistances.m         ← Main MATLAB build script
```

### Source/Library Files
```
tsd_mex.c                   ← MEX gateway source
tsdistances.h               ← C header
tsd_*.m                     ← Function stubs (13 functions)
```

---

## 🎯 Quick Reference

### I want to...

**Install tsdistances MATLAB**
→ [QUICK_START.md](QUICK_START.md) or [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)

**Know what dependencies I need**
→ [DEPENDENCIES.md](DEPENDENCIES.md)

**Build it manually**
→ [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md#manual-installation)

**Use the MEX functions**
→ [README.md](README.md#usage-examples)

**Troubleshoot build errors**
→ [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md#troubleshooting)

**Understand the build system**
→ [BUILD_SYSTEM_SUMMARY.md](BUILD_SYSTEM_SUMMARY.md)

**Check if Rust is installed**
→ Run `install_dependencies.*` script, or [DEPENDENCIES.md](DEPENDENCIES.md)

**Learn about GPU acceleration**
→ [DEPENDENCIES.md](DEPENDENCIES.md#gpu-support) or [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md#optional-install-vulkan-sdk)

---

## 🚀 Installation Checklist

### Before You Start
- [ ] You have MATLAB R2018a or newer
- [ ] You have administrator/sudo access for installations
- [ ] You have 1-2 GB free disk space (for Rust build)

### Quick Setup (5 minutes)
- [ ] Read [QUICK_START.md](QUICK_START.md)
- [ ] Run `install_dependencies.*` for your OS
- [ ] In MATLAB: Run `build_tsdistances`
- [ ] Verify: Run test code in MATLAB

### If Something Goes Wrong
- [ ] Check [QUICK_START.md](QUICK_START.md#need-help) troubleshooting table
- [ ] Read relevant section in [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md#troubleshooting)
- [ ] Review [DEPENDENCIES.md](DEPENDENCIES.md#troubleshooting-reference)

---

## 📋 Document Descriptions

### QUICK_START.md
- **Length**: ~150 lines
- **Reading time**: 2-5 minutes
- **Content**: 30-second setup, help table, common commands, examples
- **Best for**: Getting started immediately

### INSTALLATION_GUIDE.md
- **Length**: ~400 lines
- **Reading time**: 15-20 minutes
- **Content**: Platform-specific setup, manual build, GPU support, troubleshooting
- **Best for**: Complete setup information

### DEPENDENCIES.md
- **Length**: ~300 lines
- **Reading time**: 10-15 minutes
- **Content**: All required tools, system libraries, Rust crates, checklists
- **Best for**: Understanding what you need to install

### README.md
- **Length**: ~350 lines
- **Reading time**: 15-20 minutes
- **Content**: Usage examples, function reference, performance notes, citation
- **Best for**: Using the library after installation

### BUILD_SYSTEM_SUMMARY.md
- **Length**: ~250 lines
- **Reading time**: 10-15 minutes
- **Content**: What changed, improvements, features, dependencies
- **Best for**: Understanding the build system

### This File (INDEX.md)
- **Length**: ~250 lines
- **Reading time**: 5 minutes
- **Content**: Navigation guide, file organization, quick reference
- **Best for**: Navigating the documentation

---

## 💻 Platform-Specific Quick Links

### macOS Users
1. [QUICK_START.md - macOS](QUICK_START.md)
2. [INSTALLATION_GUIDE.md - macOS](INSTALLATION_GUIDE.md#macos-installation)
3. Run: `bash install_dependencies.sh`
4. In MATLAB: `build_tsdistances`

### Linux Users
1. [QUICK_START.md - Linux](QUICK_START.md)
2. [INSTALLATION_GUIDE.md - Linux](INSTALLATION_GUIDE.md#linux-installation)
3. Run: `bash install_dependencies.sh`
4. In MATLAB: `build_tsdistances`

### Windows Users
1. [QUICK_START.md - Windows](QUICK_START.md)
2. [INSTALLATION_GUIDE.md - Windows](INSTALLATION_GUIDE.md#windows-installation)
3. Run: `install_dependencies.bat`
4. In MATLAB: `build_tsdistances`

---

## 🔧 Build System Features

### Automatic Dependency Checking
- ✓ Detects your OS automatically
- ✓ Checks for required tools (Rust, MATLAB, C compiler)
- ✓ Provides installation instructions if missing
- ✓ Suggests optional GPU support
- ✓ Clear progress messages

### Supported Platforms
- ✓ macOS (Intel & Apple Silicon)
- ✓ Linux (Ubuntu, Debian, Fedora, RHEL, Arch)
- ✓ Windows (with MSVC compiler)

### Build Optimization
- ✓ Link Time Optimization (LTO)
- ✓ Release mode compilation (-O3)
- ✓ Static library preference (single file)
- ✓ Automatic path configuration

---

## 📞 Getting Help

### For Installation Issues
→ [INSTALLATION_GUIDE.md#troubleshooting](INSTALLATION_GUIDE.md#troubleshooting)

### For Dependency Questions
→ [DEPENDENCIES.md#troubleshooting-reference](DEPENDENCIES.md#troubleshooting-reference)

### For Usage Questions
→ [README.md#usage-examples](README.md#usage-examples)

### For GitHub Issues
→ https://github.com/albertoazzari/tsdistances/issues

**Include**: OS, MATLAB version, full error message, what you've tried

---

## 📊 Documentation Statistics

| Document | Lines | Minutes | Purpose |
|----------|-------|---------|---------|
| QUICK_START.md | ~150 | 2-5 | Quick setup |
| INSTALLATION_GUIDE.md | ~400 | 15-20 | Complete setup |
| DEPENDENCIES.md | ~300 | 10-15 | Dependency reference |
| README.md | ~350 | 15-20 | Usage guide |
| BUILD_SYSTEM_SUMMARY.md | ~250 | 10-15 | System overview |
| This index | ~250 | 5-10 | Navigation |
| **Total** | **~1,700** | **60-90** | Everything |

---

## 🎓 Learning Path

### Minimal Path (5 minutes)
1. [QUICK_START.md](QUICK_START.md) (2 min)
2. Run installer script (2 min)
3. `build_tsdistances` in MATLAB (1 min)
4. Done!

### Standard Path (30 minutes)
1. [QUICK_START.md](QUICK_START.md) (5 min)
2. [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) (15 min)
3. Run installer and build (5 min)
4. [README.md](README.md) quick examples (5 min)

### Complete Path (90 minutes)
1. [QUICK_START.md](QUICK_START.md) (5 min)
2. [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) (20 min)
3. [DEPENDENCIES.md](DEPENDENCIES.md) (15 min)
4. [README.md](README.md) (20 min)
5. [BUILD_SYSTEM_SUMMARY.md](BUILD_SYSTEM_SUMMARY.md) (15 min)
6. Install and test (15 min)

---

## ✅ Success Indicators

After successful installation, you should be able to:

```matlab
% 1. Run this without errors
X = rand(10, 50);
D = tsd_dtw(X);

% 2. Get a 10x10 matrix
size(D)  % Should be [10, 10]

% 3. See symmetric distances
D(1,2) == D(2,1)  % Should be true

% 4. See zeros on diagonal
D(1,1)  % Should be 0 or very close
```

If all of these work, you're ready to use tsdistances! 🎉

---

## 📝 Version Information

- **Last Updated**: January 2026
- **Build System Version**: 2.0 (with auto-dependency checking)
- **Supported MATLAB**: R2018a and newer
- **Rust Edition**: 2024
- **Python Support**: 3.12+ (for Python builds, not needed for MATLAB)

---

## 🔄 Feedback & Updates

- **Have suggestions?** Visit: https://github.com/albertoazzari/tsdistances
- **Found a bug?** Create an issue with OS, MATLAB version, and error message
- **Want to contribute?** See the GitHub repository

---

**Happy time series computing! 🚀**

*For quick setup: Start with [QUICK_START.md](QUICK_START.md)*

*For complete info: See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)*
