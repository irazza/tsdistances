#!/bin/bash
# TSDDistances Quick Installation Script
# This script attempts to automatically install all required dependencies for tsdistances

set -e  # Exit on error

echo "=================================="
echo "  tsdistances Quick Install"
echo "=================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
else
    echo "ERROR: Unsupported operating system"
    echo "Please refer to INSTALLATION_GUIDE.md for manual installation"
    exit 1
fi

echo "Detected OS: $OS"
echo ""

# macOS Installation
if [ "$OS" = "macOS" ]; then
    echo "[1/4] Installing curl (if needed) and Rust toolchain..."
    if ! command -v curl &> /dev/null; then
        if ! command -v brew &> /dev/null; then
            echo "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install curl
    fi
    if ! command -v cargo &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    else
        echo "✓ Rust already installed: $(rustc --version)"
    fi
    
    echo ""
    echo "[2/4] Installing Xcode Command Line Tools..."
    if ! command -v clang &> /dev/null; then
        xcode-select --install
        echo "Please complete the Xcode installation when prompted"
    else
        echo "✓ Xcode Command Line Tools already installed"
    fi
    
    echo ""
    echo "[3/4] Installing pkg-config (recommended)..."
    if ! command -v pkg-config &> /dev/null; then
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install pkg-config
    else
        echo "✓ pkg-config already installed"
    fi
    
    echo ""
    echo "[4/4] Vulkan SDK (optional for GPU support)..."
    if [ ! -d "/usr/local/lib/vulkan" ] && [ ! -d "/opt/vulkan" ]; then
        echo "Vulkan SDK not found. To install:"
        echo "  1. Download from: https://vulkan.lunarg.com/sdk/home"
        echo "  2. Or install via: brew install vulkan-headers vulkan-loader"
    else
        echo "✓ Vulkan SDK detected"
    fi
    
elif [ "$OS" = "Linux" ]; then
    # Detect Linux distribution
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
    else
        echo "ERROR: Could not detect Linux distribution"
        exit 1
    fi
    
    echo "Detected distribution: $DISTRO"
    echo ""
    
    if [ "$DISTRO" = "ubuntu" ] || [ "$DISTRO" = "debian" ]; then
        echo "[1/4] Updating package lists..."
        sudo apt-get update
        
        echo "[2/4] Installing curl and Rust toolchain..."
        sudo apt-get install -y curl
        if ! command -v cargo &> /dev/null; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source $HOME/.cargo/env
        else
            echo "✓ Rust already installed: $(rustc --version)"
        fi
        
        echo "[3/4] Installing build tools..."
        sudo apt-get install -y build-essential pkg-config libx11-dev libxrandr-dev
        
        echo "[4/4] Vulkan SDK (optional for GPU support)..."
        if ! dpkg -l | grep -q libvulkan1; then
            echo "To install Vulkan support, run:"
            echo "  sudo apt-get install vulkan-tools vulkan-headers libvulkan-dev"
        else
            echo "✓ Vulkan SDK detected"
        fi
        
    elif [ "$DISTRO" = "fedora" ] || [ "$DISTRO" = "rhel" ] || [ "$DISTRO" = "centos" ]; then
        echo "[1/4] Installing curl and Rust toolchain..."
        sudo yum install -y curl
        if ! command -v cargo &> /dev/null; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source $HOME/.cargo/env
        else
            echo "✓ Rust already installed: $(rustc --version)"
        fi
        
        echo "[2/4] Installing build tools..."
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y pkg-config libX11-devel libXrandr-devel
        
        echo "[3/4] Vulkan SDK (optional for GPU support)..."
        if ! rpm -q vulkan-tools &> /dev/null; then
            echo "To install Vulkan support, run:"
            echo "  sudo yum install vulkan-tools vulkan-devel"
        else
            echo "✓ Vulkan SDK detected"
        fi
        
    elif [ "$DISTRO" = "arch" ]; then
        echo "[1/4] Installing curl and Rust toolchain..."
        sudo pacman -S --noconfirm curl
        if ! command -v cargo &> /dev/null; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source $HOME/.cargo/env
        else
            echo "✓ Rust already installed: $(rustc --version)"
        fi
        
        echo "[2/4] Installing build tools..."
        sudo pacman -S --noconfirm base-devel pkg-config
        
        echo "[3/4] Vulkan SDK (optional for GPU support)..."
        if ! pacman -Q vulkan-tools &> /dev/null; then
            echo "To install Vulkan support, run:"
            echo "  sudo pacman -S vulkan-headers vulkan-loader"
        else
            echo "✓ Vulkan SDK detected"
        fi
    else
        echo "WARNING: Unsupported Linux distribution: $DISTRO"
        echo "Please refer to INSTALLATION_GUIDE.md for manual installation"
        exit 1
    fi
fi

echo ""
echo "=================================="
echo "Installation complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Ensure MATLAB MEX compiler is configured:"
echo "   In MATLAB: mex -setup C"
echo ""
echo "2. Build tsdistances:"
echo "   In MATLAB: cd /path/to/tsdistances/matlab"
echo "   In MATLAB: build_tsdistances"
echo ""
echo "For more details, see:"
echo "  - INSTALLATION_GUIDE.md"
echo "  - DEPENDENCIES.md"
echo ""
