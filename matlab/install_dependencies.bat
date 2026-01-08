@echo off
REM TSDDistances Quick Installation Script for Windows
REM This script provides guidance for installing dependencies on Windows

echo.
echo =====================================
echo   tsdistances Windows Quick Install
echo =====================================
echo.

echo [1/4] Checking Rust toolchain...
where cargo >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Rust toolchain not found!
    echo.
    echo Install Rust from: https://www.rust-lang.org/tools/install
    echo Or run this PowerShell command:
    echo   curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs^| sh
    echo.
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('rustc --version') do echo ✓ %%i
)

echo.
echo [2/4] Checking MSVC compiler...
where cl.exe >nul 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Microsoft Visual C++ not found!
    echo.
    echo Install one of the following:
    echo   1. Visual Studio Community Edition
    echo      https://visualstudio.microsoft.com/
    echo   2. Visual Studio Build Tools
    echo      https://visualstudio.microsoft.com/downloads/
    echo.
    echo When installing, select "Desktop development with C++"
    echo.
) else (
    echo ✓ Microsoft Visual C++ compiler found
)

echo.
echo [3/4] Checking MATLAB MEX...
where mex >nul 2>nul
if %errorlevel% neq 0 (
    echo WARNING: MATLAB MEX compiler not configured!
    echo.
    echo In MATLAB, run: mex -setup C
    echo Then select your C compiler
    echo.
) else (
    echo ✓ MATLAB MEX compiler found
)

echo.
echo [4/4] Checking Vulkan SDK (optional for GPU support)...
if exist "C:\VulkanSDK" (
    echo ✓ Vulkan SDK detected at C:\VulkanSDK
) else if exist "%ProgramFiles%\VulkanSDK" (
    echo ✓ Vulkan SDK detected at %ProgramFiles%\VulkanSDK
) else if exist "%ProgramFiles(x86)%\VulkanSDK" (
    echo ✓ Vulkan SDK detected at %ProgramFiles(x86)%\VulkanSDK
) else (
    echo INFO: Vulkan SDK not detected
    echo GPU support is optional. To enable it:
    echo   Download from: https://vulkan.lunarg.com/sdk/home
)

echo.
echo =====================================
echo Installation check complete!
echo =====================================
echo.
echo Next steps:
echo 1. Ensure all required tools are installed (see above)
echo 2. In MATLAB Command Window:
echo    mex -setup C
echo 3. Then build tsdistances:
echo    cd C:\path\to\tsdistances\matlab
echo    build_tsdistances
echo.
echo For more details, see:
echo   - INSTALLATION_GUIDE.md
echo   - DEPENDENCIES.md
echo.
pause
