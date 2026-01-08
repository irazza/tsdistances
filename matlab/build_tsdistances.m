function build_tsdistances()
%BUILD_TSDISTANCES Build the tsdistances MEX file with dependency checking
%
%   BUILD_TSDISTANCES() compiles the Rust library and builds the MEX file,
%   with automatic dependency checking and installation guidance.
%
%   Requirements:
%       - Rust toolchain (cargo, rustc)
%       - MATLAB with MEX compiler configured
%       - System dependencies (handled by script)
%
%   This function will:
%       1. Check and validate all required tools
%       2. Build the Rust library in release mode (with matlab feature)
%       3. Compile the MEX gateway
%       4. Copy necessary files to the matlab directory

    fprintf('========================================\n');
    fprintf('  tsdistances MATLAB Build System\n');
    fprintf('========================================\n\n');
    
    % Get the directory of this script
    script_dir = fileparts(mfilename('fullpath'));
    project_root = fullfile(script_dir, '..');
    
    % Step 0: Check dependencies
    fprintf('[0/4] Checking dependencies...\n\n');
    check_and_install_dependencies(project_root);
    
    fprintf('\n[1/4] Building Rust library...\n');
    old_dir = cd(project_root);
    try
        % Build with matlab feature and without default features (no PyO3)
        build_cmd = 'cargo build --release --no-default-features --features matlab,use-compiled-tools 2>&1';
        [status, result] = system(build_cmd);
        if status ~= 0
            error('Failed to build Rust library:\n%s', result);
        end
        fprintf('Rust library built successfully.\n');
    catch ME
        cd(old_dir);
        rethrow(ME);
    end
    cd(old_dir);
    
    % Step 2: Determine library path and name based on OS
    fprintf('\n[2/4] Locating compiled library...\n');
    if ismac
        lib_name = 'libtsdistances.dylib';
        static_lib_name = 'libtsdistances.a';
    elseif isunix
        lib_name = 'libtsdistances.so';
        static_lib_name = 'libtsdistances.a';
    elseif ispc
        lib_name = 'tsdistances.dll';
        static_lib_name = 'tsdistances.lib';
    else
        error('Unsupported platform');
    end
    
    lib_path = fullfile(project_root, 'target', 'release');
    lib_file = fullfile(lib_path, lib_name);
    
    % Prefer static library
    static_lib_file = fullfile(lib_path, static_lib_name);
    use_static = exist(static_lib_file, 'file');
    
    if use_static
        lib_file = static_lib_file;
        lib_name = static_lib_name;
        fprintf('Using static library: %s\n', lib_file);
    elseif exist(lib_file, 'file')
        fprintf('Using dynamic library: %s\n', lib_file);
    else
        error('Could not find compiled library at: %s', lib_path);
    end
    
    % Step 3: Compile MEX file
    fprintf('\n[3/4] Compiling MEX file...\n');
    
    mex_src = fullfile(script_dir, 'tsd_mex.c');
    header_dir = script_dir;  % Header file is in matlab directory
    
    % Build MEX command with static library
    if use_static
        if ismac
            % For static library on macOS, link directly to the .a file
            mex_cmd = sprintf('mex -R2018a "%s" "%s" -I"%s" -outdir "%s"', ...
                              mex_src, lib_file, header_dir, script_dir);
        elseif isunix
            mex_cmd = sprintf('mex -R2018a "%s" "%s" -I"%s" -outdir "%s"', ...
                              mex_src, lib_file, header_dir, script_dir);
        else % Windows
            mex_cmd = sprintf('mex "%s" "%s" -I"%s" -outdir "%s"', ...
                              mex_src, lib_file, header_dir, script_dir);
        end
    else
        % Dynamic library
        if ismac
            mex_cmd = sprintf('mex -R2018a "%s" -L"%s" -ltsdistances -I"%s" -outdir "%s"', ...
                              mex_src, lib_path, header_dir, script_dir);
            % Add rpath for dynamic library
            mex_cmd = sprintf('%s LDFLAGS="$LDFLAGS -Wl,-rpath,''%s''"', mex_cmd, lib_path);
        elseif isunix
            mex_cmd = sprintf('mex -R2018a "%s" -L"%s" -ltsdistances -I"%s" -outdir "%s"', ...
                              mex_src, lib_path, header_dir, script_dir);
            mex_cmd = sprintf('%s LDFLAGS="$LDFLAGS -Wl,-rpath,''%s''"', mex_cmd, lib_path);
        else % Windows
            mex_cmd = sprintf('mex "%s" -L"%s" -ltsdistances -I"%s" -outdir "%s"', ...
                              mex_src, lib_path, header_dir, script_dir);
        end
    end
    
    fprintf('Running: %s\n', mex_cmd);
    eval(mex_cmd);
    
    fprintf('MEX file compiled successfully.\n');
    
    % Step 4: Copy dynamic library if needed (only for non-static)
    fprintf('\n[4/4] Finalizing installation...\n');
    
    if ~use_static
        dest_lib = fullfile(script_dir, lib_name);
        if ~strcmp(lib_file, dest_lib)
            copyfile(lib_file, dest_lib);
            fprintf('Copied library to: %s\n', dest_lib);
        end
    else
        fprintf('Using statically linked library.\n');
    end
    
    % Add to path
    if ~contains(path, script_dir)
        addpath(script_dir);
        fprintf('Added %s to MATLAB path.\n', script_dir);
        fprintf('Run "savepath" to make this permanent.\n');
    end
    
    fprintf('\n=== Build complete! ===\n');
    fprintf('You can now use tsdistances functions:\n');
    fprintf('  D = tsd_dtw(X);         %% DTW distance\n');
    fprintf('  D = tsd_euclidean(X);   %% Euclidean distance\n');
    fprintf('  D = tsd_erp(X);         %% ERP distance\n');
    fprintf('  ... and more. Type "help tsd_" and press Tab for all functions.\n');
end

% =========================================================================
% Helper function: Check and validate all required dependencies
% =========================================================================
function check_and_install_dependencies(project_root)
    
    if ismac
        check_dependencies_macos(project_root);
    elseif isunix
        check_dependencies_linux(project_root);
    elseif ispc
        check_dependencies_windows(project_root);
    end
end

% =========================================================================
% macOS dependency checker
% =========================================================================
function check_dependencies_macos(project_root)
    fprintf('Platform: macOS\n');
    
    % Check Rust
    fprintf('\n--- Checking Rust toolchain ---\n');
    [status, ~] = system('cargo --version 2>&1');
    if status ~= 0
        fprintf('ERROR: Cargo not found!\n');
        fprintf('Install Rust by running:\n');
        fprintf('  curl --proto ''=https'' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n');
        fprintf('Then add to your shell profile: source $HOME/.cargo/env\n');
        error('Rust toolchain is required.');
    else
        [~, rustc_info] = system('rustc --version');
        fprintf('✓ %s', rustc_info);
    end
    
    % Check MATLAB mex
    fprintf('\n--- Checking MATLAB MEX ---\n');
    [status, ~] = system('mex -v 2>&1 | head -1');
    if status ~= 0
        fprintf('ERROR: MATLAB MEX not found!\n');
        fprintf('Solution:\n');
        fprintf('  1. Make sure MATLAB is installed\n');
        fprintf('  2. Run from MATLAB: mex -setup C\n');
        fprintf('  3. Follow the prompts to configure a C compiler\n');
        error('MATLAB MEX compiler is required.');
    else
        fprintf('✓ MEX compiler is available\n');
    end
    
    % Check C compiler
    fprintf('\n--- Checking C compiler ---\n');
    [status, ~] = system('clang --version 2>&1 | head -1');
    if status ~= 0
        fprintf('WARNING: Clang not found!\n');
        fprintf('Install Xcode Command Line Tools:\n');
        fprintf('  xcode-select --install\n');
    else
        [~, clang_info] = system('clang --version 2>&1 | head -1');
        fprintf('✓ %s', clang_info);
    end
    
    % Check pkg-config for Vulkan/dependencies
    fprintf('\n--- Checking pkg-config ---\n');
    [status, ~] = system('pkg-config --version 2>&1');
    if status ~= 0
        fprintf('INFO: pkg-config not found (optional, for finding Vulkan SDK)\n');
        fprintf('If you have build issues with GPU support:\n');
        fprintf('  brew install pkg-config\n');
    else
        fprintf('✓ pkg-config is available\n');
    end
    
    % Optional: Check for Vulkan SDK
    fprintf('\n--- Checking Vulkan SDK (optional, for GPU support) ---\n');
    [status, ~] = system('ls /usr/local/lib/libvulkan* 2>/dev/null || ls /opt/vulkan*/lib/libvulkan* 2>/dev/null');
    if status ~= 0
        fprintf('INFO: Vulkan SDK not detected (GPU support will be disabled)\n');
        fprintf('To enable GPU support, download from: https://vulkan.lunarg.com/sdk/home\n');
    else
        fprintf('✓ Vulkan SDK detected\n');
    end
    
    fprintf('\n--- All required tools are available ---\n');
end

% =========================================================================
% Linux dependency checker
% =========================================================================
function check_dependencies_linux(project_root)
    fprintf('Platform: Linux\n');
    
    % Check Rust
    fprintf('\n--- Checking Rust toolchain ---\n');
    [status, ~] = system('cargo --version 2>&1');
    if status ~= 0
        fprintf('ERROR: Cargo not found!\n');
        fprintf('Install Rust by running:\n');
        fprintf('  curl --proto ''=https'' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n');
        fprintf('Then add to your shell profile: source $HOME/.cargo/env\n');
        error('Rust toolchain is required.');
    else
        [~, rustc_info] = system('rustc --version');
        fprintf('✓ %s', rustc_info);
    end
    
    % Check MATLAB mex
    fprintf('\n--- Checking MATLAB MEX ---\n');
    [status, ~] = system('mex -v 2>&1 | head -1');
    if status ~= 0
        fprintf('ERROR: MATLAB MEX not found!\n');
        fprintf('Solution:\n');
        fprintf('  1. Make sure MATLAB is installed\n');
        fprintf('  2. Run from MATLAB: mex -setup C\n');
        fprintf('  3. Follow the prompts to configure a C compiler\n');
        error('MATLAB MEX compiler is required.');
    else
        fprintf('✓ MEX compiler is available\n');
    end
    
    % Check GCC
    fprintf('\n--- Checking C compiler ---\n');
    [status, ~] = system('gcc --version 2>&1 | head -1');
    if status ~= 0
        fprintf('ERROR: GCC not found!\n');
        fprintf('Install build tools:\n');
        fprintf('  Ubuntu/Debian: sudo apt-get install build-essential\n');
        fprintf('  Fedora/RHEL: sudo yum install gcc gcc-c++ make\n');
        fprintf('  Arch: sudo pacman -S base-devel\n');
        error('C compiler (GCC) is required.');
    else
        [~, gcc_info] = system('gcc --version 2>&1 | head -1');
        fprintf('✓ %s', gcc_info);
    end
    
    % Check pkg-config
    fprintf('\n--- Checking pkg-config ---\n');
    [status, ~] = system('pkg-config --version 2>&1');
    if status ~= 0
        fprintf('ERROR: pkg-config not found!\n');
        fprintf('Install pkg-config:\n');
        fprintf('  Ubuntu/Debian: sudo apt-get install pkg-config\n');
        fprintf('  Fedora/RHEL: sudo yum install pkgconfig\n');
        error('pkg-config is required.');
    else
        fprintf('✓ pkg-config is available\n');
    end
    
    % Check for development headers
    fprintf('\n--- Checking development libraries ---\n');
    [status, ~] = system('pkg-config --cflags x11 2>/dev/null');
    if status ~= 0
        fprintf('INFO: X11 development headers not found\n');
        fprintf('Install x11 development libraries:\n');
        fprintf('  Ubuntu/Debian: sudo apt-get install libx11-dev libxrandr-dev\n');
        fprintf('  Fedora/RHEL: sudo yum install libX11-devel libXrandr-devel\n');
    else
        fprintf('✓ X11 development headers found\n');
    end
    
    % Check Vulkan SDK
    fprintf('\n--- Checking Vulkan SDK (optional, for GPU support) ---\n');
    [status, ~] = system('pkg-config --cflags vulkan 2>/dev/null');
    if status ~= 0
        fprintf('INFO: Vulkan SDK not detected (GPU support will be disabled)\n');
        fprintf('To enable GPU support:\n');
        fprintf('  Ubuntu/Debian: sudo apt-get install vulkan-tools vulkan-headers libvulkan-dev\n');
        fprintf('  Fedora/RHEL: sudo yum install vulkan-tools vulkan-devel\n');
        fprintf('  Or download from: https://vulkan.lunarg.com/sdk/home\n');
    else
        fprintf('✓ Vulkan SDK detected\n');
    end
    
    fprintf('\n--- All required tools are available ---\n');
end

% =========================================================================
% Windows dependency checker
% =========================================================================
function check_dependencies_windows(project_root)
    fprintf('Platform: Windows\n');
    
    % Check Rust
    fprintf('\n--- Checking Rust toolchain ---\n');
    [status, ~] = system('cargo --version 2>&1');
    if status ~= 0
        fprintf('ERROR: Cargo not found!\n');
        fprintf('Install Rust by downloading from: https://www.rust-lang.org/tools/install\n');
        fprintf('Or use: curl --proto ''=https'' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n');
        error('Rust toolchain is required.');
    else
        [~, rustc_info] = system('rustc --version');
        fprintf('✓ %s', rustc_info);
    end
    
    % Check MATLAB mex
    fprintf('\n--- Checking MATLAB MEX ---\n');
    [status, ~] = system('mex -v 2>&1');
    if status ~= 0
        fprintf('ERROR: MATLAB MEX not found!\n');
        fprintf('Solution:\n');
        fprintf('  1. Make sure MATLAB is installed\n');
        fprintf('  2. Run from MATLAB: mex -setup C\n');
        fprintf('  3. Follow the prompts to configure a C compiler\n');
        error('MATLAB MEX compiler is required.');
    else
        fprintf('✓ MEX compiler is available\n');
    end
    
    % Check for Visual Studio Build Tools or MSVC
    fprintf('\n--- Checking C compiler (MSVC) ---\n');
    [status, ~] = system('cl.exe 2>&1 | findstr /r "Microsoft"');
    if status ~= 0
        fprintf('WARNING: Microsoft Visual C++ compiler not found!\n');
        fprintf('Install one of the following:\n');
        fprintf('  1. Visual Studio Community Edition (https://visualstudio.microsoft.com/)\n');
        fprintf('  2. Visual Studio Build Tools (https://visualstudio.microsoft.com/downloads/)\n');
        fprintf('  3. Or ensure MSVC is in your PATH\n');
        fprintf('For Rust: https://doc.rust-lang.org/book/ch01-01-installation.html#installing-on-windows\n');
    else
        fprintf('✓ Microsoft Visual C++ compiler is available\n');
    end
    
    % Check Vulkan SDK
    fprintf('\n--- Checking Vulkan SDK (optional, for GPU support) ---\n');
    vulkan_paths = {
        'C:\VulkanSDK',
        fullfile(getenv('PROGRAMFILES'), 'VulkanSDK'),
        fullfile(getenv('PROGRAMFILES(X86)'), 'VulkanSDK')
    };
    
    vulkan_found = false;
    for i = 1:length(vulkan_paths)
        if isfolder(vulkan_paths{i})
            fprintf('✓ Vulkan SDK detected at: %s\n', vulkan_paths{i});
            vulkan_found = true;
            break;
        end
    end
    
    if ~vulkan_found
        fprintf('INFO: Vulkan SDK not detected (GPU support will be disabled)\n');
        fprintf('To enable GPU support, download from: https://vulkan.lunarg.com/sdk/home\n');
    end
    
    fprintf('\n--- Dependency check complete ---\n');
end
