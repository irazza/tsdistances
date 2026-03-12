function build_tsdistances()
%BUILD_TSDISTANCES Build the tsdistances MEX file with dependency checking
%
%   BUILD_TSDISTANCES() compiles the Rust library and builds the MEX file,
%   with automatic dependency checking and installation guidance.
%
%   Requirements:
%       - Rust toolchain (cargo, rustc)
%       - MATLAB with MEX compiler configured
%         (Windows requires Microsoft Visual C++)
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
        % Build the CPU-only MATLAB library without Python or GPU dependencies.
        build_cmd = 'cargo build --release --no-default-features --features matlab 2>&1';
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
    windows_native_link_args = '';

    if ispc && use_static
        native_libs = get_windows_rust_native_link_libs(project_root);
        windows_native_link_args = format_windows_link_args(native_libs);
        if ~isempty(native_libs)
            fprintf('Linking Rust native system libraries: %s\n', strjoin(native_libs, ', '));
        end
    end
    
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
            mex_cmd = sprintf('mex "%s" "%s"%s -I"%s" -outdir "%s"', ...
                              mex_src, lib_file, windows_native_link_args, header_dir, script_dir);
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

function selected_cfg = check_matlab_mex()
    fprintf('\n--- Checking MATLAB MEX ---\n');

    mex_path = which('mex');
    if isempty(mex_path)
        fprintf('ERROR: MATLAB MEX function is not available in this session.\n');
        fprintf('Make sure you are running this script inside MATLAB.\n');
        error('build_tsdistances:MexUnavailable', ...
              'MATLAB mex is unavailable in this session. Run build_tsdistances from inside MATLAB.');
    end

    try
        selected_cfg = mex.getCompilerConfigurations('C', 'Selected');
    catch ME
        fprintf('ERROR: Unable to query MATLAB MEX compiler configuration.\n');
        fprintf('Run from MATLAB: mex -setup C\n');
        fprintf('Underlying error: %s\n', ME.message);
        error('build_tsdistances:MexConfigQueryFailed', ...
              'Unable to query the MATLAB C MEX compiler configuration. Run "mex -setup C".');
    end

    if isempty(selected_cfg)
        fprintf('ERROR: No MATLAB C MEX compiler is configured.\n');
        fprintf('Solution:\n');
        fprintf('  1. Run from MATLAB: mex -setup C\n');
        fprintf('  2. Select a supported C compiler\n');
        if ispc
            fprintf('  3. On Windows, install Visual Studio Build Tools if needed\n');
            fprintf('  4. Choose Microsoft Visual C++ when MATLAB prompts you\n');
        end

        try
            installed_cfg = mex.getCompilerConfigurations('C', 'Installed');
            if isempty(installed_cfg)
                fprintf('MATLAB did not detect any supported installed C compiler.\n');
            else
                fprintf('MATLAB detected these installed C compilers:\n');
                for i = 1:length(installed_cfg)
                    fprintf('  - %s\n', installed_cfg(i).Name);
                end
            end
        catch
            % Ignore secondary detection errors and keep the main guidance concise.
        end

        error('build_tsdistances:MexCompilerNotConfigured', ...
              'No MATLAB C MEX compiler is configured. Run "mex -setup C" and select a supported compiler.');
    end

    fprintf('✓ MEX compiler is available: %s\n', selected_cfg.Name);
end

function native_libs = get_windows_rust_native_link_libs(project_root)
    native_libs = {};
    if ~ispc
        return;
    end

    old_dir = cd(project_root);
    try
        query_cmd = 'cargo rustc --release --lib --no-default-features --features matlab -- --print native-static-libs 2>&1';
        [status, output] = system(query_cmd);
    catch ME
        cd(old_dir);
        rethrow(ME);
    end
    cd(old_dir);

    if status ~= 0
        fprintf('WARNING: Unable to query Rust native link libraries.\n');
        fprintf('Continuing without extra Windows system libraries.\n');
        return;
    end

    output_lines = regexp(output, '\r\n|\n|\r', 'split');
    native_line = '';
    for i = 1:numel(output_lines)
        line = strtrim(output_lines{i});
        if contains(line, 'native-static-libs:')
            native_line = line;
            break;
        end
    end

    if isempty(native_line)
        return;
    end

    match = regexp(native_line, 'native-static-libs:\s*(.*)$', 'tokens', 'once');
    if isempty(match)
        return;
    end

    tokens = regexp(strtrim(match{1}), '\s+', 'split');
    for i = 1:numel(tokens)
        lib_name = strtrim(tokens{i});
        if isempty(lib_name)
            continue;
        end

        lib_name = regexprep(lib_name, '^[,;]+|[,;]+$', '');

        if strncmp(lib_name, '/defaultlib:', 12)
            lib_name = lib_name(13:end);
        elseif strncmp(lib_name, '-l', 2)
            lib_name = lib_name(3:end);
        end

        if length(lib_name) > 4 && strcmpi(lib_name(end-3:end), '.lib')
            lib_name = lib_name(1:end-4);
        end

        if isempty(lib_name) || ...
           isempty(regexp(lib_name, '^[A-Za-z0-9_.-]+$', 'once')) || ...
           any(strcmpi(native_libs, lib_name))
            continue;
        end

        native_libs{end+1} = lib_name; %#ok<AGROW>
    end
end

function link_args = format_windows_link_args(native_libs)
    link_args = '';

    for i = 1:numel(native_libs)
        link_args = sprintf('%s -l%s', link_args, native_libs{i});
    end
end

function check_windows_mex_compiler(selected_cfg)
    compiler_name = char(selected_cfg(1).Name);
    compiler_name_lower = lower(compiler_name);

    if contains(compiler_name_lower, 'microsoft') || ...
       contains(compiler_name_lower, 'visual') || ...
       contains(compiler_name_lower, 'msvc')
        return;
    end

    fprintf('ERROR: MATLAB is configured to use "%s".\n', compiler_name);
    fprintf('Windows builds require Microsoft Visual C++ for MATLAB MEX and Rust to link cleanly.\n');
    fprintf('Run from MATLAB: mex -setup C\n');
    fprintf('Then select a Microsoft Visual C++ compiler.\n');
    error('build_tsdistances:WindowsCompilerMismatch', ...
          'Windows builds require MATLAB to use Microsoft Visual C++. Run "mex -setup C" and select a Microsoft compiler.');
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
    check_matlab_mex();
    
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
    check_matlab_mex();
    
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
    selected_cfg = check_matlab_mex();
    check_windows_mex_compiler(selected_cfg);
    
    % Check for Visual Studio Build Tools or MSVC
    fprintf('\n--- Checking C compiler (MSVC) ---\n');
    [status, ~] = system('cl.exe 2>&1 | findstr /r "Microsoft"');
    if status ~= 0
        fprintf('WARNING: Microsoft Visual C++ was not found on PATH.\n');
        fprintf('Rust built successfully, so this may be harmless in this session.\n');
        fprintf('If linking later fails, install or repair one of the following:\n');
        fprintf('  1. Visual Studio Community Edition (https://visualstudio.microsoft.com/)\n');
        fprintf('  2. Visual Studio Build Tools (https://visualstudio.microsoft.com/downloads/)\n');
        fprintf('  3. Or ensure MSVC is available on PATH / in a Developer Command Prompt\n');
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
