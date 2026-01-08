function build_tsdistances()
%BUILD_TSDISTANCES Build the tsdistances MEX file
%
%   BUILD_TSDISTANCES() compiles the Rust library and builds the MEX file.
%
%   Requirements:
%       - Rust toolchain (cargo)
%       - MATLAB with MEX compiler configured
%
%   This function will:
%       1. Build the Rust library in release mode (with matlab feature, no PyO3)
%       2. Compile the MEX gateway
%       3. Copy necessary files to the matlab directory

    % Get the directory of this script
    script_dir = fileparts(mfilename('fullpath'));
    project_root = fullfile(script_dir, '..');
    
    fprintf('Building tsdistances MATLAB bindings...\n');
    
    % Step 1: Build Rust library without Python bindings
    fprintf('\n[1/3] Building Rust library...\n');
    old_dir = cd(project_root);
    try
        % Build with matlab feature and without default features (no PyO3)
        [status, result] = system('cargo build --release --no-default-features --features matlab,use-compiled-tools');
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
    fprintf('\n[2/3] Compiling MEX file...\n');
    
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
    fprintf('\n[3/3] Finalizing installation...\n');
    
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
