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
%       1. Build the Rust library in release mode
%       2. Compile the MEX gateway
%       3. Copy necessary files to the matlab directory

    % Get the directory of this script
    script_dir = fileparts(mfilename('fullpath'));
    project_root = fullfile(script_dir, '..');
    
    fprintf('Building tsdistances MATLAB bindings...\n');
    
    % Step 1: Build Rust library
    fprintf('\n[1/3] Building Rust library...\n');
    old_dir = cd(project_root);
    try
        [status, result] = system('cargo build --release --features matlab');
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
        lib_name = 'libtsdistances_matlab.dylib';
        lib_ext = '.dylib';
    elseif isunix
        lib_name = 'libtsdistances_matlab.so';
        lib_ext = '.so';
    elseif ispc
        lib_name = 'tsdistances_matlab.dll';
        lib_ext = '.dll';
    else
        error('Unsupported platform');
    end
    
    lib_path = fullfile(project_root, 'target', 'release');
    lib_file = fullfile(lib_path, lib_name);
    
    if ~exist(lib_file, 'file')
        % Try static library
        if ismac || isunix
            lib_name = 'libtsdistances_matlab.a';
        else
            lib_name = 'tsdistances_matlab.lib';
        end
        lib_file = fullfile(lib_path, lib_name);
    end
    
    if ~exist(lib_file, 'file')
        error('Could not find compiled library at: %s', lib_path);
    end
    
    fprintf('Found library: %s\n', lib_file);
    
    % Step 3: Compile MEX file
    fprintf('\n[2/3] Compiling MEX file...\n');
    
    mex_src = fullfile(script_dir, 'tsd_mex.c');
    
    % Build MEX command
    if ismac
        mex_cmd = sprintf('mex -R2018a "%s" -L"%s" -ltsdistances_matlab -outdir "%s"', ...
                          mex_src, lib_path, script_dir);
        % Add rpath for dynamic library
        mex_cmd = sprintf('%s LDFLAGS="$LDFLAGS -Wl,-rpath,''%s''"', mex_cmd, lib_path);
    elseif isunix
        mex_cmd = sprintf('mex -R2018a "%s" -L"%s" -ltsdistances_matlab -outdir "%s"', ...
                          mex_src, lib_path, script_dir);
        mex_cmd = sprintf('%s LDFLAGS="$LDFLAGS -Wl,-rpath,''%s''"', mex_cmd, lib_path);
    else % Windows
        mex_cmd = sprintf('mex "%s" -L"%s" -ltsdistances_matlab -outdir "%s"', ...
                          mex_src, lib_path, script_dir);
    end
    
    fprintf('Running: %s\n', mex_cmd);
    eval(mex_cmd);
    
    fprintf('MEX file compiled successfully.\n');
    
    % Step 4: Copy dynamic library if needed
    fprintf('\n[3/3] Finalizing installation...\n');
    
    dest_lib = fullfile(script_dir, lib_name);
    if ~strcmp(lib_file, dest_lib)
        copyfile(lib_file, dest_lib);
        fprintf('Copied library to: %s\n', dest_lib);
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
