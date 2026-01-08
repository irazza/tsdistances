%% TSDistances MATLAB Example
% This script demonstrates the usage of tsdistances functions

%% Setup
% Make sure you've run build_tsdistances first
% build_tsdistances

%% Generate sample data
rng(42);  % For reproducibility
n_series = 20;   % Number of time series
ts_length = 100; % Length of each time series

% Create synthetic time series with different patterns
X = zeros(n_series, ts_length);
t = linspace(0, 4*pi, ts_length);

for i = 1:n_series
    freq = 0.5 + rand() * 1.5;
    phase = rand() * 2 * pi;
    noise = 0.1 * randn(1, ts_length);
    X(i, :) = sin(freq * t + phase) + noise;
end

fprintf('Generated %d time series of length %d\n', n_series, ts_length);

%% Euclidean Distance
fprintf('\n=== Euclidean Distance ===\n');
tic;
D_eucl = tsd_euclidean(X);
t_eucl = toc;
fprintf('Time: %.4f seconds\n', t_eucl);
fprintf('Matrix size: %d x %d\n', size(D_eucl, 1), size(D_eucl, 2));
fprintf('Min: %.4f, Max: %.4f, Mean: %.4f\n', min(D_eucl(:)), max(D_eucl(:)), mean(D_eucl(:)));

%% DTW Distance
fprintf('\n=== DTW Distance ===\n');
tic;
D_dtw = tsd_dtw(X);
t_dtw = toc;
fprintf('Time: %.4f seconds\n', t_dtw);
fprintf('Min: %.4f, Max: %.4f, Mean: %.4f\n', min(D_dtw(:)), max(D_dtw(:)), mean(D_dtw(:)));

%% DTW with Sakoe-Chiba Band
fprintf('\n=== DTW with 10%% Sakoe-Chiba Band ===\n');
tic;
D_dtw_band = tsd_dtw(X, [], 0.1);
t_dtw_band = toc;
fprintf('Time: %.4f seconds (%.1fx speedup)\n', t_dtw_band, t_dtw/t_dtw_band);

%% Derivative DTW
fprintf('\n=== Derivative DTW ===\n');
tic;
D_ddtw = tsd_ddtw(X);
t_ddtw = toc;
fprintf('Time: %.4f seconds\n', t_ddtw);
fprintf('Min: %.4f, Max: %.4f, Mean: %.4f\n', min(D_ddtw(:)), max(D_ddtw(:)), mean(D_ddtw(:)));

%% ERP Distance
fprintf('\n=== ERP Distance ===\n');
tic;
D_erp = tsd_erp(X, [], 1.0, 0.0);
t_erp = toc;
fprintf('Time: %.4f seconds\n', t_erp);
fprintf('Min: %.4f, Max: %.4f, Mean: %.4f\n', min(D_erp(:)), max(D_erp(:)), mean(D_erp(:)));

%% LCSS Distance
fprintf('\n=== LCSS Distance ===\n');
tic;
D_lcss = tsd_lcss(X, [], 1.0, 0.5);
t_lcss = toc;
fprintf('Time: %.4f seconds\n', t_lcss);
fprintf('Min: %.4f, Max: %.4f, Mean: %.4f\n', min(D_lcss(:)), max(D_lcss(:)), mean(D_lcss(:)));

%% MSM Distance
fprintf('\n=== MSM Distance ===\n');
tic;
D_msm = tsd_msm(X);
t_msm = toc;
fprintf('Time: %.4f seconds\n', t_msm);
fprintf('Min: %.4f, Max: %.4f, Mean: %.4f\n', min(D_msm(:)), max(D_msm(:)), mean(D_msm(:)));

%% Shape-Based Distance
fprintf('\n=== Shape-Based Distance ===\n');
tic;
D_sbd = tsd_sbd(X);
t_sbd = toc;
fprintf('Time: %.4f seconds\n', t_sbd);
fprintf('Min: %.4f, Max: %.4f, Mean: %.4f\n', min(D_sbd(:)), max(D_sbd(:)), mean(D_sbd(:)));

%% Compare two different sets
fprintf('\n=== Computing distance between two sets ===\n');
X1 = X(1:10, :);
X2 = X(11:20, :);
tic;
D_cross = tsd_dtw(X1, X2);
t_cross = toc;
fprintf('Time: %.4f seconds\n', t_cross);
fprintf('Matrix size: %d x %d\n', size(D_cross, 1), size(D_cross, 2));

%% Visualization
figure('Position', [100, 100, 1200, 800]);

subplot(2, 3, 1);
imagesc(D_eucl);
colorbar;
title('Euclidean Distance');
xlabel('Time Series Index');
ylabel('Time Series Index');

subplot(2, 3, 2);
imagesc(D_dtw);
colorbar;
title('DTW Distance');
xlabel('Time Series Index');
ylabel('Time Series Index');

subplot(2, 3, 3);
imagesc(D_ddtw);
colorbar;
title('Derivative DTW');
xlabel('Time Series Index');
ylabel('Time Series Index');

subplot(2, 3, 4);
imagesc(D_erp);
colorbar;
title('ERP Distance');
xlabel('Time Series Index');
ylabel('Time Series Index');

subplot(2, 3, 5);
imagesc(D_lcss);
colorbar;
title('LCSS Distance');
xlabel('Time Series Index');
ylabel('Time Series Index');

subplot(2, 3, 6);
imagesc(D_sbd);
colorbar;
title('Shape-Based Distance');
xlabel('Time Series Index');
ylabel('Time Series Index');

sgtitle('TSDistances - Distance Matrix Comparison');

%% Benchmark: Parallel vs Sequential
fprintf('\n=== Parallel vs Sequential Benchmark ===\n');

% Larger dataset for benchmarking
X_large = randn(50, 200);

tic;
D_par = tsd_dtw(X_large, [], 1.0, true);
t_par = toc;
fprintf('Parallel: %.4f seconds\n', t_par);

tic;
D_seq = tsd_dtw(X_large, [], 1.0, false);
t_seq = toc;
fprintf('Sequential: %.4f seconds\n', t_seq);
fprintf('Speedup: %.2fx\n', t_seq / t_par);

% Verify results are the same
max_diff = max(abs(D_par(:) - D_seq(:)));
fprintf('Max difference: %.2e (should be ~0)\n', max_diff);

fprintf('\n=== All tests completed successfully! ===\n');
