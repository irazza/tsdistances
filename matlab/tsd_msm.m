function D = tsd_msm(X1, X2, cost, parallel)
%TSD_MSM Compute Move-Split-Merge (MSM) distance matrix
%
%   D = TSD_MSM(X1) computes the pairwise MSM distance matrix within X1.
%
%   D = TSD_MSM(X1, X2) computes the MSM distance matrix between all pairs
%   of time series in X1 and X2.
%
%   D = TSD_MSM(X1, X2, cost, parallel) specifies additional parameters.
%
%   Inputs:
%       X1 - M x N matrix where each row is a time series
%       X2 - (optional) P x N matrix, or [] for pairwise within X1
%       cost - (optional) cost for split/merge operations, default: 1.0
%       parallel - (optional) boolean to enable parallel computation
%
%   Output:
%       D - Distance matrix
%
%   Reference:
%       Stefan, A. et al., "The Move-Split-Merge Metric for Time Series", 
%       2013.
%
%   Example:
%       X1 = randn(10, 100);
%       D = tsd_msm(X1, [], 0.5);
%
%   See also: TSD_DTW, TSD_ERP, TSD_TWE

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        cost = 1.0;
    end
    if nargin < 4
        parallel = true;
    end
    
    D = tsd_mex('msm', X1, X2, parallel, cost);
end
