function D = tsd_euclidean(X1, X2, parallel)
%TSD_EUCLIDEAN Compute Euclidean distance matrix between time series sets
%
%   D = TSD_EUCLIDEAN(X1) computes the pairwise Euclidean distance matrix
%   within X1.
%
%   D = TSD_EUCLIDEAN(X1, X2) computes the Euclidean distance matrix between
%   all pairs of time series in X1 and X2.
%
%   D = TSD_EUCLIDEAN(X1, X2, parallel) optionally enables parallel 
%   computation (default: true).
%
%   Inputs:
%       X1 - M x N matrix where each row is a time series of length N
%       X2 - (optional) P x N matrix where each row is a time series
%       parallel - (optional) boolean to enable parallel computation
%
%   Output:
%       D - Distance matrix (M x M if only X1 provided, M x P otherwise)
%
%   Example:
%       X1 = randn(10, 100);  % 10 time series of length 100
%       D = tsd_euclidean(X1);  % 10x10 pairwise distance matrix
%
%   See also: TSD_DTW, TSD_ERP, TSD_LCSS

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        parallel = true;
    end
    
    D = tsd_mex('euclidean', X1, X2, parallel);
end
