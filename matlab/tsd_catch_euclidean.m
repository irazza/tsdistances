function D = tsd_catch_euclidean(X1, X2, parallel)
%TSD_CATCH_EUCLIDEAN Compute Catch22-Euclidean distance matrix
%
%   D = TSD_CATCH_EUCLIDEAN(X1) computes the pairwise Catch22-Euclidean 
%   distance matrix within X1. The time series are first transformed using
%   the Catch22 feature set before computing Euclidean distances.
%
%   D = TSD_CATCH_EUCLIDEAN(X1, X2) computes the distance matrix between
%   all pairs of time series in X1 and X2.
%
%   D = TSD_CATCH_EUCLIDEAN(X1, X2, parallel) optionally enables parallel 
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
%       X1 = randn(10, 100);
%       D = tsd_catch_euclidean(X1);
%
%   See also: TSD_EUCLIDEAN, TSD_DTW

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        parallel = true;
    end
    
    D = tsd_mex('catch_euclidean', X1, X2, parallel);
end
