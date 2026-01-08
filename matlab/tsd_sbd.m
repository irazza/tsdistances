function D = tsd_sbd(X1, X2, parallel)
%TSD_SBD Compute Shape-Based Distance (SBD) matrix
%
%   D = TSD_SBD(X1) computes the pairwise SBD distance matrix within X1.
%   SBD is based on normalized cross-correlation and is shift-invariant.
%
%   D = TSD_SBD(X1, X2) computes the SBD distance matrix between all pairs
%   of time series in X1 and X2.
%
%   D = TSD_SBD(X1, X2, parallel) optionally enables parallel computation.
%
%   Inputs:
%       X1 - M x N matrix where each row is a time series
%       X2 - (optional) P x N matrix, or [] for pairwise within X1
%       parallel - (optional) boolean to enable parallel computation
%
%   Output:
%       D - Distance matrix (values in [0, 2])
%
%   Reference:
%       Paparrizos, J. and Gravano, L., "k-Shape: Efficient and Accurate
%       Clustering of Time Series", 2015.
%
%   Example:
%       X1 = randn(10, 100);
%       D = tsd_sbd(X1);
%
%   See also: TSD_EUCLIDEAN, TSD_DTW

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        parallel = true;
    end
    
    D = tsd_mex('sbd', X1, X2, parallel);
end
