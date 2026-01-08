function D = tsd_mp(X1, X2, window_size, parallel)
%TSD_MP Compute Matrix Profile distance matrix
%
%   D = TSD_MP(X1) computes the pairwise Matrix Profile distance matrix 
%   within X1.
%
%   D = TSD_MP(X1, X2) computes the MP distance matrix between all pairs
%   of time series in X1 and X2.
%
%   D = TSD_MP(X1, X2, window_size, parallel) specifies additional parameters.
%
%   Inputs:
%       X1 - M x N matrix where each row is a time series
%       X2 - (optional) P x N matrix, or [] for pairwise within X1
%       window_size - (optional) subsequence window size, default: N/4
%       parallel - (optional) boolean to enable parallel computation
%
%   Output:
%       D - Distance matrix
%
%   Reference:
%       Yeh, C.-C. M. et al., "Matrix Profile I: All Pairs Similarity Joins
%       for Time Series", 2016.
%
%   Example:
%       X1 = randn(10, 100);
%       D = tsd_mp(X1, [], 25);
%
%   See also: TSD_DTW, TSD_SBD

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        window_size = floor(size(X1, 2) / 4);
    end
    if nargin < 4
        parallel = true;
    end
    
    D = tsd_mex('mp', X1, X2, parallel, window_size);
end
