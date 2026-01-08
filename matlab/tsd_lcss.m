function D = tsd_lcss(X1, X2, band, epsilon, parallel)
%TSD_LCSS Compute Longest Common Subsequence (LCSS) distance matrix
%
%   D = TSD_LCSS(X1) computes the pairwise LCSS distance matrix within X1.
%
%   D = TSD_LCSS(X1, X2) computes the LCSS distance matrix between all pairs
%   of time series in X1 and X2.
%
%   D = TSD_LCSS(X1, X2, band, epsilon, parallel) specifies additional
%   parameters.
%
%   Inputs:
%       X1 - M x N matrix where each row is a time series
%       X2 - (optional) P x N matrix, or [] for pairwise within X1
%       band - (optional) Sakoe-Chiba band size [0-1], default: 1.0
%       epsilon - (optional) matching threshold, default: 1.0
%       parallel - (optional) boolean to enable parallel computation
%
%   Output:
%       D - Distance matrix (values in [0, 1])
%
%   Reference:
%       Vlachos, M. et al., "Discovering Similar Multidimensional 
%       Trajectories", 2002.
%
%   Example:
%       X1 = randn(10, 100);
%       D = tsd_lcss(X1, [], 1.0, 0.5);
%
%   See also: TSD_DTW, TSD_ERP

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        band = 1.0;
    end
    if nargin < 4
        epsilon = 1.0;
    end
    if nargin < 5
        parallel = true;
    end
    
    D = tsd_mex('lcss', X1, X2, parallel, band, epsilon);
end
