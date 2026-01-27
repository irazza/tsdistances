function D = tsd_twe(X1, X2, band, stiffness, penalty, parallel)
%TSD_TWE Compute Time Warp Edit (TWE) distance matrix
%
%   D = TSD_TWE(X1) computes the pairwise TWE distance matrix within X1.
%
%   D = TSD_TWE(X1, X2) computes the TWE distance matrix between all pairs
%   of time series in X1 and X2.
%
%   D = TSD_TWE(X1, X2, band, stiffness, penalty, parallel) specifies
%   additional parameters.
%
%   Inputs:
%       X1 - M x N matrix where each row is a time series
%       X2 - (optional) P x N matrix, or [] for pairwise within X1
%       band - (optional) Sakoe-Chiba band size [0-1], default: 1.0
%       stiffness - (optional) stiffness parameter (nu), default: 1.0
%       penalty - (optional) edit penalty (lambda), default: 1.0
%       parallel - (optional) boolean to enable parallel computation
%
%   Output:
%       D - Distance matrix
%
%   Reference:
%       Marteau, P.-F., "Time Warp Edit Distance with Stiffness Adjustment
%       for Time Series Matching", 2009.
%
%   Example:
%       X1 = randn(10, 100);
%       D = tsd_twe(X1, [], 1.0, 0.5, 0.5);
%
%   See also: TSD_DTW, TSD_ERP, TSD_MSM

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        band = 1.0;
    end
    if nargin < 4
        stiffness = 1.0;
    end
    if nargin < 5
        penalty = 1.0;
    end
    if nargin < 6
        parallel = true;
    end
    
    D = tsd_mex('twe', X1, X2, parallel, band, stiffness, penalty);
end
