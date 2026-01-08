function D = tsd_adtw(X1, X2, band, warp_penalty, parallel)
%TSD_ADTW Compute Amerced Dynamic Time Warping (ADTW) distance matrix
%
%   D = TSD_ADTW(X1) computes the pairwise ADTW distance matrix within X1.
%   ADTW adds a penalty for warping to discourage excessive time distortion.
%
%   D = TSD_ADTW(X1, X2) computes the ADTW distance matrix between all pairs
%   of time series in X1 and X2.
%
%   D = TSD_ADTW(X1, X2, band, warp_penalty, parallel) specifies additional
%   parameters.
%
%   Inputs:
%       X1 - M x N matrix where each row is a time series
%       X2 - (optional) P x N matrix, or [] for pairwise within X1
%       band - (optional) Sakoe-Chiba band size [0-1], default: 1.0
%       warp_penalty - (optional) penalty for warping, default: 1.0
%       parallel - (optional) boolean to enable parallel computation
%
%   Output:
%       D - Distance matrix
%
%   Example:
%       X1 = randn(10, 100);
%       D = tsd_adtw(X1, [], 1.0, 0.5);
%
%   See also: TSD_DTW, TSD_WDTW

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        band = 1.0;
    end
    if nargin < 4
        warp_penalty = 1.0;
    end
    if nargin < 5
        parallel = true;
    end
    
    D = tsd_mex('adtw', X1, X2, parallel, band, warp_penalty);
end
