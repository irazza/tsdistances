function D = tsd_wddtw(X1, X2, band, g, parallel)
%TSD_WDDTW Compute Weighted Derivative DTW (WDDTW) distance matrix
%
%   D = TSD_WDDTW(X1) computes the pairwise WDDTW distance matrix within X1.
%   WDDTW combines derivative transformation with weighted DTW.
%
%   D = TSD_WDDTW(X1, X2) computes the WDDTW distance matrix between all pairs
%   of time series in X1 and X2.
%
%   D = TSD_WDDTW(X1, X2, band, g, parallel) specifies additional parameters.
%
%   Inputs:
%       X1 - M x N matrix where each row is a time series
%       X2 - (optional) P x N matrix, or [] for pairwise within X1
%       band - (optional) Sakoe-Chiba band size [0-1], default: 1.0
%       g - (optional) weight parameter controlling penalty, default: 0.05
%       parallel - (optional) boolean to enable parallel computation
%
%   Output:
%       D - Distance matrix
%
%   Example:
%       X1 = randn(10, 100);
%       D = tsd_wddtw(X1);
%
%   See also: TSD_DDTW, TSD_WDTW, TSD_DTW

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        band = 1.0;
    end
    if nargin < 4
        g = 0.05;
    end
    if nargin < 5
        parallel = true;
    end
    
    D = tsd_mex('wddtw', X1, X2, parallel, band, g);
end
