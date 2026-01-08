function D = tsd_wdtw(X1, X2, band, g, parallel)
%TSD_WDTW Compute Weighted Dynamic Time Warping (WDTW) distance matrix
%
%   D = TSD_WDTW(X1) computes the pairwise WDTW distance matrix within X1.
%   WDTW applies weights that penalize warping based on the phase difference.
%
%   D = TSD_WDTW(X1, X2) computes the WDTW distance matrix between all pairs
%   of time series in X1 and X2.
%
%   D = TSD_WDTW(X1, X2, band, g, parallel) specifies additional parameters.
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
%   Reference:
%       Jeong Y.-S. et al., "Weighted dynamic time warping for time series 
%       classification", 2011.
%
%   Example:
%       X1 = randn(10, 100);
%       D = tsd_wdtw(X1, [], 1.0, 0.1);
%
%   See also: TSD_DTW, TSD_WDDTW, TSD_ADTW

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
    
    D = tsd_mex('wdtw', X1, X2, parallel, band, g);
end
