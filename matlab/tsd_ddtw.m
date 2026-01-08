function D = tsd_ddtw(X1, X2, band, parallel)
%TSD_DDTW Compute Derivative Dynamic Time Warping (DDTW) distance matrix
%
%   D = TSD_DDTW(X1) computes the pairwise DDTW distance matrix within X1.
%   DDTW applies DTW to the derivatives of the time series.
%
%   D = TSD_DDTW(X1, X2) computes the DDTW distance matrix between all pairs
%   of time series in X1 and X2.
%
%   D = TSD_DDTW(X1, X2, band, parallel) specifies additional parameters.
%
%   Inputs:
%       X1 - M x N matrix where each row is a time series
%       X2 - (optional) P x N matrix, or [] for pairwise within X1
%       band - (optional) Sakoe-Chiba band size [0-1], default: 1.0
%       parallel - (optional) boolean to enable parallel computation
%
%   Output:
%       D - Distance matrix
%
%   Reference:
%       Keogh, E. et al., "Derivative Dynamic Time Warping", 2001.
%
%   Example:
%       X1 = randn(10, 100);
%       D = tsd_ddtw(X1);
%
%   See also: TSD_DTW, TSD_WDTW, TSD_WDDTW

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        band = 1.0;
    end
    if nargin < 4
        parallel = true;
    end
    
    D = tsd_mex('ddtw', X1, X2, parallel, band);
end
