function D = tsd_dtw(X1, X2, band, parallel)
%TSD_DTW Compute Dynamic Time Warping (DTW) distance matrix
%
%   D = TSD_DTW(X1) computes the pairwise DTW distance matrix within X1.
%
%   D = TSD_DTW(X1, X2) computes the DTW distance matrix between all pairs
%   of time series in X1 and X2.
%
%   D = TSD_DTW(X1, X2, band, parallel) specifies additional parameters.
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
%       Berndt, D.J. and Clifford, J., "Using Dynamic Time Warping to Find 
%       Patterns in Time Series", 1994.
%
%   Example:
%       X1 = randn(10, 100);
%       D = tsd_dtw(X1);
%       
%       % With 10% Sakoe-Chiba band constraint
%       D = tsd_dtw(X1, [], 0.1);
%
%   See also: TSD_DDTW, TSD_WDTW, TSD_ADTW

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        band = 1.0;
    end
    if nargin < 4
        parallel = true;
    end
    
    D = tsd_mex('dtw', X1, X2, parallel, band);
end
