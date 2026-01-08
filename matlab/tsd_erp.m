function D = tsd_erp(X1, X2, band, gap_penalty, parallel)
%TSD_ERP Compute Edit Distance with Real Penalty (ERP) matrix
%
%   D = TSD_ERP(X1) computes the pairwise ERP distance matrix within X1.
%
%   D = TSD_ERP(X1, X2) computes the ERP distance matrix between all pairs
%   of time series in X1 and X2.
%
%   D = TSD_ERP(X1, X2, band, gap_penalty, parallel) specifies additional
%   parameters.
%
%   Inputs:
%       X1 - M x N matrix where each row is a time series
%       X2 - (optional) P x N matrix, or [] for pairwise within X1
%       band - (optional) Sakoe-Chiba band size [0-1], default: 1.0
%       gap_penalty - (optional) penalty for gap insertion, default: 0.0
%       parallel - (optional) boolean to enable parallel computation
%
%   Output:
%       D - Distance matrix
%
%   Reference:
%       Chen, L. et al., "On The Marriage of Lp-norms and Edit Distance", 2004.
%
%   Example:
%       X1 = randn(10, 100);
%       D = tsd_erp(X1, [], 0.5, 0.1);
%
%   See also: TSD_DTW, TSD_LCSS, TSD_TWE

    if nargin < 2
        X2 = [];
    end
    if nargin < 3
        band = 1.0;
    end
    if nargin < 4
        gap_penalty = 0.0;
    end
    if nargin < 5
        parallel = true;
    end
    
    D = tsd_mex('erp', X1, X2, parallel, band, gap_penalty);
end
