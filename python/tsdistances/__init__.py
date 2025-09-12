from typing import List, Optional, Union
from typeguard import TypeCheckError, typechecked, check_type
from tsdistances import tsdistances as tsd
import numpy as np


def check_input(
    u: np.ndarray, v: Optional[np.ndarray] = None
) -> Union[np.ndarray, Optional[np.ndarray]]:
    if u.ndim == 1:
        if v is None:
            raise ValueError("If `u` is 1-D, `v` must be not None.")
        if v.ndim == 1:
            _u = u.reshape((1, u.shape[0]))
            _v = v.reshape((1, v.shape[0]))
            return _u, _v
        elif v.ndim == 2:
            _u = u.reshape((1, u.shape[0]))
            _v = v
            return _u, _v
        else:
            raise ValueError("`v` must be 1-D or 2-D.")
    elif u.ndim == 2:
        _u = u
        if v is None:
            return _u, None
        else:
            if v.ndim == 1:
                _v = v.reshape((1, v.shape[0]))
                return _u, _v
            elif v.ndim == 2:
                _v = v
                return _u, _v
            else:
                raise ValueError("`v` must be 1-D or 2-D.")
    else:
        raise ValueError("`u` must be 1-D or 2-D.")


@typechecked
def euclidean_distance(
    u: np.ndarray, v: Optional[np.ndarray] = None, par: Optional[int] = 1
) -> Union[np.ndarray, float]:
    """
    Computes the Euclidean distance between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise Euclidean distances within `u`.

    Parameters
    ----------
    u : (N,) array_like or (M, N) array_like
        Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N) array_like, optional
    Input array. If provided, `v` should have the same shape as `u`.
    If `v` is None, pairwise distances within `u` are computed.
    par : int, optional
        Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
        The Euclidean distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> euclidean_distance([1, 0, 0], [0, 1, 0])
    1.4142135623730951
    >>> euclidean_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[1.41421356, 2.44948974],
           [1.        , 1.73205081]])
    >>> euclidean_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.        , 1.        ],
           [1.        , 0.        ]])

    """
    if u.ndim == 1 and v.ndim == 1:
        _u = u.reshape((1, u.shape[0]))
        _v = v.reshape((1, v.shape[0]))

    if u.ndim == 2:
        _u = u
        if v is None:
            return np.array(tsd.euclidean(_u, None, par))
        if v.ndim == 2:
            _v = v

    return np.array(tsd.euclidean(_u, _v, par))


@typechecked
def catcheucl_distance(
    u: np.ndarray, v: Optional[np.ndarray] = None, par: Optional[int] = 1
) -> Union[np.ndarray, float]:
    """
    Computes the Catch22-Euclidean distance between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise Catch22-Euclidean distances within `u`.

    Parameters
    ----------
    u : (N,) array_like or (M, N) array_like
        Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N) array_like, optional
    Input array. If provided, `v` should have the same shape as `u`.
    If `v` is None, pairwise distances within `u` are computed.
    par : int, optional
        Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
        The Catch22-Euclidean distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> catcheucl_distance([1, 0, 0], [0, 1, 0])
    1.0
    >>> catcheucl_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[1.0, 2.0], [1.0, 1.0]])
    >>> catcheucl_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 1.0], [1.0, 0.0]])

    """
    if u.ndim == 1 and v.ndim == 1:
        _u = u.reshape((1, u.shape[0]))
        _v = v.reshape((1, v.shape[0]))

    if u.ndim == 2:
        _u = u
        if v is None:
            return np.array(tsd.catch_euclidean(_u, None, par))
        if v.ndim == 2:
            _v = v

    return np.array(tsd.catch_euclidean(_u, _v, par))


@typechecked
def erp_distance(
    u: np.ndarray,
    v: Optional[np.ndarray] = None,
    band: Optional[float] = 1.0,
    gap_penalty: Optional[float] = 1.0,
    par: Optional[bool] = True,
    device: Optional[str] = "cpu",
) -> Union[np.ndarray, float]:
    """
    Computes the Edit Distance with Real Penalty (ERP) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise ERP distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Chen, L. et al., On The Marriage of Lp-norms and Edit Distance, 2004.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array.
    If `v` is None, pairwise distances within `u` are computed.
    band : double, optional
    Band size for the Sakoe-Chiba dynamic programming algorithm (default is 1.0).
    gap_penalty : double, optional
    Penalty for gap insertion/deletion (default is 0.0).
    par : int, optional
    Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
        The ERP distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> erp_distance([1, 0, 0], [0, 1, 0])
    2.0
    >>> erp_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[2.0, 4.0], [1.0, 3.0]])
    >>> erp_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 1.0], [1.0, 0.0]])
    """
    _u, _v = check_input(u, v)

    if _u.shape[0] == 1:
        if _v.shape[0] == 1:
            return tsd.erp(_u, _v, band, gap_penalty, par, device)[0][0]
        elif v.shape[0] >= 2:
            return np.array(tsd.erp(_u, _v, band, gap_penalty, par, device))
    elif _u.shape[0] >= 2:
        if _v is None:
            return np.array(tsd.erp(_u, _v, band, gap_penalty, par, device))
        else:
            if _v.shape[0] == 1:
                return np.array(tsd.erp(_u, _v, band, gap_penalty, par, device))
            elif _v.shape[0] >= 2:
                return np.array(tsd.erp(_u, _v, band, gap_penalty, par, device))


@typechecked
def lcss_distance(
    u: np.ndarray,
    v: Optional[np.ndarray] = None,
    band: Optional[float] = 1.0,
    epsilon: Optional[float] = 1.0,
    par: Optional[bool] = True,
    device: Optional[str] = "cpu",
) -> Union[np.ndarray, float]:
    """
    Computes the Longest Common Subsequence (LCSS) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise LCSS distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Vlachos, M. et al., Discovering Similar Multidimensional Trajectories, 2002.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array.
    If `v` is None, pairwise distances within `u` are computed.
    band : double, optional
    Band size for the Sakoe-Chiba dynamic programming algorithm (default is 1.0).
    epsilon : double, optional
    Threshold value for the distance between two elements (default is 1.0).
    par : int, optional

    Returns
    -------
    distance : double or ndarray
        The LCSS distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> lcss_distance([1, 0, 0], [0, 1, 0], epsilon=0.5)
    0.3333333333333333
    >>> lcss_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]], epsilon=0.5)
    array([[0.3333333333333333, 0.6666666666666666], [0.0, 0.3333333333333333]])
    >>> lcss_distance([[1, 1, 1], [0, 1, 1]], epsilon=0.5)
    array([[0.0, 0.3333333333333333], [0.3333333333333333, 0.0]])
    """
    _u, _v = check_input(u, v)

    if _u.shape[0] == 1:
        if _v.shape[0] == 1:
            return tsd.lcss(_u, _v, band, epsilon, par, device)[0][0]
        elif v.shape[0] >= 2:
            return np.array(tsd.lcss(_u, _v, band, epsilon, par, device))
    elif _u.shape[0] >= 2:
        if _v is None:
            return np.array(tsd.lcss(_u, _v, band, epsilon, par, device))
        else:
            if _v.shape[0] == 1:
                return np.array(tsd.lcss(_u, _v, band, epsilon, par, device))
            elif _v.shape[0] >= 2:
                return np.array(tsd.lcss(_u, _v, band, epsilon, par, device))


@typechecked
def dtw_distance(
    u: np.ndarray,
    v: Optional[np.ndarray] = None,
    band: Optional[float] = 1.0,
    par: Optional[bool] = True,
    device: Optional[str] = "cpu",
) -> Union[np.ndarray, float]:
    """
    Computes the Dynamic Time Warping (DTW) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise DTW distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Berndt, D.J. and Clifford, J., Using Dynamic Time Warping to Find Patterns in Time Series, 1994.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array.
    If `v` is None, pairwise distances within `u` are computed.
    band : double, optional
    Band size for the Sakoe-Chiba dynamic programming algorithm (default is 1.0).
    par : int, optional
    Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
    The DTW distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> dtw_distance([1, 0, 0], [0, 1, 0])
    1.0
    >>> dtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[2.0, 6.0], [1.0, 3.0]])
    >>> dtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.        , 1.        ], [1.        , 0.        ]])
    """
    _u, _v = check_input(u, v)

    if _u.shape[0] == 1:
        if _v.shape[0] == 1:
            return tsd.dtw(_u, _v, band, par, device)[0][0]
        elif v.shape[0] >= 2:
            return np.array(tsd.dtw(_u, _v, band, par, device))
    elif _u.shape[0] >= 2:
        if _v is None:
            return np.array(tsd.dtw(_u, _v, band, par, device))
        else:
            if _v.shape[0] == 1:
                return np.array(tsd.dtw(_u, _v, band, par, device))
            elif _v.shape[0] >= 2:
                return np.array(tsd.dtw(_u, _v, band, par, device))


@typechecked
def ddtw_distance(
    u: np.ndarray,
    v: Optional[np.ndarray] = None,
    band: Optional[float] = 1.0,
    par: Optional[bool] = True,
    device: Optional[str] = "cpu",
) -> Union[np.ndarray, float]:
    """
    Computes the Derivative Dynamic Time Warping (DDTW) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise DDTW distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Keogh, E. et al., Derivative Dynamic Time Warping, 2001.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array.
    If `v` is None, pairwise distances within `u` are computed.
    band : double, optional
    Band size for the Sakoe-Chiba dynamic programming algorithm (default is 1.0).
    par : int, optional
    Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
    The DDTW distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> ddtw_distance([1, 0, 0], [0, 1, 0])
    4.6875
    >>> ddtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[0.75, 1.6875], [0.1875, 0.0]])
    >>> ddtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 1.6875], [1.6875, 0.0]])
    """
    _u, _v = check_input(u, v)

    if _u.shape[0] == 1:
        if _v.shape[0] == 1:
            return tsd.ddtw(_u, _v, band, par, device)[0][0]
        elif v.shape[0] >= 2:
            return np.array(tsd.ddtw(_u, _v, band, par, device))
    elif _u.shape[0] >= 2:
        if _v is None:
            return np.array(tsd.ddtw(_u, _v, band, par, device))
        else:
            if _v.shape[0] == 1:
                return np.array(tsd.ddtw(_u, _v, band, par, device))
            elif _v.shape[0] >= 2:
                return np.array(tsd.ddtw(_u, _v, band, par, device))


@typechecked
def wdtw_distance(
    u: np.ndarray,
    v: Optional[np.ndarray] = None,
    band: Optional[float] = 1.0,
    g: Optional[float] = 0.05,
    par: Optional[bool] = True,
    device: Optional[str] = "cpu",
) -> Union[np.ndarray, float]:
    """
    Computes the Weighted Dynamic Time Warping (WDTW) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise WDTW distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Jeong Y.-S. et al., Weighted dynamic time warping for time series classification, 2011.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array.
    If `v` is None, pairwise distances within `u` are computed.
    band : double, optional
    Band size for the Sakoe-Chiba dynamic programming algorithm (default is 1.0).
    par : int, optional
    Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
    The WDTW distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> wdtw_distance([1, 0, 0], [0, 1, 0])
    0.18242552380635635
    >>> wdtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[0.3648510476127127, 1.094553142838138], [0.18242552380635635, 0.547276571419069]])
    >>> wdtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 0.18242552380635635], [0.18242552380635635, 0.0]])
    """
    _u, _v = check_input(u, v)

    if _u.shape[0] == 1:
        if _v.shape[0] == 1:
            return tsd.wdtw(_u, _v, band, g, par, device)[0][0]
        elif v.shape[0] >= 2:
            return np.array(tsd.wdtw(_u, _v, band, g, par, device))
    elif _u.shape[0] >= 2:
        if _v is None:
            return np.array(tsd.wdtw(_u, _v, band, g, par, device))
        else:
            if _v.shape[0] == 1:
                return np.array(tsd.wdtw(_u, _v, band, g, par, device))
            elif _v.shape[0] >= 2:
                return np.array(tsd.wdtw(_u, _v, band, g, par, device))


@typechecked
def wddtw_distance(
    u: np.ndarray,
    v: Optional[np.ndarray] = None,
    band: Optional[float] = 1.0,
    g: Optional[float] = 0.05,
    par: Optional[bool] = True,
    device: Optional[str] = "cpu",
) -> Union[np.ndarray, float]:
    """
    Computes the Weighted Derivative Dynamic Time Warping (WDDTW) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise WDDTW distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Jeong, Y.-S. et al., Weighted dynamic time warping for time series classification, 2011.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array.
    If `v` is None, pairwise distances within `u` are computed.
    band : double, optional
    Band size for the Sakoe-Chiba dynamic programming algorithm (default is 1.0).
    par : int, optional

    Returns
    -------
    distance : double or ndarray
    The WDDTW distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> wddtw_distance([1, 0, 0], [0, 1, 0])
    0.8551196428422955
    >>> wddtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[0.13681914285476726, 0.3078430714232263], [0.034204785713691815, 0.0]])
    >>> wddtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 0.3078430714232263], [0.3078430714232263, 0.0]])
    """
    _u, _v = check_input(u, v)

    if _u.shape[0] == 1:
        if _v.shape[0] == 1:
            return tsd.wddtw(_u, _v, band, g, par, device)[0][0]
        elif v.shape[0] >= 2:
            return np.array(tsd.wddtw(_u, _v, band, g, par, device))
    elif _u.shape[0] >= 2:
        if _v is None:
            return np.array(tsd.wddtw(_u, _v, band, g, par, device))
        else:
            if _v.shape[0] == 1:
                return np.array(tsd.wddtw(_u, _v, band, g, par, device))
            elif _v.shape[0] >= 2:
                return np.array(tsd.wddtw(_u, _v, band, g, par, device))


@typechecked
def adtw_distance(
    u: np.ndarray,
    v: Optional[np.ndarray] = None,
    band: Optional[float] = 1.0,
    warp_penalty: Optional[float] = 1.0,
    par: Optional[bool] = True,
    device: Optional[str] = "cpu",
) -> Union[np.ndarray, float]:
    """
    Computes the Amercing Dynamic Time Warping (ADTW) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise ADTW distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Hermann, M. et al., Amercing: An intuitive and effective constraint for dynamic time warping, 2023

    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array.
    If `v` is None, pairwise distances within `u` are computed.
    band : double, optional
    Band size for the Sakoe-Chiba dynamic programming algorithm (default is 1.0).
    w : double, optional
    Weight amercing penalty (default is 0.1).
    par : int, optional
    Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
    The ADTW distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> adtw_distance([1, 0, 0], [0, 1, 0])
    0.0
    >>> adtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[0.0, 0.0], [0.0, 0.0]])
    >>> adtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 0.0], [0.0, 0.0]])
    """
    _u, _v = check_input(u, v)

    if _u.shape[0] == 1:
        if _v.shape[0] == 1:
            return tsd.adtw(_u, _v, band, warp_penalty, par, device)[0][0]
        elif v.shape[0] >= 2:
            return np.array(tsd.adtw(_u, _v, band, warp_penalty, par, device))
    elif _u.shape[0] >= 2:
        if _v is None:
            return np.array(tsd.adtw(_u, _v, band, warp_penalty, par, device))
        else:
            if _v.shape[0] == 1:
                return np.array(tsd.adtw(_u, _v, band, warp_penalty, par, device))
            elif _v.shape[0] >= 2:
                return np.array(tsd.adtw(_u, _v, band, warp_penalty, par, device))


@typechecked
def msm_distance(
    u: np.ndarray,
    v: Optional[np.ndarray] = None,
    band: Optional[float] = 1.0,
    par: Optional[bool] = True,
    device: Optional[str] = "cpu",
) -> Union[np.ndarray, float]:
    """
    Computes the Move-Split-Merge (MSM) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise MSM distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Stefan, A. et al., The Move-Split-Merge Metric for Time Series, 2012.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array.
    If `v` is None, pairwise distances within `u` are computed.
    band : double, optional
    Band size for the Sakoe-Chiba dynamic programming algorithm (default is 1.0).
    par : int, optional
    Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
    The MSM distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> msm_distance([1, 0, 0], [0, 1, 0])
    2.0
    >>> msm_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[2.0, 4.0], [1.0, 3.0]])
    >>> msm_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 1.0], [1.0, 0.0]])
    """
    _u, _v = check_input(u, v)

    if _u.shape[0] == 1:
        if _v.shape[0] == 1:
            return tsd.msm(_u, _v, band, par, device)[0][0]
        elif v.shape[0] >= 2:
            return np.array(tsd.msm(_u, _v, band, par, device))
    elif _u.shape[0] >= 2:
        if _v is None:
            return np.array(tsd.msm(_u, _v, band, par, device))
        else:
            if _v.shape[0] == 1:
                return np.array(tsd.msm(_u, _v, band, par, device))
            elif _v.shape[0] >= 2:
                return np.array(tsd.msm(_u, _v, band, par, device))


@typechecked
def twe_distance(
    u: np.ndarray,
    v: Optional[np.ndarray] = None,
    band: Optional[float] = 1.0,
    stifness: Optional[float] = 0.001,
    penalty: Optional[float] = 1.0,
    par: Optional[bool] = True,
    device: Optional[str] = "cpu",
) -> Union[np.ndarray, float]:
    """
    Computes the Time Warp Edit (TWE) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise TWE distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Marteau, P., Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching, 2008.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array.
    If `v` is None, pairwise distances within `u` are computed.
    band : double, optional
    Band size for the Sakoe-Chiba dynamic programming algorithm (default is 1.0).
    stifness : double, optional
    Elasticity parameter (default is 0.001).
    penalty : double, optional
    Penalty for gap insertion/deletion (default is 1.0).
    par : int, optional
    Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
    The TWE distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> twe_distance([1, 0, 0], [0, 1, 0])
    4.0
    >>> twe_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[3.0, 7.0], [1.0, 5.0]])
    >>> twe_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.0, 2.0], [2.0, 0.0]])
    """
    _u, _v = check_input(u, v)

    if _u.shape[0] == 1:
        if _v.shape[0] == 1:
            return tsd.twe(_u, _v, band, stifness, penalty, par, device)[0][0]
        elif v.shape[0] >= 2:
            return np.array(tsd.twe(_u, _v, band, stifness, penalty, par, device))
    elif _u.shape[0] >= 2:
        if _v is None:
            return np.array(tsd.twe(_u, _v, band, stifness, penalty, par, device))
        else:
            if _v.shape[0] == 1:
                return np.array(tsd.twe(_u, _v, band, stifness, penalty, par, device))
            elif _v.shape[0] >= 2:
                return np.array(tsd.twe(_u, _v, band, stifness, penalty, par, device))


@typechecked
def sb_distance(
    u: np.ndarray,
    v: Optional[np.ndarray] = None,
    par: Optional[bool] = True,
) -> Union[np.ndarray, float]:
    """
    Computes the Shape-Based Distance (SBD) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise SBD distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Paparrizos, J. et al., k-Shape: Efficient and Accurate Clustering of Time Series, 2015.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    v : (N,) array_like or (M, N), optional
    Input array.
    If `v` is None, pairwise distances within `u` are computed.
    par : int, optional
    Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
    The SBD distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> sb_distance([1, 0, 0], [0, 1, 0])
    1.4142135623730951
    >>> sb_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[1.41421356, 2.44948974], [1.        , 1.73205081]])
    >>> sb_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.        , 1.        ], [1.        , 0.        ]])
    """
    _u, _v = check_input(u, v)

    if _u.shape[0] == 1:
        if _v.shape[0] == 1:
            return tsd.sb(_u, _v, par)[0][0]
        elif v.shape[0] >= 2:
            return np.array(tsd.sb(_u, _v, par))
    elif _u.shape[0] >= 2:
        if _v is None:
            return np.array(tsd.sb(_u, _v, par))
        else:
            if _v.shape[0] == 1:
                return np.array(tsd.sb(_u, _v, par))
            elif _v.shape[0] >= 2:
                return np.array(tsd.sb(_u, _v, par))


def mp_distance(
    u: np.ndarray,
    window: Optional[int] = 20,
    v: Optional[np.ndarray] = None,
    par: Optional[bool] = True,
):
    """
    Computes the Matrix Profile distance (MPdist) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise MP distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Gharghabi S. et al., An Ultra-Fast Time Series Distance Measure to allow Data Mining in more Complex Real-World Deployments, 2020.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
    Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.

    v : (N,) array_like or (M, N), optional
    Input array.
    If `v` is None, pairwise distances within `u` are computed.

    window : int, optional
    Window size for the Matrix Profile calculation (default is 1).

    par : int, optional
    Number of jobs to use for computation (default is 1).

    Returns
    -------
    distance : double or ndarray
    The MP distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> mp_distance([1, 0, 0], [0, 1, 0])
    1.4142135623730951
    >>> mp_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[1.41421356, 2.44948974], [1.        , 1.73205081]])
    >>> mp_distance([[1, 1, 1], [0, 1, 1]])
    p array([[0.        , 1.        ], [1.        , 0.        ]])
    """
    _u, _v = check_input(u, v)

    if _u.shape[0] == 1:
        if _v.shape[0] == 1:
            return tsd.mp(_u, window, _v, par)[0][0]
        elif v.shape[0] >= 2:
            return np.array(tsd.mp(_u, window, _v, par))
    elif _u.shape[0] >= 2:
        if _v is None:
            return np.array(tsd.mp(_u, window, _v, par))
        else:
            if _v.shape[0] == 1:
                return np.array(tsd.mp(_u, window, _v, par))
            elif _v.shape[0] >= 2:
                return np.array(tsd.mp(_u, window, _v, par))


__all__ = [
    "euclidean",
    "catch_euclidean",
    "erp",
    "lcss",
    "dtw",
    "ddtw",
    "wdtw",
    "wddtw",
    "adtw",
    "msm",
    "twe",
    "sb",
    "mp",
]
