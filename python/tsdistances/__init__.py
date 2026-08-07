from typing import Optional, Union
from numpy.typing import ArrayLike
from typeguard import TypeCheckError, typechecked, check_type
from tsdistances import tsdistances as tsd
import numpy as np


def check_input(
    u: ArrayLike, v: Optional[ArrayLike] = None
) -> Union[np.ndarray, Optional[np.ndarray]]:
    u = np.asarray(u)
    v = None if v is None else np.asarray(v)
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


def _to_backend_inputs(
    u: np.ndarray, v: Optional[np.ndarray]
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    # Hand the Rust extension contiguous float64 arrays so it can borrow the
    # buffers zero-copy. This is a no-op when the input is already C-contiguous
    # float64; otherwise it materializes a single copy (e.g. int dtype or a
    # non-contiguous view).
    u = np.ascontiguousarray(u, dtype=np.float64)
    v = None if v is None else np.ascontiguousarray(v, dtype=np.float64)
    return u, v


@typechecked
def euclidean_distance(
    u: ArrayLike, v: Optional[ArrayLike] = None, par: Optional[bool] = True
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
    par : bool, optional
        Enable parallel computation (default is True).

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
    _u, _v = check_input(u, v)
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.euclidean(_u_backend, _v_backend, par)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def catcheucl_distance(
    u: ArrayLike, v: Optional[ArrayLike] = None, par: Optional[bool] = True
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
    par : bool, optional
        Enable parallel computation (default is True).

    Returns
    -------
    distance : double or ndarray
        The Catch22-Euclidean distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> catcheucl_distance([[1, 1, 1, 1], [0, 1, 1, 0]], [[0, 1, 0, 1], [-1, 0, 0, 1]])
    array([[...]])
    >>> catcheucl_distance([[1, 1, 1, 1], [0, 1, 1, 0]])
    array([[0., 1.],
           [1., 0.]])

    """
    _u, _v = check_input(u, v)
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.catch_euclidean(_u_backend, _v_backend, par)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def erp_distance(
    u: ArrayLike,
    v: Optional[ArrayLike] = None,
    band: Optional[float] = 1.0,
    gap_penalty: Optional[float] = 0.0,
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
    par : bool, optional
        Enable parallel computation (default is True).
    device : str, optional
        Device to run the computation on, either 'cpu' or 'gpu' (default is 'cpu').

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
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.erp(_u_backend, _v_backend, band, gap_penalty, par, device)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def lcss_distance(
    u: ArrayLike,
    v: Optional[ArrayLike] = None,
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
    par : bool, optional
        Enable parallel computation (default is True).
    device : str, optional
        Device to run the computation on, either 'cpu' or 'gpu' (default is 'cpu').

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
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.lcss(_u_backend, _v_backend, band, epsilon, par, device)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def dtw_distance(
    u: ArrayLike,
    v: Optional[ArrayLike] = None,
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
    par : bool, optional
        Enable parallel computation (default is True).
    device : str, optional
        Device to run the computation on, either 'cpu' or 'gpu' (default is 'cpu').

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
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.dtw(_u_backend, _v_backend, band, par, device)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def ddtw_distance(
    u: ArrayLike,
    v: Optional[ArrayLike] = None,
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
    par : bool, optional
        Enable parallel computation (default is True).
    device : str, optional
        Device to run the computation on, either 'cpu' or 'gpu' (default is 'cpu').

    Returns
    -------
    distance : double or ndarray
        The DDTW distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> ddtw_distance([1, 0, 0], [0, 1, 0])
    1.5625
    >>> ddtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[0.25  , 0.5625],
           [0.0625, 0.    ]])
    >>> ddtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.    , 0.5625],
           [0.5625, 0.    ]])
    """
    _u, _v = check_input(u, v)
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.ddtw(_u_backend, _v_backend, band, par, device)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def wdtw_distance(
    u: ArrayLike,
    v: Optional[ArrayLike] = None,
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
    g : double, optional
        Controls the strength of the logistic weight applied to warping (default is 0.05).
    par : bool, optional
        Enable parallel computation (default is True).
    device : str, optional
        Device to run the computation on, either 'cpu' or 'gpu' (default is 'cpu').

    Returns
    -------
    distance : double or ndarray
        The WDTW distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> wdtw_distance([1, 0, 0], [0, 1, 0])
    0.4812587841214648
    >>> wdtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[0.96251757, 2.8875527 ],
           [0.48125878, 1.44377635]])
    >>> wdtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.        , 0.48125878],
           [0.48125878, 0.        ]])
    """
    _u, _v = check_input(u, v)
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.wdtw(_u_backend, _v_backend, band, g, par, device)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def wddtw_distance(
    u: ArrayLike,
    v: Optional[ArrayLike] = None,
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
    g : double, optional
        Controls the strength of the logistic weight applied to warping (default is 0.05).
    par : bool, optional
        Enable parallel computation (default is True).
    device : str, optional
        Device to run the computation on, either 'cpu' or 'gpu' (default is 'cpu').

    Returns
    -------
    distance : double or ndarray
        The WDDTW distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> wddtw_distance([1, 0, 0], [0, 1, 0])
    0.7714848835945151
    >>> wddtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[0.12343758, 0.27773456],
           [0.0308594 , 0.        ]])
    >>> wddtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0.        , 0.27773456],
           [0.27773456, 0.        ]])
    """
    _u, _v = check_input(u, v)
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.wddtw(_u_backend, _v_backend, band, g, par, device)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def adtw_distance(
    u: ArrayLike,
    v: Optional[ArrayLike] = None,
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
    warp_penalty : double, optional
        Additive penalty applied to each warping step (default is 1.0).
    par : bool, optional
        Enable parallel computation (default is True).
    device : str, optional
        Device to run the computation on, either 'cpu' or 'gpu' (default is 'cpu').

    Returns
    -------
    distance : double or ndarray
        The ADTW distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> adtw_distance([1, 0, 0], [0, 1, 0])
    2.0
    >>> adtw_distance([[1, 1, 1], [0, 1, 1]], [[0, 1, 0], [-1, 0, 0]])
    array([[2., 6.],
           [1., 3.]])
    >>> adtw_distance([[1, 1, 1], [0, 1, 1]])
    array([[0., 1.],
           [1., 0.]])
    """
    _u, _v = check_input(u, v)
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.adtw(_u_backend, _v_backend, band, warp_penalty, par, device)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def msm_distance(
    u: ArrayLike,
    v: Optional[ArrayLike] = None,
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
    par : bool, optional
        Enable parallel computation (default is True).
    device : str, optional
        Device to run the computation on, either 'cpu' or 'gpu' (default is 'cpu').

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
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.msm(_u_backend, _v_backend, band, par, device)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def twe_distance(
    u: ArrayLike,
    v: Optional[ArrayLike] = None,
    band: Optional[float] = 1.0,
    stiffness: Optional[float] = 0.001,
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
    stiffness : double, optional
        Elasticity parameter, also referred to as nu (default is 0.001).
    penalty : double, optional
        Penalty for gap insertion/deletion, also referred to as lambda (default is 1.0).
    par : bool, optional
        Enable parallel computation (default is True).
    device : str, optional
        Device to run the computation on, either 'cpu' or 'gpu' (default is 'cpu').

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
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.twe(_u_backend, _v_backend, band, stiffness, penalty, par, device)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def sb_distance(
    u: ArrayLike,
    v: Optional[ArrayLike] = None,
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
    par : bool, optional
        Enable parallel computation (default is True).

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
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.sb(_u_backend, _v_backend, par)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


@typechecked
def mp_distance(
    u: ArrayLike,
    window: Optional[int] = 20,
    v: Optional[ArrayLike] = None,
    par: Optional[bool] = True,
) -> Union[np.ndarray, float]:
    """
    Computes the Matrix Profile distance (MPdist) [1] between two 1-D arrays or between two sets of 1-D arrays.
    If `v` is None, the function computes the pairwise MP distances within `u`.
    The length of the input arrays are not required to be the same.

    [1] Gharghabi S. et al., An Ultra-Fast Time Series Distance Measure to allow Data Mining in more Complex Real-World Deployments, 2020.

    Parameters
    ----------
    u : (N,) array_like or (M, N)
        Input array. If 1-D, `u` represents a single vector. If 2-D, `u` represents a set of vectors.
    window : int, optional
        Window size for the Matrix Profile calculation (default is 20).
    v : (N,) array_like or (M, N), optional
        Input array.
        If `v` is None, pairwise distances within `u` are computed.
    par : bool, optional
        Enable parallel computation (default is True).

    Returns
    -------
    distance : double or ndarray
        The MP distance(s) between vectors/sets `u` and `v`.

    Examples
    --------
    >>> mp_distance([1, 0, 0], [0, 1, 0], window=2)
    1.4142135623730951
    >>> mp_distance([[1, 1, 1], [0, 1, 1]], window=2, v=[[0, 1, 0], [-1, 0, 0]])
    array([[1.41421356, 2.44948974], [1.        , 1.73205081]])
    >>> mp_distance([[1, 1, 1], [0, 1, 1]], window=2)
    array([[0.        , 1.        ], [1.        , 0.        ]])
    """
    _u, _v = check_input(u, v)
    _u_backend, _v_backend = _to_backend_inputs(_u, _v)

    result = tsd.mp(_u_backend, window, _v_backend, par)
    if _v is not None and _u.shape[0] == 1 and _v.shape[0] == 1:
        return float(result[0, 0])
    return result


__all__ = [
    "euclidean_distance",
    "catcheucl_distance",
    "erp_distance",
    "lcss_distance",
    "dtw_distance",
    "ddtw_distance",
    "wdtw_distance",
    "wddtw_distance",
    "adtw_distance",
    "msm_distance",
    "twe_distance",
    "sb_distance",
    "mp_distance",
]
