import numpy as np
import pytest
from tsdistances import (
    euclidean_distance,
    catcheucl_distance,
    erp_distance,
    lcss_distance,
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
    wddtw_distance,
    adtw_distance,
    msm_distance,
    twe_distance,
    sb_distance,
    mp_distance,
)


DISTANCES_CHECK_INPUT = [
    erp_distance,
    lcss_distance,
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
    wddtw_distance,
    adtw_distance,
    msm_distance,
    twe_distance,
    sb_distance,
    mp_distance,
]

TYPECHECKED_DISTANCES = [
    euclidean_distance,
    catcheucl_distance,
    erp_distance,
    lcss_distance,
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
    wddtw_distance,
    adtw_distance,
    msm_distance,
    twe_distance,
    sb_distance,
]

EUCLIDEAN_LIKE = [euclidean_distance, catcheucl_distance]

BAND_VALIDATED_DISTANCES = [
    erp_distance,
    lcss_distance,
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
    wddtw_distance,
    adtw_distance,
    msm_distance,
    twe_distance,
]

DEVICE_VALIDATED_DISTANCES = [
    erp_distance,
    lcss_distance,
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
    wddtw_distance,
    adtw_distance,
    msm_distance,
    twe_distance,
]

PAIRWISE_DISTANCES = [
    erp_distance,
    lcss_distance,
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
    wddtw_distance,
    adtw_distance,
    msm_distance,
    twe_distance,
    sb_distance,
    mp_distance,
]


@pytest.mark.parametrize("dist", DISTANCES_CHECK_INPUT)
def test_1d_requires_v(dist):
    u = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="If `u` is 1-D"):
        dist(u)


@pytest.mark.parametrize("dist", EUCLIDEAN_LIKE)
def test_euclidean_like_rejects_1d_without_v(dist):
    u = np.array([1.0, 2.0, 3.0])
    with pytest.raises(Exception):
        dist(u)


@pytest.mark.parametrize("dist", DISTANCES_CHECK_INPUT)
def test_invalid_u_ndim(dist):
    u = np.zeros((2, 2, 2))
    v = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="`u` must be 1-D or 2-D"):
        dist(u, v=v)


@pytest.mark.parametrize("dist", DISTANCES_CHECK_INPUT)
def test_invalid_v_ndim(dist):
    u = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = np.zeros((1, 1, 1))
    with pytest.raises(ValueError, match="`v` must be 1-D or 2-D"):
        dist(u, v=v)


@pytest.mark.parametrize("dist", TYPECHECKED_DISTANCES)
def test_accepts_array_like_inputs(dist):
    u = [1.0, 2.0, 3.0]
    v = [1.0, 2.0, 3.0]
    assert dist(np.array(u), np.array(v)) is not None
    assert dist(u, v) is not None


@pytest.mark.parametrize("dist", BAND_VALIDATED_DISTANCES)
@pytest.mark.parametrize("band", [-0.1, 1.1])
def test_band_out_of_range(dist, band):
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Sakoe-Chiba band"):
        dist(u, v, band=band)


@pytest.mark.parametrize(
    "dist, kwargs, match",
    [
        (erp_distance, {"gap_penalty": -0.1}, "Gap penalty"),
        (lcss_distance, {"epsilon": -0.1}, "Epsilon"),
        (adtw_distance, {"warp_penalty": -0.1}, "Weight must be non-negative"),
        (twe_distance, {"stifness": -0.1}, "Stiffness"),
        (twe_distance, {"penalty": -0.1}, "Penalty"),
    ],
)
def test_negative_parameters_raise(dist, kwargs, match):
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=match):
        dist(u, v, **kwargs)


@pytest.mark.parametrize("dist", DEVICE_VALIDATED_DISTANCES)
def test_invalid_device(dist):
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Device must be either 'cpu' or 'gpu'"):
        dist(u, v, device="tpu")


@pytest.mark.parametrize("dist", DISTANCES_CHECK_INPUT)
def test_shape_handling_mixed_dims(dist):
    u_1d = np.array([1.0, 2.0, 3.0])
    v_2d = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    u_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v_1d = np.array([1.0, 0.0, 1.0])

    d1 = dist(u_1d, v=v_2d)
    assert d1.shape == (1, v_2d.shape[0])

    d2 = dist(u_2d, v=v_1d)
    assert d2.shape == (u_2d.shape[0], 1)

    d3 = dist(u_2d, v=None)
    assert d3.shape == (u_2d.shape[0], u_2d.shape[0])


@pytest.mark.parametrize(
    "dist, kwargs",
    [
        (dtw_distance, {"band": 0.0}),
        (dtw_distance, {"band": 1.0}),
        (lcss_distance, {"band": 0.0, "epsilon": 0.0}),
        (erp_distance, {"band": 1.0, "gap_penalty": 0.0}),
        (wdtw_distance, {"band": 1.0, "g": 0.0}),
        (wddtw_distance, {"band": 0.5, "g": 0.1}),
        (adtw_distance, {"band": 1.0, "warp_penalty": 0.0}),
        (twe_distance, {"band": 1.0, "stifness": 0.0, "penalty": 0.0}),
    ],
)
def test_parameter_edge_values(dist, kwargs):
    u = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([2.0, 3.0, 4.0, 5.0])
    out = dist(u, v, **kwargs)
    assert np.isfinite(out)


@pytest.mark.parametrize("dist", PAIRWISE_DISTANCES)
def test_non_contiguous_inputs(dist):
    base = np.arange(30.0).reshape(5, 6)
    u = base[:, ::2]  # non-contiguous view
    v = base[:, 1::2]
    u_c = np.ascontiguousarray(u)
    v_c = np.ascontiguousarray(v)
    if dist is mp_distance:
        d_nc = dist(u, v=v)
        d_c = dist(u_c, v=v_c)
    else:
        d_nc = dist(u, v)
        d_c = dist(u_c, v_c)
    assert np.allclose(d_nc, d_c, atol=1e-8)


def test_mp_window_extremes():
    u = np.array([[1.0, 2.0, 3.0, 4.0]])
    v = np.array([[4.0, 3.0, 2.0, 1.0]])
    d_min = mp_distance(u, window=1, v=v)
    d_max = mp_distance(u, window=u.shape[1], v=v)
    assert not np.isnan(d_min).any()
    assert np.isfinite(d_max).all()
