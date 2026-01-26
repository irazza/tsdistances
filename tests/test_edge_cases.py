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
