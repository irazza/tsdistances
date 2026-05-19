import numpy as np

from tsdistances import (
    check_input,
    dtw_distance,
    euclidean_distance,
    mp_distance,
    sb_distance,
)


def test_check_input_normalizes_supported_shapes():
    u, v = check_input([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert u.shape == (1, 3)
    assert v.shape == (1, 3)

    u, v = check_input([1.0, 2.0, 3.0], [[3.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
    assert u.shape == (1, 3)
    assert v.shape == (2, 3)

    u, v = check_input([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], [3.0, 2.0, 1.0])
    assert u.shape == (2, 3)
    assert v.shape == (1, 3)

    u, v = check_input([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    assert u.shape == (2, 3)
    assert v is None


def test_euclidean_wrapper_accepts_numpy_arrays_and_returns_scalar():
    distance = euclidean_distance(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        par=False,
    )
    assert isinstance(distance, float)
    assert np.isclose(distance, np.sqrt(2.0))


def test_matrix_wrappers_return_expected_shapes_and_values():
    u = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    v = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    euclidean = euclidean_distance(u, v, par=False)
    dtw = dtw_distance(u, v, par=False)

    assert euclidean.shape == (2, 2)
    assert dtw.shape == (2, 2)
    assert np.allclose(
        euclidean,
        np.array([[np.sqrt(2.0), np.sqrt(6.0)], [1.0, np.sqrt(3.0)]]),
        atol=1e-8,
    )
    assert np.allclose(dtw, np.array([[2.0, 6.0], [1.0, 3.0]]), atol=1e-8)


def test_parallel_and_sequential_binding_results_match():
    u = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0, 1.0],
        ]
    )
    v = np.array(
        [
            [0.0, 0.5, 1.5, 3.0],
            [2.0, 1.0, 0.5, 0.0],
        ]
    )

    assert np.allclose(euclidean_distance(u, v, par=False), euclidean_distance(u, v, par=True))
    assert np.allclose(dtw_distance(u, v, par=False), dtw_distance(u, v, par=True))


def test_shape_based_and_matrix_profile_wrappers_return_finite_outputs():
    u = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 1.0, 0.0]])
    v = np.array([[0.0, 0.5, 1.5, 3.0], [2.0, 1.0, 0.5, 0.0]])

    sb = sb_distance(u, v, par=False)
    mp = mp_distance(u, window=2, v=v, par=False)

    assert sb.shape == (2, 2)
    assert mp.shape == (2, 2)
    assert np.isfinite(sb).all()
    assert np.isfinite(mp).all()
