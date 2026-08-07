"""Correctness of the Catch22-Euclidean distance.

There is no exact external oracle for tsdistances' ``catcheucl``: the Rust
``catch22`` crate computes 25 features (the 22 catch22 features on the z-scored
series, plus raw mean/std/slope) while ``pycatch22`` exposes 24 (catch24, no
slope), and the two are independent reimplementations. So this module checks two
things:

* **Layer A - feature oracle (structural).** Build a reference distance matrix
  from ``pycatch22`` catch24 features using the exact pipeline tsdistances uses
  (non-finite -> 0, per-column z-normalization with an ``std ~ 0 -> 1`` guard,
  then Euclidean) and assert the off-diagonal distances are strongly *rank*
  correlated with tsdistances. A tight numeric tolerance is not appropriate here
  (different implementations + the extra slope feature), so we assert Spearman
  rank agreement instead.
* **Layer B - exact self-consistency invariants** (``atol=1e-8``): zero diagonal,
  symmetry, ``catcheucl(A, B) == catcheucl(B, A).T``, non-negativity and
  finiteness. These hold regardless of the feature implementation and directly
  guard against regressions such as asymmetric non-finite handling.
"""

import numpy as np

from tsdistances import catcheucl_distance


def _spearman(x, y):
    """Spearman rank correlation via numpy (Pearson correlation of ranks)."""
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def _pycatch22_reference(series_set, pycatch22_mod):
    """Reference catcheucl matrix from pycatch22, mirroring tsdistances' pipeline."""
    features = np.array(
        [
            pycatch22_mod.catch22_all(list(map(float, s)), catch24=True)["values"]
            for s in series_set
        ],
        dtype=float,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    std = features.std(axis=0)
    std = np.where(np.abs(std) < np.finfo(float).eps, 1.0, std)
    z = (features - features.mean(axis=0)) / std
    return np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))


# ---- Layer A: structural agreement with the pycatch22 feature oracle ----


def test_catcheucl_structure_matches_pycatch22(acsf1, pycatch22_mod):
    a, _ = acsf1
    reference = _pycatch22_reference(a, pycatch22_mod)
    result = np.asarray(catcheucl_distance(a, par=False))

    upper = np.triu_indices(len(a), k=1)
    rho = _spearman(reference[upper], result[upper])
    assert rho > 0.85, (
        "catcheucl off-diagonal distances only weakly correlate with the "
        f"pycatch22 reference (Spearman rho={rho:.3f})"
    )


# ---- Layer B: exact self-consistency invariants ----


def test_catcheucl_zero_diagonal_symmetry_and_finiteness(acsf1):
    a, _ = acsf1
    d = np.asarray(catcheucl_distance(a, par=False))
    assert d.shape == (len(a), len(a))
    assert np.allclose(np.diag(d), 0.0, atol=1e-8)
    assert np.allclose(d, d.T, atol=1e-8)
    assert np.all(np.isfinite(d))
    assert np.all(d >= -1e-12)


def test_catcheucl_cross_matrix_transpose_invariant(acsf1):
    a, b = acsf1
    d_ab = np.asarray(catcheucl_distance(a, b, par=False))
    d_ba = np.asarray(catcheucl_distance(b, a, par=False))
    assert np.allclose(d_ab, d_ba.T, atol=1e-8)


def test_catcheucl_parallel_matches_sequential(acsf1):
    a, b = acsf1
    assert np.allclose(
        np.asarray(catcheucl_distance(a, b, par=False)),
        np.asarray(catcheucl_distance(a, b, par=True)),
        atol=1e-8,
    )
