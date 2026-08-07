"""Baseline (band=1.0) correctness of the CPU path against aeon and stumpy.

Uses the shared fixtures in ``conftest.py``: ``acsf1`` loads the dataset via a
path anchored to this file (with a deterministic synthetic fallback), and the
oracle fixtures skip locally / hard-fail under ``--require-oracles`` in CI.

The constrained-band cross-checks live in ``test_correctness_banded.py``.
"""

import numpy as np
import pytest

from tsdistances import (
    adtw_distance,
    ddtw_distance,
    dtw_distance,
    erp_distance,
    euclidean_distance,
    lcss_distance,
    mp_distance,
    msm_distance,
    sb_distance,
    twe_distance,
    wddtw_distance,
    wdtw_distance,
)

BAND = 1.0

# (name, tsdistances fn, ts kwargs, aeon pairwise fn name, aeon kwargs)
AEON_CASES = [
    ("euclidean", euclidean_distance, {"par": True}, "euclidean_pairwise_distance", {}),
    ("erp", erp_distance, {"band": BAND, "gap_penalty": 0.0, "par": True}, "erp_pairwise_distance", {"g": 0.0, "window": BAND}),
    ("lcss", lcss_distance, {"band": BAND, "epsilon": 0.1, "par": True}, "lcss_pairwise_distance", {"epsilon": 0.1, "window": BAND}),
    ("dtw", dtw_distance, {"band": BAND, "par": True}, "dtw_pairwise_distance", {"window": BAND}),
    ("ddtw", ddtw_distance, {"band": BAND, "par": True}, "ddtw_pairwise_distance", {"window": BAND}),
    ("wdtw", wdtw_distance, {"band": BAND, "g": 0.05, "par": True}, "wdtw_pairwise_distance", {"g": 0.05, "window": BAND}),
    ("wddtw", wddtw_distance, {"band": BAND, "g": 0.05, "par": True}, "wddtw_pairwise_distance", {"g": 0.05, "window": BAND}),
    ("adtw", adtw_distance, {"band": BAND, "warp_penalty": 1.0, "par": True}, "adtw_pairwise_distance", {"window": BAND, "warp_penalty": 1.0}),
    ("msm", msm_distance, {"band": BAND, "par": True}, "msm_pairwise_distance", {"window": BAND}),
    ("twe", twe_distance, {"band": BAND, "stiffness": 0.1, "penalty": 0.1, "par": True}, "twe_pairwise_distance", {"nu": 0.1, "lmbda": 0.1, "window": BAND}),
    ("sbd", sb_distance, {"par": True}, "sbd_pairwise_distance", {}),
]
AEON_IDS = [c[0] for c in AEON_CASES]


@pytest.mark.parametrize(
    "name, tsdist, ts_kwargs, aeon_name, aeon_kwargs", AEON_CASES, ids=AEON_IDS
)
def test_aeon_distances(acsf1, aeon_distances, name, tsdist, ts_kwargs, aeon_name, aeon_kwargs):
    a, b = acsf1
    result = np.asarray(tsdist(a, b, **ts_kwargs))
    aeon_fn = getattr(aeon_distances, aeon_name)
    expected = np.asarray(aeon_fn(a, b, **aeon_kwargs))
    assert np.allclose(result, expected, atol=1e-8)


def test_mp_distance(acsf1, stumpy_mod):
    a, b = acsf1
    window = int(0.1 * a.shape[1])
    result = np.asarray(mp_distance(a, window, b, par=True))
    expected = np.zeros_like(result)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            expected[i, j] = stumpy_mod.mpdist(a[i], b[j], m=window)
    assert np.allclose(result, expected, atol=1e-8)
