import pytest
import numpy as np
from tsdistances import (
    euclidean_distance,
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
try:
    from aeon import distances as aeon
except Exception:  # pragma: no cover - env dependent
    pytest.skip("aeon.distances is unavailable in this environment", allow_module_level=True)
stumpy = pytest.importorskip("stumpy")

N_SAMPLES = 10
A = np.loadtxt("tests/ACSF1/ACSF1_TRAIN.tsv", delimiter="\t")[:N_SAMPLES, 1:]
B = np.loadtxt("tests/ACSF1/ACSF1_TEST.tsv", delimiter="\t")[-N_SAMPLES:, 1:]
band = 1.0

AEON_CASES = [
    (euclidean_distance, {"par": True}, lambda a, b: aeon.euclidean_pairwise_distance(a, b)),
    (erp_distance, {"band": band, "gap_penalty": 0.0, "par": True}, lambda a, b: aeon.erp_pairwise_distance(a, b, g=0.0, window=band)),
    (lcss_distance, {"band": band, "epsilon": 0.1, "par": True}, lambda a, b: aeon.lcss_pairwise_distance(a, b, epsilon=0.1, window=band)),
    (dtw_distance, {"band": band, "par": True}, lambda a, b: aeon.dtw_pairwise_distance(a, b, window=band)),
    (ddtw_distance, {"band": band, "par": True}, lambda a, b: aeon.ddtw_pairwise_distance(a, b, window=band)),
    (wdtw_distance, {"band": band, "g": 0.05, "par": True}, lambda a, b: aeon.wdtw_pairwise_distance(a, b, g=0.05, window=band)),
    (wddtw_distance, {"band": band, "g": 0.05, "par": True}, lambda a, b: aeon.wddtw_pairwise_distance(a, b, g=0.05, window=band)),
    (adtw_distance, {"band": band, "warp_penalty": 1.0, "par": True}, lambda a, b: aeon.adtw_pairwise_distance(a, b, window=band, warp_penalty=1.0)),
    (msm_distance, {"band": band, "par": True}, lambda a, b: aeon.msm_pairwise_distance(a, b, window=band)),
    (twe_distance, {"band": band, "stifness": 0.1, "penalty": 0.1, "par": True}, lambda a, b: aeon.twe_pairwise_distance(a, b, nu=0.1, lmbda=0.1, window=band)),
    (sb_distance, {"par": True}, lambda a, b: aeon.sbd_pairwise_distance(a, b)),
]

@pytest.mark.parametrize("tsdist, ts_kwargs, aeon_fn", AEON_CASES)
def test_aeon_distances(tsdist, ts_kwargs, aeon_fn):
    D = tsdist(A, B, **ts_kwargs)
    aeon_D = aeon_fn(A, B)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_mp_distance():
    window = int(0.1 * A.shape[1])
    D = mp_distance(A, window, B, par=True)
    D_stumpy = np.zeros_like(D)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            D_stumpy[i, j] = stumpy.mpdist(A[i], B[j], m=window)
    assert np.allclose(D, D_stumpy, atol=1e-8)
