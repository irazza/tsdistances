"""Banded (Sakoe-Chiba) correctness cross-checks against aeon.

The band=1.0 baseline lives in ``test_correctness_cpu.py``; this module exercises
the *constrained* band path, which aeon's own test-suite never covers here.

Findings (empirically verified against aeon 1.5.0):

* For **equal-length** inputs, tsdistances matches aeon *exactly* at every band
  in {0.0, 0.1, 0.25, 0.5, 0.8, 1.0} for all nine elastic metrics. The
  ``window = band`` mapping is confirmed.
* For **unequal-length** inputs at full band (1.0), tsdistances matches aeon for
  every metric (LCSS included, after switching its normalization to the shorter
  series to match aeon's ``1 - lcss / min(x, y)``).
* For **unequal-length** inputs at a *constrained* band, results diverge: aeon
  builds a per-column staircase bounding matrix (radius ``int(window * min_len)``
  in the original (i, j) grid), while tsdistances projects the band in
  diagonal-wavefront coordinates. The two visited-cell regions differ once the
  series have different lengths. This is a known, tracked divergence, captured
  below as ``xfail(strict=True)``; see the "banded unequal-length" follow-up.
* Also part of that deferred bucket: for *strongly* unequal lengths (e.g. 12 vs
  60) msm and twe return ``inf`` even at full band, because the diagonal
  early-abandon upper bound over-prunes. Moderate ratios (used by the full-band
  assertion below) are unaffected.
"""

import numpy as np
import pytest

from tsdistances import (
    adtw_distance,
    ddtw_distance,
    dtw_distance,
    erp_distance,
    lcss_distance,
    msm_distance,
    twe_distance,
    wddtw_distance,
    wdtw_distance,
)

BANDS = [0.0, 0.1, 0.25, 0.5, 0.8, 1.0]

# (name, tsdistances fn, ts kwargs, aeon pairwise fn name, aeon single-pair fn name,
#  aeon kwargs)
CASES = [
    ("erp", erp_distance, {"gap_penalty": 0.0}, "erp_pairwise_distance", "erp_distance", {"g": 0.0}),
    ("lcss", lcss_distance, {"epsilon": 0.5}, "lcss_pairwise_distance", "lcss_distance", {"epsilon": 0.5}),
    ("dtw", dtw_distance, {}, "dtw_pairwise_distance", "dtw_distance", {}),
    ("ddtw", ddtw_distance, {}, "ddtw_pairwise_distance", "ddtw_distance", {}),
    ("wdtw", wdtw_distance, {"g": 0.05}, "wdtw_pairwise_distance", "wdtw_distance", {"g": 0.05}),
    ("wddtw", wddtw_distance, {"g": 0.05}, "wddtw_pairwise_distance", "wddtw_distance", {"g": 0.05}),
    ("adtw", adtw_distance, {"warp_penalty": 1.0}, "adtw_pairwise_distance", "adtw_distance", {"warp_penalty": 1.0}),
    ("msm", msm_distance, {}, "msm_pairwise_distance", "msm_distance", {}),
    ("twe", twe_distance, {"stiffness": 0.1, "penalty": 0.1}, "twe_pairwise_distance", "twe_distance", {"nu": 0.1, "lmbda": 0.1}),
]
CASE_IDS = [c[0] for c in CASES]

# Deterministic moderate-ratio unequal-length 1-D pairs. At full band every
# metric matches aeon for these ratios; the first pair (17, 53) is also the one
# used to characterize the constrained-band divergence, so its behavior is pinned.
_RNG = np.random.default_rng(1234)
UNEQUAL_PAIRS = [
    (_RNG.standard_normal(17), _RNG.standard_normal(53)),
    (_RNG.standard_normal(30), _RNG.standard_normal(41)),
]


def test_aeon_window_maps_to_sakoe_chiba_radius():
    """Pin aeon's window -> band-radius mapping for the installed aeon version.

    aeon's ``create_bounding_matrix`` uses radius ``int(window * min(x, y))`` and a
    full matrix at ``window == 1.0``. This is the mapping the cross-checks below
    rely on (``window = band``).
    """
    mod = pytest.importorskip(
        "aeon.distances.elastic._bounding_matrix",
        reason="aeon internal bounding-matrix module unavailable for this version",
    )
    create = mod.create_bounding_matrix
    n = 20
    for w in (0.0, 0.1, 0.25, 0.5, 0.8):
        matrix = np.asarray(create(n, n, window=w, itakura_max_slope=None))
        radius = int(w * n)
        # For equal length the first row allows columns [0, radius].
        assert matrix[0].sum() == radius + 1
        # Symmetric band around the diagonal.
        assert bool(matrix[0, 0])
    assert np.asarray(create(n, n, window=1.0, itakura_max_slope=None)).all()


@pytest.mark.parametrize(
    "name, ts_fn, ts_kwargs, aeon_pw, aeon_single, aeon_kwargs", CASES, ids=CASE_IDS
)
@pytest.mark.parametrize("band", BANDS)
def test_banded_equal_length_matches_aeon(
    equal_length_pair, aeon_distances, name, ts_fn, ts_kwargs, aeon_pw, aeon_single, aeon_kwargs, band
):
    a, b = equal_length_pair
    result = np.asarray(ts_fn(a, b, band=band, **ts_kwargs))
    aeon_fn = getattr(aeon_distances, aeon_pw)
    expected = np.asarray(aeon_fn(a, b, window=band, **aeon_kwargs))
    assert np.allclose(result, expected, atol=1e-8)


@pytest.mark.parametrize(
    "name, ts_fn, ts_kwargs, aeon_pw, aeon_single, aeon_kwargs", CASES, ids=CASE_IDS
)
def test_unequal_length_full_band_matches_aeon(
    aeon_distances, name, ts_fn, ts_kwargs, aeon_pw, aeon_single, aeon_kwargs
):
    """At full band (no constraint) unequal-length pairs must match aeon."""
    aeon_fn = getattr(aeon_distances, aeon_single)
    for x, y in UNEQUAL_PAIRS:
        result = float(ts_fn(x, y, band=1.0, **ts_kwargs))
        expected = float(aeon_fn(x, y, window=1.0, **aeon_kwargs))
        assert np.isclose(result, expected, atol=1e-8), (
            f"{name}: {result} vs aeon {expected} for lengths {len(x)},{len(y)}"
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known divergence: for unequal-length series tsdistances projects the "
        "Sakoe-Chiba band in diagonal-wavefront coordinates, whereas aeon uses a "
        "per-column staircase bounding matrix (int(window*min_len)). At window=0.0 "
        "tsdistances' band collapses (inf) while aeon stays finite. Equal-length "
        "matches aeon exactly. Tracked as the 'banded unequal-length' follow-up."
    ),
)
@pytest.mark.parametrize(
    "name, ts_fn, ts_kwargs, aeon_pw, aeon_single, aeon_kwargs", CASES, ids=CASE_IDS
)
def test_unequal_length_constrained_band_diverges_from_aeon(
    aeon_distances, name, ts_fn, ts_kwargs, aeon_pw, aeon_single, aeon_kwargs
):
    aeon_fn = getattr(aeon_distances, aeon_single)
    x, y = UNEQUAL_PAIRS[0]  # lengths 17 vs 53
    result = float(ts_fn(x, y, band=0.0, **ts_kwargs))
    expected = float(aeon_fn(x, y, window=0.0, **aeon_kwargs))
    assert np.isclose(result, expected, atol=1e-8)
