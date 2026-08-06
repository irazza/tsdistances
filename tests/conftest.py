"""Shared pytest fixtures for the tsdistances test-suite.

This module centralizes two things that used to be copy-pasted (and fragile)
across the correctness tests:

* Loading the ACSF1 sample dataset via a path anchored to this file (so tests
  no longer depend on the current working directory), with a deterministic
  synthetic fallback when the dataset is absent.
* Importing the reference "oracle" libraries (``aeon``, ``stumpy``,
  ``pycatch22``). Locally a missing oracle skips the dependent tests; in CI the
  ``--require-oracles`` flag turns those skips into hard errors so the oracle
  suites can never pass vacuously.
"""

import importlib
import warnings
from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).parent
ACSF1_DIR = TESTS_DIR / "ACSF1"
N_SAMPLES = 10


def pytest_addoption(parser):
    parser.addoption(
        "--require-oracles",
        action="store_true",
        default=False,
        help=(
            "Treat a missing oracle library (aeon/stumpy/pycatch22) as an error "
            "instead of a skip. Intended for CI so the oracle correctness suites "
            "cannot pass vacuously when a dependency failed to install."
        ),
    )


def _import_oracle(request, module_name, reason):
    """Import an oracle module, or skip/fail depending on --require-oracles."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on the environment
        message = f"{reason} (could not import {module_name!r}: {exc})"
        if request.config.getoption("--require-oracles"):
            pytest.fail(message, pytrace=False)
        pytest.skip(message)


@pytest.fixture(scope="session")
def acsf1():
    """Return ``(A, B)`` sample collections from the ACSF1 UCR dataset.

    ``A`` is the first ``N_SAMPLES`` training series and ``B`` the last
    ``N_SAMPLES`` test series, each with the leading class-label column dropped.
    If the dataset is not present the fixture emits a visible warning and falls
    back to a deterministic synthetic dataset so the correctness checks still
    run (against the oracle) rather than silently disappearing.
    """
    train = ACSF1_DIR / "ACSF1_TRAIN.tsv"
    test = ACSF1_DIR / "ACSF1_TEST.tsv"
    if train.exists() and test.exists():
        a = np.loadtxt(train, delimiter="\t")[:N_SAMPLES, 1:]
        b = np.loadtxt(test, delimiter="\t")[-N_SAMPLES:, 1:]
        return a, b

    warnings.warn(
        f"ACSF1 dataset not found under {ACSF1_DIR}; falling back to a "
        "deterministic synthetic dataset. Correctness is still checked against "
        "the oracle, but on synthetic data rather than ACSF1.",
        stacklevel=2,
    )
    rng = np.random.default_rng(0)
    a = rng.standard_normal((N_SAMPLES, 100))
    b = rng.standard_normal((N_SAMPLES, 100))
    return a, b


@pytest.fixture(scope="session")
def equal_length_pair():
    """Deterministic equal-length collections for banded cross-checks."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal((6, 40))
    b = rng.standard_normal((5, 40))
    return a, b


@pytest.fixture(scope="session")
def unequal_length_pair():
    """Deterministic collections of *variable-length* (ragged) series.

    Returned as lists of Python lists because a ragged collection cannot be a
    single rectangular ndarray. Both tsdistances (ragged input) and aeon
    (looped single-pair calls) accept this shape.
    """
    rng = np.random.default_rng(7)
    a = [rng.standard_normal(n).tolist() for n in (30, 35, 40)]
    b = [rng.standard_normal(n).tolist() for n in (28, 36)]
    return a, b


@pytest.fixture(scope="session")
def aeon_distances(request):
    """The ``aeon.distances`` module (the primary elastic-distance oracle)."""
    return _import_oracle(
        request,
        "aeon.distances",
        "aeon is required for the aeon distance correctness oracle",
    )


@pytest.fixture(scope="session")
def stumpy_mod(request):
    """The ``stumpy`` module (matrix-profile oracle)."""
    return _import_oracle(
        request,
        "stumpy",
        "stumpy is required for the matrix-profile (MPdist) oracle",
    )


@pytest.fixture(scope="session")
def pycatch22_mod(request):
    """The ``pycatch22`` module (catch22 feature oracle)."""
    return _import_oracle(
        request,
        "pycatch22",
        "pycatch22 is required for the catch22 feature oracle",
    )
