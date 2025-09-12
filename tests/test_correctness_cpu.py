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
from aeon import distances as aeon
import stumpy
import time

N_SAMPLES = 10
A = np.loadtxt("tests/ACSF1/ACSF1_TRAIN.tsv", delimiter="\t")[:N_SAMPLES, 1:]
B = np.loadtxt("tests/ACSF1/ACSF1_TEST.tsv", delimiter="\t")[-N_SAMPLES:, 1:]
band = 1.0


def test_euclidean_distance():
    D = euclidean_distance(A, B, par=True)
    aeon_D = aeon.euclidean_pairwise_distance(A, B)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_erp_distance():
    gap_penalty = 0.0
    D = erp_distance(A, B, gap_penalty=gap_penalty, band=band, par=True)
    aeon_D = aeon.erp_pairwise_distance(A, B, g=gap_penalty, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_lcss_distance():
    epsilon = 0.1
    D = lcss_distance(A, B, epsilon=epsilon, band=band, par=True)
    aeon_D = aeon.lcss_pairwise_distance(A, B, epsilon=epsilon, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_dtw_distance():
    D = dtw_distance(A, B, band=band, par=True)
    aeon_D = aeon.dtw_pairwise_distance(A, B, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_ddtw_distance():
    D = ddtw_distance(A, B, band=band, par=True)
    aeon_D = aeon.ddtw_pairwise_distance(A, B, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_wdtw_distance():
    g = 0.05
    D = wdtw_distance(A, B, g=g, band=band, par=True)
    aeon_D = aeon.wdtw_pairwise_distance(A, B, g=g, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_wddtw_distance():
    g = 0.05
    D = wddtw_distance(A, B, g=g, band=band, par=True)
    aeon_D = aeon.wddtw_pairwise_distance(A, B, g=g, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_adtw_distance():
    warp_penalty = 1.0
    D = adtw_distance(A, B, band=band, warp_penalty=warp_penalty, par=True)
    aeon_D = aeon.adtw_pairwise_distance(A, B, window=band, warp_penalty=warp_penalty)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_msm_distance():
    D = msm_distance(A, B, band=band, par=True)
    aeon_D = aeon.msm_pairwise_distance(A, B, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_twe_distance():
    stiffness = 0.1
    penalty = 0.1
    D = twe_distance(A, B, band=band, stifness=stiffness, penalty=penalty, par=True)
    aeon_D = aeon.twe_pairwise_distance(A, B, nu=stiffness, lmbda=penalty, window=band)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_sb_distance():
    D = sb_distance(A, B, par=True)
    aeon_D = aeon.sbd_pairwise_distance(A, B)
    assert np.allclose(D, aeon_D, atol=1e-8)


def test_mp_distance():
    window = int(0.1 * A.shape[1])
    D = mp_distance(A, window, B, par=True)
    D_stumpy = np.zeros_like(D)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            D_stumpy[i, j] = stumpy.mpdist(A[i], B[j], m=window)
    assert np.allclose(D, D_stumpy, atol=1e-8)
