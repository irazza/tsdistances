import pytest
import numpy as np
from tsdistances import (
    euclidean_distance,
    erp_distance,
    dtw_distance,
    adtw_distance,
    twe_distance,
)
from aeon.distances import (
    erp_pairwise_distance,
    dtw_pairwise_distance,
    adtw_pairwise_distance,
    twe_pairwise_distance,
)
import time
import pandas as pd
import pathlib

UCR_ARCHIVE_PATH = pathlib.Path('../../DATA/ucr')
BENCHMARKS_DS = ["ACSF1", "Adiac", "Beef", "CBF", "ChlorineConcentration", "CinCECGTorso", "CricketX", "DiatomSizeReduction", "DistalPhalanxOutlineCorrect", "ECG200", "EthanolLevel", "FreezerRegularTrain", "FreezerSmallTrain", "Ham", "Haptics", "HouseTwenty", "ItalyPowerDemand", "MixedShapesSmallTrain", "NonInvasiveFetalECGThorax1", "ShapesAll", "Strawberry", "UWaveGestureLibraryX", "Wafer"]
TSDISTANCES = [erp_distance, dtw_distance, adtw_distance]
AEONDISTANCES = [erp_pairwise_distance, dtw_distance, adtw_pairwise_distance]
MODALITIES = ["", "par", "gpu"]

def load_benchmark_filtered():
    benchmark_ds = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.name in BENCHMARKS_DS])
    return benchmark_ds

def load_benchmark():
    benchmark_ds = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.is_dir()])
    return benchmark_ds

DATASETS_PATH = load_benchmark()


def test_tsdistances():
    tsdistances_times = np.full((len(DATASETS_PATH), len(TSDISTANCES), len(MODALITIES)), np.nan)
    aeon_times = np.full((len(DATASETS_PATH), len(TSDISTANCES)), np.nan)

    for i, dataset in enumerate(DATASETS_PATH):
        print(f"\nDataset: {dataset.name}")
        train = np.loadtxt(dataset / f"{dataset.name}_TRAIN.tsv", delimiter="\t")
        test = np.loadtxt(dataset / f"{dataset.name}_TEST.tsv", delimiter="\t")
        X_train = train[:, 1:]
        X_test = test[:, 1:]

        for j, (tsdist, aeondist)  in enumerate(zip(TSDISTANCES, AEONDISTANCES)):
            start = time.time()
            D = tsdist(X_train, X_test, par=False)
            end = time.time()
            tsdistances_times[i, j, 0] = end - start

            start = time.time()
            D_par = tsdist(X_train, X_test, par=True)
            end = time.time()
            tsdistances_times[i, j, 1] = end - start

            if tsdist.__name__ != "euclidean_distance":
                start = time.time()
                D_gpu = tsdist(X_train, X_test, device='gpu')
                end = time.time()
                print(end - start)
                tsdistances_times[i, j, 2] = end - start
            # AEON distances
            start = time.time()
            D_aeon = aeondist(X_train, X_test)
            end = time.time()
            aeon_times[i, j] = end - start

            print(f"\t{tsdist.__name__} - \n\t\tTime: {tsdistances_times[i, j, 0]:.4f} (s), {tsdistances_times[i, j, 1]:.4f} (p), {tsdistances_times[i, j, 2]:.4f} (gpu) | AEON: {aeon_times[i, j]:.4f}")
            if not np.allclose(D, D_par):
                print("Parallel and single-threaded results do not match")

            if not np.allclose(D, D_aeon):
                print("AEON and tsdistances results do not match")

            np.save("times_tsdistances_all.npy", tsdistances_times)
            np.save("times_aeon_all.npy", aeon_times)