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
AEONDISTANCES = [erp_pairwise_distance, dtw_pairwise_distance, adtw_pairwise_distance]
MODALITIES = ["", "par", "gpu"]
NUM_RUNS = 1  # Number of times to run each benchmark

def load_benchmark_filtered():
    benchmark_ds = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.name in BENCHMARKS_DS])
    return benchmark_ds

def load_benchmark():
    benchmark_ds = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.is_dir()])
    return benchmark_ds

DATASETS_PATH = load_benchmark()

def test_tsdistances():
    tsdistances_times = np.full((len(DATASETS_PATH), len(TSDISTANCES), len(MODALITIES), NUM_RUNS), np.nan)
    aeon_times = np.full((len(DATASETS_PATH), len(TSDISTANCES), NUM_RUNS), np.nan)

    for i, dataset in enumerate(DATASETS_PATH):
        print(f"\nDataset: {dataset.name}")
        train = np.loadtxt(dataset / f"{dataset.name}_TRAIN.tsv", delimiter="\t")
        test = np.loadtxt(dataset / f"{dataset.name}_TEST.tsv", delimiter="\t")
        X_train = train[:, 1:]
        X_test = test[:, 1:]

        for j, (tsdist, aeondist) in enumerate(zip(TSDISTANCES, AEONDISTANCES)):
            # Create small subset for warmup
            X_train_warmup = X_train[:min(10, len(X_train))]
            X_test_warmup = X_test[:min(10, len(X_test))]
            
            # Warmup run
            _ = tsdist(X_train_warmup, X_test_warmup, par=False)
            
            # Single-threaded runs
            for run in range(NUM_RUNS):
                start = time.time()
                D = tsdist(X_train, X_test, par=False)
                end = time.time()
                tsdistances_times[i, j, 0, run] = end - start

            # Warmup parallel
            _ = tsdist(X_train_warmup, X_test_warmup, par=True)
            
            # Parallel runs
            for run in range(NUM_RUNS):
                start = time.time()
                D_par = tsdist(X_train, X_test, par=True)
                end = time.time()
                tsdistances_times[i, j, 1, run] = end - start

            # GPU runs (if supported)
            if tsdist.__name__ != "euclidean_distance":
                # Warmup GPU
                _ = tsdist(X_train_warmup, X_test_warmup, device='gpu')
                
                for run in range(NUM_RUNS):
                    start = time.time()
                    D_gpu = tsdist(X_train, X_test, device='gpu')
                    end = time.time()
                    tsdistances_times[i, j, 2, run] = end - start

            # AEON distances - warmup
            _ = aeondist(X_train_warmup, X_test_warmup)
            
            for run in range(NUM_RUNS):
                start = time.time()
                D_aeon = aeondist(X_train, X_test)
                end = time.time()
                aeon_times[i, j, run] = end - start

            # Print statistics
            mean_single = np.mean(tsdistances_times[i, j, 0, :])
            std_single = np.std(tsdistances_times[i, j, 0, :])
            mean_par = np.mean(tsdistances_times[i, j, 1, :])
            std_par = np.std(tsdistances_times[i, j, 1, :])
            mean_aeon = np.mean(aeon_times[i, j, :])
            std_aeon = np.std(aeon_times[i, j, :])
            
            gpu_str = ""
            if tsdist.__name__ != "euclidean_distance":
                mean_gpu = np.mean(tsdistances_times[i, j, 2, :])
                std_gpu = np.std(tsdistances_times[i, j, 2, :])
                gpu_str = f", {mean_gpu:.4f}±{std_gpu:.4f} (gpu)"
            
            print(f"\t{tsdist.__name__}:")
            print(f"\t\tSingle: {mean_single:.4f}±{std_single:.4f} (s)")
            print(f"\t\tParallel: {mean_par:.4f}±{std_par:.4f} (s){gpu_str}")
            print(f"\t\tAEON: {mean_aeon:.4f}±{std_aeon:.4f} (s)")

            if not np.allclose(D, D_aeon, atol=1e-8):
                print("\t\tWARNING: AEON and tsdistances results do not match")

            np.save("times_tsdistances_all.npy", tsdistances_times)
            np.save("times_aeon_all.npy", aeon_times)