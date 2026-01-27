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
TSDISTANCES = [erp_distance, dtw_distance, adtw_distance]
AEONDISTANCES = [erp_pairwise_distance, dtw_pairwise_distance, adtw_pairwise_distance]
MODALITIES = ["", "par", "gpu"]
NUM_RUNS = 1  # Number of times to run each benchmark


def generate_benchmark(seed=0):
    rng = np.random.default_rng(seed)
    datasets = []
    for name, n_train, n_test, length in [
        ("SYN_SMALL", 16, 16, 64),
        ("SYN_MEDIUM", 32, 32, 128),
        ("SYN_LONG", 16, 16, 256),
    ]:
        X_train = rng.normal(size=(n_train, length))
        X_test = rng.normal(size=(n_test, length))
        datasets.append((name, X_train, X_test))
    return datasets


DATASETS = generate_benchmark()

def test_tsdistances():
    tsdistances_times = np.full((len(DATASETS), len(TSDISTANCES), len(MODALITIES), NUM_RUNS), np.nan)
    aeon_times = np.full((len(DATASETS), len(TSDISTANCES), NUM_RUNS), np.nan)

    for i, (name, X_train, X_test) in enumerate(DATASETS):
        print(f"\nDataset: {name}")

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
