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
# TSDISTANCES = [erp_distance, adtw_distance]
# AEONDISTANCES = [erp_pairwise_distance, adtw_pairwise_distance]
TSDISTANCES = [dtw_distance]
AEONDISTANCES = [dtw_distance]
MODALITIES = ["", "par", "gpu"]

def load_benchmark():
    benchmark_ds = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.name in BENCHMARKS_DS])
    return benchmark_ds

DATASETS_PATH = load_benchmark()

def test_draw_scatter_ucr():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    ucr_datasets = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.is_dir()])
    ucr_info = np.zeros((len(ucr_datasets), 3), dtype=int)
    is_benchmark = np.empty(len(ucr_datasets), dtype=str)

    for i, dataset in enumerate(ucr_datasets):
        train = np.loadtxt(dataset / f"{dataset.name}_TRAIN.tsv", delimiter="\t")
        test = np.loadtxt(dataset / f"{dataset.name}_TEST.tsv", delimiter="\t")
        X_train, _ = train[:, 1:], train[:, 0]
        X_test, _ = test[:, 1:], test[:, 0]

        X = np.vstack((X_train, X_test))
        ucr_info[i] = [X_train.shape[0], X_test.shape[0], X.shape[1]]  # Total number of time series, Time series length, Time series length - 1
        is_benchmark[i] = "Benchmarked" if dataset.name in BENCHMARKS_DS else "Non-benchmarked"
    df = pd.DataFrame(np.column_stack([np.where(is_benchmark=='B')[0]+1, ucr_info[np.where(is_benchmark=='B')[0]]]), columns=["ID", "Train Size", "Test Size", "Time Series Length"], index=[ds.name for ds in DATASETS_PATH])
    df.to_latex("ucr_dataset_info.tex", index=True, float_format="%.0f", escape=False, column_format="lcccc", label='tab:ucr_datasets_info', caption="UCR Dataset Information. The table shows the number of time series in the training and test sets, as well as the length of the time series for each dataset.")
    # Create the scatter plot
    ds_size = ucr_info[:, :2].sum(axis=1)
    data = pd.DataFrame({"Dataset size": ds_size, "Time series Length": ucr_info[:, 2], "Benchmark Status": is_benchmark})

    sns.scatterplot(data=data[data["Benchmark Status"]=="N"], x='Dataset size', y='Time series Length', label='Non-Benchmarked', marker='o')
    sns.scatterplot(data=data[data["Benchmark Status"]=="B"], x='Dataset size', y='Time series Length', label='Benchmarked', marker='x', linewidth=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dataset size (log scale)")
    plt.ylabel("Time series Length (log scale)")
    plt.legend()
    plt.title("UCR Archive Datasets")
    flag = True
    for i in range(len(ucr_datasets)):
        if is_benchmark[i] == "B":
            if i not in [43, 44]:  # Exclude the last two datasets for clarity
                plt.text(
                    ds_size[i],
                    ucr_info[i, 2],
                    str(i+1),
                    ha = 'center',
                    va = 'top',
                    color = 'black',
                    fontsize=6,
                )
            else:
                if flag:
                    plt.text(
                        ds_size[i],
                        ucr_info[i, 2],
                        f"[44-45]",
                        ha = 'center',
                        va = 'top',
                        color = 'black',
                        fontsize=6,
                    )
                    flag = False

    plt.savefig("benchmark_datasets.svg", dpi=300)

def test_with_threads():
    import matplotlib.pyplot as plt
    import seaborn as sns

    times = np.loadtxt("tests/ACSF1/times_per_thread.csv", delimiter=",")
    times = times.mean(axis=1)
    # bar plot with number of threads on x and times on y
    sns.barplot(x=np.arange(1, 17), y=times[::-1])
    plt.xlabel("Number of Threads")
    plt.ylabel("Time (s)")
    plt.title("ACSF1 vs Number of Threads")
    plt.savefig("tests/times_per_thread.svg", dpi=300)
    caption = "Elapsed time computing the DTW distance on the ACSF1 Dataset (TRAIN vs TEST), changing the number of threads used."

def test_tsdistances():
    tsdistances_times = np.full((len(DATASETS_PATH), len(TSDISTANCES), len(MODALITIES)), np.nan)
    aeon_times = np.full((len(DATASETS_PATH), len(TSDISTANCES)), np.nan)

    for i, dataset in enumerate(DATASETS_PATH):
        if dataset.name != "NonInvasiveFetalECGThorax1":
            continue
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

            np.save("times_tsdistances.npy", tsdistances_times)
            np.save("times_aeon.npy", aeon_times)