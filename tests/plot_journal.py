import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

UCR_ARCHIVE_PATH = pathlib.Path('../../DATA/ucr')
BENCHMARKS_DS = ["ACSF1", "Adiac", "Beef", "CBF", "ChlorineConcentration", "CinCECGTorso", "CricketX", "DiatomSizeReduction", "DistalPhalanxOutlineCorrect", "ECG200", "EthanolLevel", "FreezerRegularTrain", "FreezerSmallTrain", "Ham", "Haptics", "HouseTwenty", "ItalyPowerDemand", "MixedShapesSmallTrain", "NonInvasiveFetalECGThorax1", "ShapesAll", "Strawberry", "UWaveGestureLibraryX", "Wafer"]

def load_benchmark_filtered():
    benchmark_ds = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.name in BENCHMARKS_DS])
    return benchmark_ds

def load_benchmark():
    benchmark_ds = sorted([x for x in UCR_ARCHIVE_PATH.iterdir() if x.is_dir()])
    return benchmark_ds

DATASETS_PATH = load_benchmark_filtered()

def draw_scatter_ucr():
    
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

def tsdistances_with_threads():

    times = np.loadtxt("tests/ACSF1/times_per_thread.csv", delimiter=",")
    times = times.mean(axis=1)
    # bar plot with number of threads on x and times on y
    sns.barplot(x=np.arange(1, 17), y=times[::-1])
    plt.xlabel("Number of Threads")
    plt.ylabel("Time (s)")
    plt.title("ACSF1 vs Number of Threads")
    plt.savefig("tests/times_per_thread.svg", dpi=300)
    caption = "Elapsed time computing the DTW distance on the ACSF1 Dataset (TRAIN vs TEST), changing the number of threads used."
