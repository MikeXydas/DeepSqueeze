import os
import time
import numpy as np
import pandas as pd
import logging
import shutil

from pathlib import Path
from deep_squeeze.disk_storing import calculate_compression_ratio

def repeat_n_times(n):
    """
    A decorator that repeats a decorated function (in our case the compression pipeline) n times and returns
    its mean and its std of its return values.
    Note that the decorated function must return a number.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            comp_ratios = [func(*args) for _ in range(n)]
            comp_ratios = np.array(comp_ratios)
            return np.mean(comp_ratios), np.std(comp_ratios)

        return wrapper

    return decorator


def display_compression_results(mean_ratio, std_ratio, repeats):
    print(f"\n>>> Final results after {repeats} executions:")
    print(f"\tMean compression ratio: {mean_ratio:.3f}")
    print(f"\tStd of compression ratio: {std_ratio:.3f}")


def run_full_experiments(pipeline_func, dataset_paths, errors, params, save_path, repeats):
    results_df = pd.DataFrame(columns=['Data', 'Error', 'MeanRatio', 'StdRatio', 'Time'])

    for dataset in dataset_paths:
        params['data_path'] = dataset
        dataset_name = dataset.split('/')[-1]
        for error in errors:
            start_time = time.time()
            params['error_threshold'] = error

            mean_ratio, std_ratio = pipeline_func(params)

            results_df = results_df.append({'Data': dataset_name,
                                            'Error': error,
                                            'MeanRatio': mean_ratio,
                                            'StdRatio': std_ratio,
                                            'Time': np.round((time.time() - start_time) / repeats, 2)},
                                           ignore_index=True)
            logging.info(f">>> Completed {dataset_name} with {error} error threshold.")

    results_df.to_csv(save_path)


def run_scaling_experiment(sample_sizes, pipeline_func, dataset_path, params, save_path, repeats):
    """
    We run the compression pipeline on increasing size samples of the same dataset to examine the time scaling.
    """
    # Create a temporary directory that will hold the sample csv files and the compressed outputs
    Path("storage/temporary_time_exp/").mkdir(parents=True, exist_ok=True)

    # Init the results df
    results_df = pd.DataFrame(columns=['SampleSize', 'DeepSqueeze', 'Gzip', 'Parquet'])
    # Read the dataset
    df_full = pd.read_csv(dataset_path)
    params['data_path'] = 'storage/temporary_time_exp/temp.csv'
    for sample_size in sample_sizes:
        sample_df = df_full.sample(frac=sample_size)

        # We have to store the file, for our experiment to take into account reading time
        sample_df.to_csv('storage/temporary_time_exp/temp.csv', header=None, index=False)

        # Run and time the DeepSqueeze compression pipeline
        start_time = time.time()
        _, _ = pipeline_func(params)
        deep_squeeze_time = np.round((time.time() - start_time) / repeats, 2)

        # Gzip time
        start_time = time.time()
        sample_df.to_csv("storage/temporary_time_exp/gzip_temp.csv.zip",
                         index=False,
                         compression="zip")
        gzip_time = np.round((time.time() - start_time), 2)

        # Parquet time
        start_time = time.time()
        sample_df.to_parquet("storage/temporary_time_exp/parquet_temp.parquet", index=False)
        parquet_time = np.round((time.time() - start_time), 2)

        results_df = results_df.append({'SampleSize': sample_size,
                                        'DeepSqueeze': deep_squeeze_time,
                                        'Gzip': gzip_time,
                                        'Parquet': parquet_time},
                                       ignore_index=True)
    # Delete created temp files
    shutil.rmtree('storage/temporary_time_exp')

    results_df.to_csv(save_path)


def baseline_compression_ratios(datasets, results_path):
    """
    Calculate the baseline compression ratios of gzip and parquet
    """
    results_df = pd.DataFrame(columns=['Dataset', 'Gzip', 'Parquet'])

    Path("storage/temporary_baseline/").mkdir(parents=True, exist_ok=True)
    for dataset_path in datasets:
        pd.read_csv(dataset_path).to_csv("storage/temporary_baseline/gzip_temp.csv.zip",
                                         index=False,
                                         compression="zip")
        gzip_comp_ratio, _, _ = calculate_compression_ratio(dataset_path,
                                                            "storage/temporary_baseline/gzip_temp.csv.zip")

        pd.read_csv(dataset_path).to_parquet("storage/temporary_baseline/parquet_temp.parquet", index=False)
        parquet_comp_ratio, _, _ = calculate_compression_ratio(dataset_path,
                                                               "storage/temporary_baseline/parquet_temp.parquet")

        results_df = results_df.append({'Dataset': dataset_path.split('/')[-1],
                                        'Gzip': gzip_comp_ratio,
                                        'Parquet': parquet_comp_ratio},
                                       ignore_index=True)

    shutil.rmtree('storage/temporary_baseline')

    results_df.to_csv(results_path)
