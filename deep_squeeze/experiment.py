import numpy as np
import pandas as pd
import logging


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


def run_full_experiments(pipeline_func, dataset_paths, errors, params, save_path):
    results_df = pd.DataFrame(columns=['Data', 'Error', 'MeanRatio', 'StdRatio'])

    for dataset in dataset_paths:
        params['data_path'] = dataset
        dataset_name = dataset.split('/')[-1]
        for error in errors:
            params['error_threshold'] = error

            mean_ratio, std_ratio = pipeline_func(params)

            results_df = results_df.append({'Data': dataset_name,
                                            'Error': error,
                                            'MeanRatio': mean_ratio,
                                            'StdRatio': std_ratio},
                                           ignore_index=True)
            logging.info(f">>> Completed {dataset_name} with {error} error threshold.")

    results_df.to_csv(save_path)
