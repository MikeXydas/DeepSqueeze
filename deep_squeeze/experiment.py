import numpy as np


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
