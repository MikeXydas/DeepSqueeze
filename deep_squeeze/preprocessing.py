import numpy as np

from sklearn.preprocessing import MinMaxScaler


def ds_preprocessing(x, error_threshold, min_val=0, max_val=1):
    """
    Performs dataset preprocessing as described in the DeepSqueeze paper. We first scale in range [min_val, max_val]
    and then we perform error quantization based on error threshold.

    Note: For simplicity reasons, error_threshold is the same for all the columns.

    Args:
        x: A 2D numpy array of shape [#datapoints, #features]
        error_threshold: The acceptable error bound for all the columns. Make sure to scale it appropriately with
        min_val, max_val, i.e. if min_val=0, max_val=1 an error of 0.1 means 10% error
        min_val: The minimum value of our minmax scaling
        max_val: The maximum value of our minmax scaling

    Returns:
        The quantized table ready to be inputted in our autoencoder and the scaler to inverse the minmax scaling.
    """
    # Scale in range [min_val, max_val]
    scaler = MinMaxScaler((min_val, max_val))
    processed = scaler.fit_transform(x)

    # Quantization
    bins = np.arange(min_val, max_val, 2 * error_threshold)
    digitized = np.digitize(processed, bins)
    quantized = (digitized - 1) * (2 * error_threshold) + error_threshold

    return quantized, scaler
