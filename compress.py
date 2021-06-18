import math

import numpy as np
import pandas as pd
import logging
import time
import torch
import argparse

from datetime import datetime

from deep_squeeze.autoencoder import AutoEncoder
from deep_squeeze.preprocessing import ds_preprocessing
from deep_squeeze.train_loop import train
from deep_squeeze.materialization import materialize, materialize_with_post_binning, \
    materialize_with_bin_difference
from deep_squeeze.disk_storing import store_on_disk, calculate_compression_ratio
from deep_squeeze.experiment import repeat_n_times, display_compression_results, run_full_experiments, \
    run_scaling_experiment, baseline_compression_ratios
from deep_squeeze.bayesian_optimizer import minimize_comp_ratio

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(asctime)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')
compression_repeats = 1


@repeat_n_times(n=compression_repeats)  # To produce a consistent result we repeat the experiment n times
def compression_pipeline(params):
    """
    The full compression pipeline performing the following steps:
        1. Preprocess the input table (scaling and quantization)
        2. Initialize the autoencoder model
        3. Train the autoencoder
        4. Materialize the results by retrieving the codes and their respective failures
        5. Store the codes, the failures, the model and the scaler on disk
        6. Get the final compression ratio we managed to achieve, which is our main evaluation metric
    Args:
        params: A dictionary of hyper-parameters (check main below for an example)

    """
    start_time = time.time()

    # Check if a CUDA enabled GPU exists
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.debug(f"Running with: {device}")

    # Read and preprocess the data
    logging.debug("Reading and preprocessing data...")
    raw_table = np.array(pd.read_csv(params['data_path'], header=None))
    quantized, scaler = ds_preprocessing(raw_table, params['error_threshold'], min_val=0, max_val=1)
    params['features'] = quantized.shape[1]  # Need to store feature number for decompression
    logging.debug("Done\n")

    # Create the model and send it to the GPU (if a GPU exists)
    logging.debug("Creating model...")
    ae = AutoEncoder(quantized.shape[1], params['code_size'], params['width_multiplier'], params['ae_depth'])
    ae.to(device)
    logging.debug("Done\n")

    # If the dataset is too big we must first sample it
    # Of course at the end we compress the whole file and not just the sample
    sample_data_size = int(min([params['sample_max_size'], len(quantized)]))
    sample_data_inds = np.random.choice(len(quantized), sample_data_size, replace=False)
    sample_data = quantized[sample_data_inds, :]

    # Train the autoencoder
    logging.debug("Training...")
    model, loss = train(ae, device, sample_data, epochs=params['epochs'],
                        batch_size=sample_data.shape[0] // params['batch_size'], lr=params['lr'])
    logging.debug(f"Training finished. Final loss: {float(loss):.3f}")

    # Set the model to eval mode
    model.eval()

    # Materialization step
    if params['binning_strategy'] == "POST_BINNING":
        codes, failures = materialize_with_post_binning(model, quantized, device, params['error_threshold'])
    elif params['binning_strategy'] == "BIN_DIFFERENCE":
        codes, failures = materialize_with_bin_difference(model, quantized, device, params['error_threshold'])
    elif params['binning_strategy'] == "NONE":
        codes, failures = materialize(model, quantized, device)
    else:
        raise ValueError("Available binning strategies: \"NONE\", "
                         "\"POST_BINNING\", \"BIN_DIFFERENCE\"")

    # Store the final file on disk
    comp_path = store_on_disk(params['compression_path'], model, codes, failures, scaler, params)

    total_time = time.time() - start_time

    # Log the final compression ratio DeepSqueeze achieved
    comp_ratio, comp_size, orig_size = calculate_compression_ratio(params['data_path'], comp_path)
    logging.debug(
        f"Compression ratio: {(comp_ratio * 100):.2f}% ({comp_size * 1e-6:.2f}MB / {orig_size * 1e-6:.2f}MB) | "
        f"Time: {total_time:.2f}s")

    return comp_ratio


if __name__ == '__main__':
    params = {
        "epochs": 1,
        "ae_depth": 2,  # Value in paper: 2
        "width_multiplier": 2,  # Value in paper: 2
        "batch_size": [1_000, 2_000],  # Optimized through bayesian optimization
        "lr": 1e-4,
        "code_size": [1, 3],  # Optimized through bayesian optimization
        "binning_strategy": "POST_BINNING",  # "NONE", "POST_BINNING", "BIN_DIFFERENCE",
        "sample_max_size": 2e5
    }

    # Parse the input arguments, of input file, output file and error threshold
    parser = argparse.ArgumentParser(description='Give the input, output and error threshold.')
    parser.add_argument('-i', '--input', type=str, help='path to input table', required=True)
    parser.add_argument('-o', '--output', type=str, help='path to compressed file', required=True)
    parser.add_argument('-e', '--error', type=float, help='Percentage [0, 100] of error allowed', required=True)

    args = parser.parse_args()
    params['data_path'] = args.input
    params['compression_path'] = args.output
    params['error_threshold'] = args.error / 100  # Transform percentage [0, 100] to [0,1] range

    # Getting starting date and time for logging
    today = datetime.now().strftime("%d_%m_%Y__%HH_%MM_%SS")

    # __________ Bayesian optimization run __________
    logging.info("Starting Bayesian Optimization, fine-tuning code size and batch size\n")
    best_params = minimize_comp_ratio(compression_pipeline, params)['params']

    # __________ Best parameters run __________
    logging.info("Creating final compressed file with the so far best parameters")
    for par, val in best_params.items():
        params[par] = int(val)  # Set the best parameters we found as the best parameters
    params['sample_max_size'] = math.inf
    comp_ratio, _ = compression_pipeline(params)
    logging.info(f"Finished. Final compression ratio: {(comp_ratio * 100):.2f}%")
