import numpy as np
import pandas as pd
import logging
import torch

from datetime import datetime

from deep_squeeze.autoencoder import AutoEncoder
from deep_squeeze.preprocessing import ds_preprocessing
from deep_squeeze.train_loop import train
from deep_squeeze.materialization import materialize, materialize_with_post_binning, \
    materialize_with_bin_difference
from deep_squeeze.disk_storing import store_on_disk, calculate_compression_ratio
from deep_squeeze.experiment import repeat_n_times,display_compression_results

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(asctime)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')
compression_repeats = 5


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
    # Check if a CUDA enabled GPU exists
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running with: {device}")

    # Read and preprocess the data
    logging.info("Reading and preprocessing data...")
    raw_table = np.array(pd.read_csv(params['data_path'], header=None))
    quantized, scaler = ds_preprocessing(raw_table, params['error_threshold'], min_val=0, max_val=1)
    params['features'] = quantized.shape[1]  # Need to store feature number for decompression
    logging.info("Done\n")

    # Create the model and send it to the GPU (if a GPU exists)
    logging.info("Creating model...")
    ae = AutoEncoder(quantized.shape[1], params['code_size'], params['width_multiplier'], params['ae_depth'])
    ae.to(device)
    logging.info("Done\n")

    # Train the autoencoder
    logging.info("Training...")
    model, loss = train(ae, device, quantized, epochs=params['epochs'],
                        batch_size=params['batch_size'], lr=params['lr'])
    logging.info(f"Training finished. Final loss: {float(loss):.3f}")

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

    # Log the final compression ratio DeepSqueeze achieved
    comp_ratio, comp_size, orig_size = calculate_compression_ratio(params['data_path'], comp_path)
    logging.info(f"Compression ratio: {(comp_ratio*100):.2f}% ({comp_size*1e-6:.2f}MB / {orig_size*1e-6:.2f}MB)")

    return comp_ratio


if __name__ == '__main__':
    # Getting starting date and time for logging
    today = datetime.now().strftime("%d_%m_%Y__%HH_%MM_%SS")

    # Set experiment parameters
    params = {
        "data_path": "storage/datasets/corel_processed.csv",
        "epochs": 1,
        "ae_depth": 2,  # Value in paper: 2
        "width_multiplier": 2,  # Value in paper: 2
        "batch_size": 64,
        "lr": 1e-4,
        "error_threshold": 0.005,
        "code_size": 1,
        "compression_path": f"storage/compressed/MSE_{today}/",
        "binning_strategy": "BIN_DIFFERENCE"  # "NONE", "POST_BINNING", "BIN_DIFFERENCE"
    }

    # Run the full pipeline (check if compression_pipeline is decorated)
    mean_ratio, std_ratio = compression_pipeline(params)
    display_compression_results(mean_ratio, std_ratio, compression_repeats)

