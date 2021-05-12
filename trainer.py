import numpy as np
import pandas as pd
import logging
import torch

from datetime import datetime

from deep_squeeze.autoencoder import AutoEncoder
from deep_squeeze.preprocessing import ds_preprocessing
from deep_squeeze.train_loop import train
from deep_squeeze.materialization import materialize, materialize_with_post_binning
from deep_squeeze.disk_storing import store_on_disk

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(asctime)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')


if __name__ == '__main__':
    # Getting starting date and time for logging
    today = datetime.now().strftime("%d_%m_%Y__%HH_%MM_%SS")

    # Set experiment parameters
    params = {
        "data_path": "storage/datasets/corel_processed.csv",
        "epochs": 5,
        "batch_size": 64,
        "lr": 1e-4,
        "error_threshold": 0.005,
        "code_size": 1,
        "compression_path": f"storage/compressed/original/MSE_{today}/",
        "post_binning": True
    }

    # Check if a CUDA enabled GPU exists
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running with: {device}")

    # Read and preprocess the data
    logging.info("Reading and preprocessing data...")
    raw_table = np.array(pd.read_csv(params['data_path'], header=None))
    quantized, scaler = ds_preprocessing(raw_table, params['error_threshold'], min_val=0, max_val=1)
    logging.info("Done\n")

    # Create the model and sen it to the GPU
    logging.info("Creating model...")
    ae = AutoEncoder(quantized.shape[1], params['code_size'])
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
    if params['post_binning']:
        codes, failures = materialize_with_post_binning(model, quantized, device, params['error_threshold'])
    else:
        codes, failures = materialize(model, quantized, device)

    # Store the final file on disk
    store_on_disk(params['compression_path'], model, codes, failures, scaler, params)
