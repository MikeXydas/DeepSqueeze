from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import torch
import logging
import json
import shutil
import os
import zipfile
import json

from deep_squeeze.autoencoder import AutoEncoder


def store_on_disk(path, model, codes, failures, scaler, hyper_params):
    """
    Our goal is to compress a file as much as possible meaning that our final evaluation
    will be the size of the final file.

    The final size consists of:
    * The decoder weights
    * The code (lower dimensional representation) of each row in the table
    * The failures
    * The minmax scaler
    """
    # Create the directory that we will store our model in
    # Will throw a FileExistsError if the directory already exists
    Path(path).mkdir(parents=True, exist_ok=False)

    # Get the state dict of the model
    torch.save(model.state_dict(), path + "model.pth")

    # Store the codes in a parquet file
    codes_df = pd.DataFrame(codes, columns=None)
    codes_df.columns = codes_df.columns.astype(str)
    codes_df.to_parquet(path + "codes.parquet", index=False, compression='brotli')

    # Store the failures in a parquet file
    failures_df = pd.DataFrame(failures, columns=None)
    failures_df.columns = failures_df.columns.astype(str)
    failures_df.to_parquet(path + "failures.parquet", index=False, compression='brotli')

    # Store the scaler
    joblib.dump(scaler, path + 'scaler.pkl')

    # Store run hyper-parameters (not actually needed)
    with open(path + 'hyper_params.json', 'w') as outfile:
        json.dump(hyper_params, outfile)

    # Create a zipfile from the temporary folder that keeps our compression data
    shutil.make_archive(path[:-1], 'zip', path)

    # Delete the temporary folder
    shutil.rmtree(path)

    logging.debug(f"Stored files in {path[:-1]}.zip")

    return path[:-1] + '.zip'


def calculate_compression_ratio(original_file_path, compressed_file_path):
    orig_size = os.path.getsize(original_file_path)
    compress_size = os.path.getsize(compressed_file_path)

    return compress_size / orig_size, compress_size, orig_size


def unzip_file(path):
    temp_path = f"{path[:-4]}_temp"
    # Extract the zip file to a temporary folder
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(temp_path)

    with open(f'{temp_path}/hyper_params.json') as f:
        hyper_params = json.load(f)

    return hyper_params, temp_path


def load_model(folder_path, model):
    model.load_state_dict(torch.load(f"{folder_path}/model.pth"))
    model.eval()

    return model


def load_codes_failures(folder_path):
    codes = np.array(pd.read_parquet(f"{folder_path}/codes.parquet"))
    failures = np.array(pd.read_parquet(f"{folder_path}/failures.parquet"))

    return codes, failures


def load_scaler(folder_path):
    return joblib.load(f"{folder_path}/scaler.pkl")


def load_files(comp_path):
    # Unzip the file and load the hyper parameters
    hyper_params, folder_path = unzip_file(comp_path)

    # Initialize an autoencoder that we will load the parameters into
    ae = AutoEncoder(hyper_params['features'], hyper_params['code_size'])

    # Load model, codes, failures and scaler
    ae = load_model(folder_path, ae)
    codes, failures = load_codes_failures(folder_path)
    scaler = load_scaler(folder_path)

    # Since we have loaded everything we need, delete the temp folder
    shutil.rmtree(folder_path + "/")

    return ae, codes, failures, scaler
