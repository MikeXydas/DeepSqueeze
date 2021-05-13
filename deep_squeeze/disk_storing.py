from pathlib import Path

import pandas as pd
import joblib
import torch
import logging
import json


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
    if len(codes) == 2:
        codes_df = pd.DataFrame(codes, columns=None)
        codes_df.columns = codes_df.columns.astype(str)
        codes_df.to_parquet(path + "codes.parquet", index=False)
    else:
        for ind, expert_code in enumerate(codes):
            codes_df = pd.DataFrame(expert_code, columns=None)
            codes_df.columns = codes_df.columns.astype(str)
            codes_df.to_parquet(path + f"codes_{ind}.parquet", index=False)

    # Store the failures in a parquet file
    failures_df = pd.DataFrame(failures, columns=None)
    failures_df.columns = failures_df.columns.astype(str)
    failures_df.to_parquet(path + "failures.parquet", index=False)

    # Store the scaler
    joblib.dump(scaler, path + 'scaler.pkl')

    # Store run hyper-parameters (not actually needed)
    with open(path + 'hyper_params.json', 'w') as outfile:
        json.dump(hyper_params, outfile)

    logging.info(f"Stored files in {path}.")
