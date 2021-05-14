import numpy as np
import pandas as pd
import logging
import torch

from datetime import datetime

from deep_squeeze.disk_storing import load_files
from deep_squeeze.materialization import codes_to_table

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(asctime)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')


if __name__ == '__main__':
    outfile = "storage/compressed/corel_comp.zip"

    model, codes, failures, scaler = load_files(outfile)

    # If a CUDA enabled GPU exists, send both the codes and the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running with: {device}")
    model.to(device)
    codes = torch.from_numpy(codes).to(device)

    # Get a numpy array of the model_recons + failures
    print(">>> Getting reconstructions from codes...", end='')
    decompressed_arr = codes_to_table(model, codes, failures)
    print("Done")

    # Invert the minmax scaling
    print(">>> Inverting the minmax scaling...", end='')
    rescaled_arr = scaler.inverse_transform(decompressed_arr)
    print("Done")

    # Store the final decompressed array as a csv on disk
    print(f">>> Storing table on {outfile[:-4]}.csv...", end='')
    table_df = pd.DataFrame(rescaled_arr)
    table_df.to_csv(f"{outfile[:-4]}.csv", index=False)
    print("Done")
