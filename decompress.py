import argparse
import pandas as pd
import logging
import torch
import numpy as np

from deep_squeeze.disk_storing import load_files
from deep_squeeze.materialization import codes_to_table

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(asctime)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')


if __name__ == '__main__':
    # Parse the input arguments, of input file, output file and error threshold
    parser = argparse.ArgumentParser(description='Give as input the compressed file.')
    parser.add_argument('-i', '--input', type=str, help='path to input compressed file', required=True)

    args = parser.parse_args()
    comp_file = args.input

    model, codes, failures, scaler = load_files(comp_file)

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
    print(f">>> Storing table on {comp_file[:-4]}.csv...", end='')
    table_df = pd.DataFrame(np.round(rescaled_arr, 3))
    table_df.to_csv(f"{comp_file[:-4]}.csv", index=False, header=False)
    print("Done")
