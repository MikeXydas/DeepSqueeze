import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from deep_squeeze.autoencoder import AutoEncoder
from deep_squeeze.preprocessing import ds_preprocessing

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(asctime)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')


def train(model, device, quantized_data, epochs=30, batch_size=64, lr=1e-4):
    # Create a DataLoader
    train_loader = torch.utils.data.DataLoader(quantized_data,
                                               batch_size=batch_size,
                                               shuffle=True)

    # Set loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss = -1  # Initialization to avoid IDE warning

    for epoch in range(epochs):
        for original_rows in tqdm(train_loader, desc=f"Batch loss: {loss:.3f}"):
            # Get the batch data and put them to device
            original_rows = original_rows.float().to(device)

            # Perform forward pass
            recon_rows = model(original_rows)

            # Calculate loss
            loss = criterion(recon_rows, original_rows)

            # Calculate the gradients
            loss.backward()

            # Perform the gradient descent step and reset the gradients
            optimizer.step()
            optimizer.zero_grad()

        print(f"> Epoch: {epoch + 1} / {epochs} | MSE Train Loss: {float(loss):.3f}\n")

    return model, loss


if __name__ == '__main__':
    # Set experiment parameters
    params = {
        "data_path": "storage/datasets/corel_preprocessed.csv",
        "epochs": 30,
        "batch_size": 256,
        "lr": 1e-4,
        "error_threshold": 0.05,
        "code_size": 6
    }

    # Check if a CUDA enabled GPU exists
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running with: {device}")

    # Read the data and preprocess them
    logging.info("Reading and preprocessing data...")
    raw_table = np.array(pd.read_csv(params['data_path'], header=None))
    quantized, scaler = ds_preprocessing(raw_table, params['error_threshold'], min_val=0, max_val=1)
    logging.info("Done\n")

    logging.info("Creating model...")
    ae = AutoEncoder(quantized.shape[1], params['code_size'])
    ae.to(device)
    logging.info("Done\n")

    logging.info("Training...")
    model, loss = train(ae, device, quantized, epochs=params['epochs'],
          batch_size=params['batch_size'], lr=params['lr'])
    logging.info(f">>> Training finished. Final loss: {float(loss):.3f}")
