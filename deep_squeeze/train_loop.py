import logging

from tqdm import tqdm

import torch
import torch.nn as nn


def train(model, device, quantized_data, epochs=30, batch_size=64, lr=1e-4):
    # Create a DataLoader
    train_loader = torch.utils.data.DataLoader(quantized_data,
                                               batch_size=batch_size,
                                               shuffle=True)

    # Set loss function and optimizer
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    for epoch in range(epochs):
        epoch_loss = 0
        for original_rows in tqdm(train_loader):
            # Get the batch data and put them to device
            original_rows = original_rows.float().to(device)

            # Perform forward pass
            recon_rows = model(original_rows)

            # Calculate loss
            loss = criterion(recon_rows, original_rows)
            # loss = custom_loss(recon_rows, original_rows, epoch / epochs)

            # Calculate the gradients
            loss.backward()

            # Perform the gradient descent step and reset the gradients
            optimizer.step()
            optimizer.zero_grad()

            # Book-keeping
            epoch_loss += loss

        logging.debug(f"Epoch: {epoch + 1} / {epochs} | "
                      f"{type(criterion).__name__}: {float(epoch_loss / len(train_loader)):.3f}\n")

    return model, epoch_loss / len(train_loader)


def custom_loss(recon, orig, epoch_frac):
    mse_loss = torch.mean((recon - orig) ** 2)
    mae_loss = torch.mean(torch.abs(recon - orig))

    # Beta chooses which of the two losses will have more impact
    # Initially, we focus more on MSE and then on MAE which aims at exact value match
    beta = epoch_frac

    return beta * mae_loss + (1 - beta) * mse_loss
