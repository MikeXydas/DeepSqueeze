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
        for batch_ind, original_rows in enumerate(tqdm(train_loader)):
            # Get the batch data and put them to device
            original_rows = original_rows.float().to(device)

            # Perform forward pass
            recon_rows = model(original_rows)

            # Calculate loss
            loss = criterion(recon_rows, original_rows)
            # loss = step_mse_to_mae(recon_rows, original_rows, epoch / epochs, 0.1)

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


# Below we define some custom losses that replace the default MSE loss of the paper
def linear_mse_to_mae(recon, orig, epoch_frac):
    mse_loss = torch.mean((recon - orig) ** 2)
    mae_loss = torch.mean(torch.abs(recon - orig))

    # Beta chooses which of the two losses will have more impact
    # Initially, we focus more on MSE and then on MAE which aims at exact value match
    beta = epoch_frac

    return beta * mae_loss + (1 - beta) * mse_loss


def step_mse_to_mae(recon, orig, epoch_frac, mae_coef):
    if epoch_frac < 0.5:
        return torch.mean((recon - orig) ** 2)
    else:
        return mae_coef * torch.mean(torch.abs(recon - orig))
