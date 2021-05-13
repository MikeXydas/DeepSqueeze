import torch
import logging
import numpy as np


def find_failures(x, recons):
    x = x.cpu().numpy()
    recons = recons.cpu().detach().numpy()

    return x - recons


def materialize(model, x, device):
    # Get the tensor form of our table and send it to the device
    x = torch.from_numpy(x).float().to(device)

    # Find the compressed codes that we will store
    codes = model.encoder(x)

    # Decode these codes to find failures
    recons = model.decoder(codes)

    # Finding the failures between the table and the reconstructions
    failures = find_failures(x, recons)

    return codes.cpu().detach().numpy(), failures


def materialize_with_post_binning(model, x, device, error_thr):
    # Get the tensor form of our table and send it to the device
    x = torch.from_numpy(x).float().to(device)

    # Find the compressed codes that we will store
    codes = model.encoder(x)

    # Decode these codes to find failures
    recons = model.decoder(codes)

    # Perform post-binning too, to improve the final compression
    post_binned = post_binning(recons, error_thr)

    # Finding the failures between the table and the reconstructions
    failures = x.cpu().numpy() - post_binned

    return codes.cpu().detach().numpy(), failures


def materialize_moe(moe, x, device, error_thr):
    x = torch.from_numpy(x).float().to(device)
    x_moe = x.reshape(x.shape[0], 1, x.shape[1])

    expert_codes, orig_shape, combine_tensor = moe.get_codes(x_moe)
    recons = moe.get_recons_from_codes(expert_codes, orig_shape, combine_tensor)

    # Perform post-binning too, to improve the final compression
    post_binned = post_binning(recons, error_thr)

    # Finding the failures between the table and the reconstructions
    failures = x.cpu().numpy() - post_binned

    return expert_codes.cpu().detach().numpy(), failures


def post_binning(recons, error_thr):
    recons = recons.cpu().detach().numpy()

    bins = np.arange(0, 1, 2 * error_thr)
    digitized = np.digitize(recons, bins)
    post_binned = (digitized - 1) * (2 * error_thr) + error_thr

    return post_binned
