import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, features_numb, code_size, width_multiplier, length):
        super().__init__()

        # In the paper, the authors found that two hidden layers of size 2*features_numb seems to be working best.
        # We could experiment with more complex architectures.
        self.activation = nn.ReLU
        hidden_layer_width = width_multiplier * features_numb

        # Encoder
        self.enc_inp_layer = nn.Linear(in_features=features_numb, out_features=hidden_layer_width)
        self.enc_fc_layer_code = nn.Linear(in_features=hidden_layer_width, out_features=code_size)

        enc_layers = [self.enc_inp_layer, self.activation()]
        for _ in range(length - 1):
            enc_layers.append(nn.Linear(in_features=hidden_layer_width, out_features=hidden_layer_width))
            enc_layers.append(self.activation())
        enc_layers.append(self.enc_fc_layer_code)
        enc_layers.append(self.activation())

        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        self.dec_fc_layer1 = nn.Linear(in_features=code_size, out_features=hidden_layer_width)
        self.dec_fc_recon = nn.Linear(in_features=hidden_layer_width, out_features=features_numb)

        dec_layers = [self.dec_fc_layer1, self.activation()]
        for _ in range(length - 1):
            dec_layers.append(nn.Linear(in_features=hidden_layer_width, out_features=hidden_layer_width))
            dec_layers.append(self.activation())
        dec_layers.append(self.dec_fc_recon)

        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return x

