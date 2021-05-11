import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, features_numb, code_size):
        super().__init__()

        # In the paper, the authors found that two hidden layers of size 2*features_numb seems to be working best.
        # We could experiment with more complex architectures.

        # Encoder
        self.enc_fc_layer1 = nn.Linear(in_features=features_numb, out_features=2*features_numb)
        self.enc_fc_layer2 = nn.Linear(in_features=2*features_numb, out_features=2*features_numb)
        self.enc_fc_layer_code = nn.Linear(in_features=2*features_numb, out_features=code_size)

        self.encoder = nn.Sequential(
            self.enc_fc_layer1,
            nn.ReLU(),
            self.enc_fc_layer2,
            nn.ReLU(),
            self.enc_fc_layer_code,
            nn.ReLU()
        )

        # Decoder
        self.dec_fc_layer1 = nn.Linear(in_features=code_size, out_features=2*features_numb)
        self.dec_fc_layer2 = nn.Linear(in_features=2*features_numb, out_features=2*features_numb)
        self.dec_fc_layer3 = nn.Linear(in_features=2*features_numb, out_features=features_numb)

        self.decoder = nn.Sequential(
            self.dec_fc_layer1,
            nn.ReLU(),
            self.dec_fc_layer2,
            nn.ReLU(),
            self.dec_fc_layer3,
            nn.Sigmoid()
        )

    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return x

