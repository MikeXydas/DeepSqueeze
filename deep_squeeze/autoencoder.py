import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_squeeze.mixture_of_experts import MoE


class AutoEncoder(nn.Module):
    def __init__(self, features_numb, code_size):
        super().__init__()

        # In the paper, the authors found that two hidden layers of size 2*features_numb seems to be working best.
        # We could experiment with more complex architectures.

        # Encoder
        self.enc_fc_layer1 = nn.Linear(in_features=features_numb, out_features=2 * features_numb)
        self.enc_fc_layer2 = nn.Linear(in_features=2 * features_numb, out_features=2 * features_numb)
        self.enc_fc_layer_code = nn.Linear(in_features=2 * features_numb, out_features=code_size)

        self.encoder = nn.Sequential(
            self.enc_fc_layer1,
            nn.ReLU(),
            self.enc_fc_layer2,
            nn.ReLU(),
            self.enc_fc_layer_code,
            nn.ReLU()
        )

        # Decoder
        self.dec_fc_layer1 = nn.Linear(in_features=code_size, out_features=2 * features_numb)
        self.dec_fc_layer2 = nn.Linear(in_features=2 * features_numb, out_features=2 * features_numb)
        self.dec_fc_layer3 = nn.Linear(in_features=2 * features_numb, out_features=features_numb)

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


class AutoencoderExperts(nn.Module):
    def __init__(self, features_numb, code_size, experts_numb):
        super().__init__()

        class ExpertEncoders(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = nn.Parameter(init_(torch.zeros(experts_numb, features_numb, features_numb * 2)))
                self.enc2 = nn.Parameter(init_(torch.zeros(experts_numb, features_numb * 2, features_numb * 2)))
                self.enc_codes = nn.Parameter(init_(torch.zeros(experts_numb, features_numb * 2, code_size)))
                self.activation = nn.ReLU()

            def forward(self, x):
                h1 = self.activation(torch.einsum('end,edh->enh', x, self.enc1))
                h2 = self.activation(torch.einsum('end,edh->enh', h1, self.enc2))
                codes = self.activation(torch.einsum('end,edh->enh', h2, self.enc_codes))

                return codes

        class ExpertDecoders(nn.Module):
            def __init__(self):
                super().__init__()
                self.dec1 = nn.Parameter(init_(torch.zeros(experts_numb, code_size, features_numb * 2)))
                self.dec2 = nn.Parameter(init_(torch.zeros(experts_numb, features_numb * 2, features_numb * 2)))
                self.dec3 = nn.Parameter(init_(torch.zeros(experts_numb, features_numb * 2, features_numb)))
                self.activation = nn.ReLU()

            def forward(self, x):
                h1 = self.activation(torch.einsum('end,edh->enh', x, self.dec1))
                h2 = self.activation(torch.einsum('end,edh->enh', h1, self.dec2))
                reconstructions = self.activation(torch.einsum('end,edh->enh', h2, self.dec3))

                return reconstructions

        self.expert_encoders = ExpertEncoders()
        self.expert_decoders = ExpertDecoders()

    def forward(self, x):
        codes = self.expert_encoders(x)
        reconstructions = self.expert_decoders(codes)

        return reconstructions


def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)


def test():
    moe = MoE(
        dim=3,
        num_experts=4,
        second_policy_train='none',
        second_policy_eval='none',
        capacity_factor_eval=1,
        capacity_factor_train=1
    )

    inputs = torch.ones(2, 1, 3)
    out, aux_loss = moe(inputs)  # (4, 1024, 512), (1,)
    dispatch_tensor, combine_tensor, loss = moe.gate(inputs)
