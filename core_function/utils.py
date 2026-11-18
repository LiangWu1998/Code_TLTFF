'''=================================================
@Author : Liang Wu
@Email  : liangwu1998@163.com
@Date   : 2025/4/10 17:24
=================================================='''

import torch
from torch import nn, optim
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega, bias=True,
                 is_first=False, use_sine=True):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.use_sine = use_sine
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.use_sine:
                if self.is_first:
                    self.linear.weight.uniform_(-1 / self.in_features,
                                                1 / self.in_features)
                else:
                    self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega,
                                                np.sqrt(6 / self.in_features) / self.omega)
            else:
                n_in = self.linear.in_features
                n_out = self.linear.out_features
                self.linear.weight.uniform_(-np.sqrt(6.0 / (n_in + n_out)),
                                            np.sqrt(6.0 / (n_in + n_out)))
            # self.linear.bias.zero_()

    def forward(self, input):
        input = self.linear(input)
        return torch.sin(self.omega * input) if self.use_sine else input


class Network(nn.Module):
    def __init__(self, n_1, n_2, n_3, shrink, expand, omega_list):
        super(Network, self).__init__()
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.shrink = shrink
        self.expand = expand
        self.r = int(self.n_2 // self.shrink)
        self.n_3_hat = int(self.n_3 * self.expand)
        self.mid_channel_1 = int(1.2 * self.r * self.n_3_hat)
        self.mid_channel_2 = int(1.2 * self.n_3_hat)
        self.omega_one = omega_list

        self.A_hat_matrix = nn.Sequential(SineLayer(1, self.mid_channel_1, self.omega_one, is_first=True),
                                          SineLayer(self.mid_channel_1, self.mid_channel_1, self.omega_one, is_first=True),
                                          nn.Linear(self.mid_channel_1, self.r * self.n_3_hat, bias=True))
        # SineLayer(self.mid_channel, self.r * self.n_3_hat, self.omega_other, use_sine=False, bias=True))
        self.B_hat_matrix = nn.Sequential(SineLayer(1, self.mid_channel_1, self.omega_one, is_first=True),
                                          SineLayer(self.mid_channel_1, self.mid_channel_1, self.omega_one, is_first=True),
                                          nn.Linear(self.mid_channel_1, self.r * self.n_3_hat, bias=True))

        self.nonlinear_transform = nn.Sequential(nn.Linear(self.n_3_hat, self.mid_channel_2, bias=True),
                                                 nn.LeakyReLU(),
                                                 nn.Linear(self.mid_channel_2, self.n_3_hat, bias=True),
                                                 nn.LeakyReLU())
                                                    # nn.LeakyReLU()

        self.weight_matrix = nn.Sequential(SineLayer(1, self.mid_channel_2, self.omega_one, is_first=True),
                                           SineLayer(self.mid_channel_2, self.mid_channel_2, self.omega_one, is_first=True),
                                           nn.Linear(self.mid_channel_2, self.n_3_hat, bias=True))

    def forward(self, n_1_input, n_2_input, n_3_input):
        A_hat = self.A_hat_matrix(n_1_input)
        A_hat = A_hat.reshape(self.n_1, self.r, self.n_3_hat)
        B_hat = self.B_hat_matrix(n_2_input)
        B_hat = B_hat.reshape(self.n_2, self.r, self.n_3_hat)
        B_hat = B_hat.permute(1, 0, 2)
        # B_hat = B_hat.reshape(self.n_2, self.n_3_hat, self.r)
        # B_hat = B_hat.permute(2, 0, 1)
        W_weight = self.weight_matrix(n_3_input)
        X_hat_mid = torch.einsum('ilk, ljk -> ijk', A_hat, B_hat)
        X_hat = self.nonlinear_transform(X_hat_mid)
        return X_hat @ W_weight.t()
