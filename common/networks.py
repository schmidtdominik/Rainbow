from functools import partial
from math import sqrt

import torch
from torch import nn as nn, Tensor
from torch.nn import init
import torch.nn.functional as F

class FactorizedNoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_0: float) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)

        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b
        return F.linear(input,
                        self.weight_mu + self.weight_sigma*self.weight_epsilon,
                        self.bias_mu + self.bias_sigma*self.bias_epsilon)


class Dueling(nn.Module):
    # only for 84x84 resolution
    def __init__(self, value_branch, advantage_branch):
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    def forward(self, x, advantages_only=False):
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))


class NatureCNN(nn.Module):
    # only for 84x84 resolution
    def __init__(self, depth, actions, linear_layer, resolution=None):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            linear_layer(3136, 512),
            nn.ReLU(),
            linear_layer(512, actions),
        )

    def forward(self, x, advantages_only=None):
        return self.main(x)


class DuelingNatureCNN(nn.Module):
    def __init__(self, depth, actions, linear_layer, resolution=None):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.dueling = Dueling(
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, 1)),
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, actions))
            )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNSmall(nn.Module):
    def __init__(self, depth, actions, linear_layer, resolution=None):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((6, 6))

        self.dueling = Dueling(
                nn.Sequential(linear_layer(1152, 256),
                              nn.ReLU(),
                              linear_layer(256, 1)),
                nn.Sequential(linear_layer(1152, 256),
                              nn.ReLU(),
                              linear_layer(256, actions))
            )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        f = self.pool(f)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNResidual(nn.Module):

    def __init__(self, depth):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv_0 = nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_ = self.conv_0(self.relu(x))
        x_ = self.conv_1(self.relu(x_))
        return x+x_


class ImpalaCNNBlock(nn.Module):

    def __init__(self, depth_in, depth_out):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out)
        self.residual_1 = ImpalaCNNResidual(depth_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class ImpalaCNNLarge(nn.Module):
    def __init__(self, in_depth, actions, linear_layer, resolution, model_size=1):
        super().__init__()

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size),
            ImpalaCNNBlock(16*model_size, 32*model_size),
            ImpalaCNNBlock(32*model_size, 32*model_size),
            nn.ReLU()
        )

        shape = self.main(torch.zeros(1, in_depth, resolution[0], resolution[1])).shape
        assert shape[0] == 1
        assert shape[1] == 32*model_size

        self.dueling = Dueling(
            nn.Sequential(linear_layer(shape[2]*shape[3]*32*model_size, 256),
                          nn.ReLU(),
                          linear_layer(256, 1)),
            nn.Sequential(linear_layer(shape[2]*shape[3]*32*model_size, 256),
                          nn.ReLU(),
                          linear_layer(256, actions))
        )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        return self.dueling(f, advantages_only=advantages_only)

def get_model(model_str):
    if model_str == 'nature': return NatureCNN              # like in Mnih et al 2015
    elif model_str == 'dueling': return DuelingNatureCNN    # nature + dueling architecture
    elif model_str == 'impala_small': return ImpalaCNNSmall # like the small model in IMPALA but with dueling and without LSTMs
    elif model_str.startswith('impala_large:'):             # like the large model in IMPALA but with dueling and without LSTMs
        return partial(ImpalaCNNLarge, model_size=int(model_str[13:]))