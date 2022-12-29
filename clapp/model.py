from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):  # for approximating polynomial activation
    def forward(self, x):
        return F.relu(x) ** 2


ACTIVATION = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "squared_relu": SquaredReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "swish": nn.SiLU,
    "none": nn.Identity,
}


@dataclass
class ModelConfig:
    capacity: int = 8
    num_layers: int = 4
    input_channels: int = 3
    output_channels: int = 1
    kernel_size: int = 3
    activation: str = "gelu"
    # some tricks
    flip_conv_norm: bool = True
    rezero: bool = True
    reskip: bool = True


class FilterBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.flip = config.flip_conv_norm
        self.rezero = config.rezero
        self.reskip = config.reskip
        padding = config.kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                config.capacity,
                config.capacity,
                config.kernel_size,
                padding=0,
            ),
            nn.ConstantPad2d(padding, 0),
        )
        self.norm = nn.GroupNorm(1, config.capacity)
        self.relu = ACTIVATION[config.activation]()
        self.alfa = nn.Parameter(torch.ones(1, config.capacity, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, config.capacity, 1, 1))

    def forward(self, x):
        if not self.flip:
            y = self.conv(y)
            y = self.norm(y)
        else:
            y = self.norm(x)
            y = self.conv(y)
        y = self.relu(y)
        if self.reskip and self.rezero:
            z = self.alfa * y + x * self.beta
        elif self.reskip:
            z = y + x * self.beta
        elif self.rezero:
            z = self.alfa * y + x
        else:
            z = y
        return z


class FilterNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.prep = nn.Conv2d(  # 1x1 conv to convert from input to capacity
            config.input_channels,
            config.capacity,
            kernel_size=1,
        )
        self.main = nn.Sequential(  # stack of filter blocks
            *[FilterBlock(config) for _ in range(config.num_layers)]
        )
        self.post = nn.Conv2d(  # 1x1 conv to convert from capacity to output
            config.capacity,
            config.output_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.main(x)
        x = self.post(x)
        return x
