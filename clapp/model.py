from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):  # for approximating polynomial activation
    def forward(self, x):
        return F.relu(x) ** 2


@dataclass
class ModelConfig:
    capacity: int = 4
    num_layers: int = 3
    input_channels: int = 3
    output_channels: int = 1
    kernel_size: int = 3
    

class FilterBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.conv = nn.Conv2d(
            config.capacity,
            config.capacity,
            config.kernel_size,
            padding=0,
        )
        self.norm = nn.GroupNorm(1, config.capacity)
        self.relu = SquaredReLU()
        self.alfa = nn.Parameter(torch.ones(1))

    def forward(self, x):
        y = F.pad(x, (1, 1, 1, 1))  # pad before applying conv
        y = self.conv(x)
        y = self.norm(y)
        y = self.relu(y)
        z = self.alfa * y + x  # ReZero Trick
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
            *[
                FilterBlock(config)
                for _ in range(config.num_layers)
            ]
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
