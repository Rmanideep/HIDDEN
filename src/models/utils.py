# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Residual Block with GroupNorm.
    """
    def __init__(self, channels):
        super().__init__()
        num_groups = min(8, channels)  # Safe for any channel count
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.gn1(self.conv1(x)), 0.2)
        out = self.gn2(self.conv2(out))
        out += residual
        return F.leaky_relu(out, 0.2)
