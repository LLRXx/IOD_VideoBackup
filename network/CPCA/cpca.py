from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Channel attention used by CPCA."""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        if channels <= 0:
            raise ValueError('channels must be positive')
        if reduction <= 0:
            raise ValueError('reduction must be positive')

        hidden_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_attention = self.shared_mlp(self.avg_pool(x))
        max_attention = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_attention + max_attention)


class _StripDepthwiseBranch(nn.Module):
    """Approximates a large square depthwise kernel with two strip kernels."""

    def __init__(self, channels, kernel_size):
        super(_StripDepthwiseBranch, self).__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError('CPCA kernel sizes must be positive odd numbers')

        padding = kernel_size // 2
        self.branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, kernel_size),
                padding=(0, padding),
                groups=channels,
                bias=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                groups=channels,
                bias=True))

    def forward(self, x):
        return self.branch(x)


class SpatialAttention(nn.Module):
    """Multi-scale, per-channel spatial attention followed by channel mixing."""

    def __init__(self, channels, kernel_sizes=(7, 11, 21)):
        super(SpatialAttention, self).__init__()
        if len(kernel_sizes) != 3:
            raise ValueError('CPCA expects exactly three spatial kernel sizes')

        self.depthwise_5x5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=True)
        self.branches = nn.ModuleList([
            _StripDepthwiseBranch(channels, kernel_size)
            for kernel_size in kernel_sizes
        ])
        self.channel_mixing = nn.Conv2d(
            channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        base_attention = self.depthwise_5x5(x)
        spatial_attention = base_attention
        for branch in self.branches:
            spatial_attention = spatial_attention + branch(base_attention)
        return self.channel_mixing(spatial_attention)


class CPCA(nn.Module):
    """Channel Prior Convolutional Attention from Huang et al."""

    def __init__(self, channels, reduction=16, kernel_sizes=(7, 11, 21)):
        super(CPCA, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(channels, kernel_sizes)

    def forward(self, x):
        channel_prior = self.channel_attention(x) * x
        spatial_attention = self.spatial_attention(channel_prior)
        return spatial_attention * channel_prior


class ResidualCPCA(nn.Module):
    """A shared residual adapter that preserves the incoming feature shape."""

    def __init__(self, channels, reduction=16, kernel_sizes=(7, 11, 21),
                 residual_scale=0.1):
        super(ResidualCPCA, self).__init__()
        self.cpca = CPCA(channels, reduction, kernel_sizes)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale), dtype=torch.float32))

    def forward(self, x):
        return x + self.residual_scale * self.cpca(x)