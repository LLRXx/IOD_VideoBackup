"""Spatial Channel Attention Module (SCAM).

This implementation follows the two sequential attention blocks described in
SCAM-P and wraps them in a fixed-scale residual branch for safe adaptation of
an already trained detector.
"""

from __future__ import absolute_import, division, print_function

import torch
from torch import nn
import torch.nn.functional as F


class SCAM(nn.Module):
    """Spatially localized channel attention followed by channel-localized
    spatial attention.

    The module preserves the input shape [N, C, H, W].  The spatial pooling
    uses non-overlapping windows and nearest-neighbor upsampling; the channel
    pooling averages contiguous channel groups and repeats each group gate.
    """

    def __init__(self, channels, reduction=16, spatial_kernel=4,
                 channel_group=4):
        super(SCAM, self).__init__()
        if channels <= 0:
            raise ValueError('SCAM channels must be positive')
        if reduction <= 0:
            raise ValueError('SCAM reduction must be positive')
        if spatial_kernel <= 0:
            raise ValueError('SCAM spatial kernel must be positive')
        if channel_group <= 0 or channels % channel_group != 0:
            raise ValueError(
                'SCAM channel_group must divide the input channel count')

        hidden = max(channels // reduction, 1)
        self.channels = channels
        self.spatial_kernel = spatial_kernel
        self.channel_group = channel_group
        self.channel_groups = channels // channel_group

        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        if x.dim() != 4 or x.size(1) != self.channels:
            raise ValueError(
                'SCAM expected [N, {}, H, W], got {}'.format(
                    self.channels, tuple(x.shape)))

        n, _, h, w = x.shape
        pooled = F.avg_pool2d(
            x,
            kernel_size=(self.spatial_kernel, self.spatial_kernel),
            stride=(self.spatial_kernel, self.spatial_kernel))
        channel_gate = torch.sigmoid(self.channel_attention(pooled))
        channel_gate = F.interpolate(
            channel_gate, size=(h, w), mode='nearest')
        refined = x * channel_gate

        grouped = refined.reshape(
            n, self.channel_groups, self.channel_group, h, w).mean(dim=2)
        spatial_gate = torch.sigmoid(grouped)
        spatial_gate = spatial_gate.repeat_interleave(
            self.channel_group, dim=1)
        return refined * spatial_gate


class ResidualSCAM(nn.Module):
    """Identity-preserving SCAM branch with a fixed residual scale."""

    def __init__(self, channels, reduction=16, spatial_kernel=4,
                 channel_group=4, residual_scale=0.1):
        super(ResidualSCAM, self).__init__()
        if residual_scale < 0:
            raise ValueError('SCAM residual scale must be non-negative')
        self.scam = SCAM(
            channels, reduction=reduction,
            spatial_kernel=spatial_kernel, channel_group=channel_group)
        self.register_buffer(
            'residual_scale', torch.tensor(float(residual_scale)))

    def forward(self, x):
        return x + self.residual_scale * self.scam(x)
