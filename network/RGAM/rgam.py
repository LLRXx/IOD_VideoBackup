"""Residual Grouped Attention Module (RGAM).

The module follows the RCAM/RSAM decomposition described in the RGAM paper.
It is intentionally self contained so that the TEA backbone can keep its
original execution path when RGAM is disabled.
"""

from __future__ import absolute_import, division, print_function

import torch
from torch import nn
import torch.nn.functional as F


class _SharedMLP(nn.Module):
    def __init__(self, channels, reduction):
        super(_SharedMLP, self).__init__()
        hidden = max(channels // reduction, 1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class RCAM(nn.Module):
    """Residual channel attention using shared dilated depthwise kernels."""

    def __init__(self, channels, reduction=16):
        super(RCAM, self).__init__()
        self.channels = channels
        # One kernel is shared by the three dilation rates, as in RCAM.
        self.shared_kernel = nn.Parameter(torch.empty(1, 1, 3, 3))
        nn.init.kaiming_normal_(self.shared_kernel, mode='fan_out', nonlinearity='relu')
        self.downsample = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1,
            groups=channels, bias=False)
        self.mlp = _SharedMLP(channels, reduction)

    def forward(self, x):
        mixed = 0.0
        # Applying the shared kernel independently to every channel preserves
        # the depthwise behavior while sharing weights over dilation rates.
        for dilation in (1, 2, 4):
            mixed = mixed + F.conv2d(
                x, self.shared_kernel.expand(self.channels, 1, 3, 3),
                padding=dilation, dilation=dilation, groups=self.channels)
        pooled = F.adaptive_avg_pool2d(self.downsample(mixed), 1)
        return torch.sigmoid(self.mlp(pooled))


class _AxisMLP(nn.Module):
    def __init__(self, length, reduction):
        super(_AxisMLP, self).__init__()
        hidden = max(length // reduction, 1)
        self.net = nn.Sequential(
            nn.Linear(length, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, length),
        )

    def forward(self, x):
        return self.net(x)


class RSAM(nn.Module):
    """Grouped spatial attention with height/width axis MLPs."""

    def __init__(self, channels, groups=4, spatial_size=(18, 18), reduction=4):
        super(RSAM, self).__init__()
        if channels % groups != 0:
            raise ValueError('RGAM groups must divide the channel count')
        height, width = spatial_size
        if height <= 0 or width <= 0:
            raise ValueError('RGAM spatial dimensions must be positive')
        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        self.height = height
        self.width = width
        self.height_mlp = _AxisMLP(height, reduction)
        self.width_mlp = _AxisMLP(width, reduction)

    def forward(self, x):
        n, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(
                'RGAM expected {} channels, got {}'.format(self.channels, c))
        grouped = x.view(n, self.groups, self.group_channels, h, w).mean(dim=2)
        height_descriptor = grouped.mean(dim=3)  # [N, groups, H]
        width_descriptor = grouped.mean(dim=2)   # [N, groups, W]
        # The paper uses fixed axis FCs. Interpolation keeps those weights
        # usable when a different input resolution is selected at inference.
        height_descriptor = F.interpolate(
            height_descriptor.reshape(n * self.groups, 1, h),
            size=self.height, mode='linear', align_corners=False).reshape(
                n, self.groups, self.height)
        width_descriptor = F.interpolate(
            width_descriptor.reshape(n * self.groups, 1, w),
            size=self.width, mode='linear', align_corners=False).reshape(
                n, self.groups, self.width)
        height_attention = torch.sigmoid(self.height_mlp(height_descriptor))
        width_attention = torch.sigmoid(self.width_mlp(width_descriptor))
        height_attention = F.interpolate(
            height_attention.reshape(n * self.groups, 1, self.height),
            size=h, mode='linear', align_corners=False).reshape(n, self.groups, h)
        width_attention = F.interpolate(
            width_attention.reshape(n * self.groups, 1, self.width),
            size=w, mode='linear', align_corners=False).reshape(n, self.groups, w)
        return (height_attention.unsqueeze(-1) * width_attention.unsqueeze(-2)).repeat_interleave(
            self.group_channels, dim=1)


class RGAM(nn.Module):
    """Combined channel and grouped spatial attention."""

    def __init__(self, channels, groups=4, channel_reduction=16,
                 spatial_reduction=4, spatial_size=(18, 18)):
        super(RGAM, self).__init__()
        self.rcam = RCAM(channels, reduction=channel_reduction)
        self.rsam = RSAM(
            channels, groups=groups, spatial_size=spatial_size,
            reduction=spatial_reduction)

    def forward(self, x):
        return x * self.rcam(x) * self.rsam(x)


class ResidualRGAM(nn.Module):
    """Identity-preserving RGAM used by the optional backbone branch."""

    def __init__(self, channels, groups=4, channel_reduction=16,
                 spatial_reduction=4, spatial_size=(18, 18), residual_scale=0.1):
        super(ResidualRGAM, self).__init__()
        self.rgam = RGAM(
            channels, groups=groups, channel_reduction=channel_reduction,
            spatial_reduction=spatial_reduction, spatial_size=spatial_size)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale), dtype=torch.float32))

    def forward(self, x):
        return x + self.residual_scale * self.rgam(x)
