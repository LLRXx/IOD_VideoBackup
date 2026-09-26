from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F


def _make_deform_conv(in_channels, out_channels, kernel_size, dilation=1,
                      groups=1):
    """Build a torchvision deformable convolution without importing it globally.

    The project can still run with DPDF disabled on environments that do not
    provide torchvision's compiled deformable-convolution operator.
    """
    try:
        from torchvision.ops import DeformConv2d
    except ImportError as exc:
        raise ImportError(
            'DPDF requires torchvision.ops.DeformConv2d. Install a '
            'torchvision build matching the project PyTorch/CUDA version.'
        ) from exc

    padding = (kernel_size // 2) * dilation
    return DeformConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


class _DeformConvBranch(nn.Module):
    """A deformable convolution with an explicitly learned offset field."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1):
        super(_DeformConvBranch, self).__init__()
        self.kernel_size = kernel_size
        self.offset = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            bias=True,
        )
        self.conv = _make_deform_conv(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
        )

        # Start from a regular sampling grid. This makes a newly enabled DPDF
        # close to an ordinary residual refinement instead of immediately
        # moving detector response peaks.
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x, offset_source=None):
        """Apply deformable convolution using an optional offset feature.

        The sampled feature remains ``x``.  When ``offset_source`` is
        provided, temporal context can guide where the current-frame feature
        is sampled without changing the detector's feature interface.
        """
        if offset_source is None:
            offset_source = x
        return self.conv(x, self.offset(offset_source))


class _TemporalAlign(nn.Module):
    """Align one adjacent-frame feature to the current-frame coordinates."""

    def __init__(self, channels, kernel_size=3):
        super(_TemporalAlign, self).__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                'Temporal alignment kernel must be a positive odd integer.')
        self.offset = nn.Conv2d(
            channels * 2,
            2 * kernel_size * kernel_size,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv = _make_deform_conv(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
        )

        # Alignment starts from the regular grid.  The outer DPDF projection
        # remains zero-initialized, so enabling V2 is still an exact identity
        # before fine-tuning.
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, neighbor, current):
        offset_source = torch.cat([current, neighbor], dim=1)
        return self.conv(neighbor, self.offset(offset_source))


class _SASPP(nn.Module):
    """Lightweight SASPP adapted from the paper's 512-channel example.

    The paper's figure shows 3x3 parallel branches at rates 1/6/12/18,
    while the preceding DPDF spatial convolution is specified as 5x5. The
    latter is exposed as ``dpdf_deform_kernel``; the SASPP branches remain
    3x3 and use narrow outputs so the 64-channel detector interface is kept.
    """

    def __init__(self, channels, branch_channels, dilation_rates):
        super(_SASPP, self).__init__()
        self.branches = nn.ModuleList([
            _DeformConvBranch(
                channels,
                branch_channels,
                kernel_size=3,
                dilation=rate,
            )
            for rate in dilation_rates
        ])
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.project = nn.Conv2d(
            branch_channels * (len(dilation_rates) + 1),
            channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        height, width = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        global_feature = self.global_branch(x)
        global_feature = F.interpolate(
            global_feature,
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        )
        features.append(global_feature)
        return self.project(torch.cat(features, dim=1))


class _ChannelMHSA(nn.Module):
    """Channel-wise multi-head self-attention.

    Attention is formed over C/heads channels for each spatial location, not
    over H*W tokens. This preserves the paper's channel-attention intent and
    avoids a quadratic 5184x5184 matrix at the project's 72x72 feature size.
    """

    def __init__(self, channels, heads):
        super(_ChannelMHSA, self).__init__()
        if channels % heads != 0:
            raise ValueError(
                'DPDF channels ({}) must be divisible by heads ({}).'.format(
                    channels, heads))
        self.channels = channels
        self.heads = heads
        self.head_channels = channels // heads
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x):
        batch, _, height, width = x.shape
        q, k, value = self.qkv(x).chunk(3, dim=1)
        q = q.reshape(batch, self.heads, self.head_channels, height * width)
        k = k.reshape(batch, self.heads, self.head_channels, height * width)
        value = value.reshape(
            batch, self.heads, self.head_channels, height * width)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = (attention * self.temperature.unsqueeze(0)).softmax(dim=-1)
        refined = torch.matmul(attention, value)
        refined = refined.reshape(batch, self.channels, height, width)
        return self.project(refined)


class DPDFAttention(nn.Module):
    """Shape-preserving DPDF attention for one fused detector frame.

    This is the paper's DPDF Attention module, rather than the larger
    two-block-per-stage DPDFA decoder. It expects and returns B,C,H,W.
    """

    def __init__(self, channels=64, heads=4, branch_channels=16,
                 deform_kernel=5, dilation_rates=(1, 6, 12, 18),
                 temporal=False, temporal_align=False):
        super(DPDFAttention, self).__init__()
        if deform_kernel <= 0 or deform_kernel % 2 == 0:
            raise ValueError('DPDF deform_kernel must be a positive odd integer.')
        if not dilation_rates or any(rate <= 0 for rate in dilation_rates):
            raise ValueError('DPDF dilation_rates must be positive integers.')

        self.proj1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        if temporal_align and not temporal:
            raise ValueError(
                'Temporal alignment requires temporal DPDF to be enabled.')
        self.temporal = temporal
        self.temporal_align = temporal_align
        if self.temporal:
            # Current feature plus two adjacent-frame differences.
            self.temporal_fuse = nn.Sequential(
                nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
                nn.GELU(),
            )
        if self.temporal_align:
            # The same aligner is shared by the previous and following frame:
            # both operations map one neighbor into the current coordinates.
            self.temporal_aligner = _TemporalAlign(channels)
            self.temporal_value_fuse = nn.Sequential(
                nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
                nn.GELU(),
            )
        self.spatial_deform = _DeformConvBranch(
            channels,
            channels,
            kernel_size=deform_kernel,
            groups=channels,
        )
        self.saspp = _SASPP(channels, branch_channels, dilation_rates)
        self.spatial_project = nn.Conv2d(channels, channels, kernel_size=1,
                                         bias=True)
        self.channel_attention = _ChannelMHSA(channels, heads)
        self.proj2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.activation = nn.GELU()

        # Keep the new block an exact residual identity at initialization.
        nn.init.zeros_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x, temporal_context=None, residual=None):
        if residual is None:
            residual = x
        projected = self.activation(self.proj1(x))
        deformable = self.spatial_deform(projected, temporal_context)
        spatial_map = self.spatial_project(self.saspp(deformable))
        spatial_refined = projected * torch.sigmoid(spatial_map)
        channel_refined = self.channel_attention(spatial_refined)
        return residual + self.proj2(channel_refined)

    def forward_sequence(self, chunk):
        """Refine K frame features with local temporal differences.

        ``chunk`` is a list of K tensors shaped [B,C,H,W].  Boundary frames
        use replicated neighbors, so the sequence length and detector branch
        contract remain unchanged.
        """
        if not self.temporal:
            return [self(feature) for feature in chunk]
        if not chunk:
            return []

        features = torch.stack(chunk, dim=1)  # [B,K,C,H,W]
        previous = torch.cat(
            [features[:, :1], features[:, :-1]], dim=1)
        following = torch.cat(
            [features[:, 1:], features[:, -1:]], dim=1)

        batch, steps, channels, height, width = features.shape
        current = features.reshape(batch * steps, channels, height, width)
        previous = previous.reshape(batch * steps, channels, height, width)
        following = following.reshape(batch * steps, channels, height, width)

        temporal_input = torch.cat([
            current,
            torch.abs(current - previous),
            torch.abs(following - current),
        ], dim=1)
        temporal_context = self.temporal_fuse(temporal_input)
        dpdf_input = current
        if self.temporal_align:
            aligned_previous = self.temporal_aligner(previous, current)
            aligned_following = self.temporal_aligner(following, current)
            dpdf_input = self.temporal_value_fuse(torch.cat([
                aligned_previous,
                current,
                aligned_following,
            ], dim=1))

        # Keep the detector-facing residual anchored to the current frame.
        # This preserves exact E0 equivalence for both temporal variants.
        refined = self.forward(
            dpdf_input,
            temporal_context,
            residual=current,
        )
        refined = refined.reshape(batch, steps, channels, height, width)
        return list(refined.unbind(dim=1))
