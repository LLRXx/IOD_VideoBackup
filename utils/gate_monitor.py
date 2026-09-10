from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch


def _percentile(sorted_values, fraction):
    """Linear-interpolated percentile compatible with older PyTorch."""
    if sorted_values.numel() == 0:
        return float('nan')
    position = (sorted_values.numel() - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower].item()
    weight = position - lower
    return ((1.0 - weight) * sorted_values[lower]
            + weight * sorted_values[upper]).item()


class GateStatistics(object):
    """Accumulate per-clip CPCA gate and effective-strength statistics."""

    def __init__(self):
        self.gates = []
        self.scales = []
        self.effective_strengths = []

    def update(self, gate, residual_scale):
        gate = gate.detach().float().reshape(-1).cpu()
        scale = residual_scale.detach().float().reshape(-1).cpu()
        if scale.numel() == 1:
            scale = scale.expand_as(gate)
        elif scale.numel() != gate.numel():
            raise ValueError(
                'residual_scale must be scalar or have one value per gate')

        self.gates.append(gate)
        self.scales.append(scale)
        self.effective_strengths.append(scale.abs() * gate)

    def update_from_output(self, output):
        if ('cpca_gate' not in output
                or 'cpca_residual_scale' not in output):
            return False
        self.update(output['cpca_gate'], output['cpca_residual_scale'])
        return True

    def __len__(self):
        return sum(values.numel() for values in self.gates)

    def summary(self):
        if not self.gates:
            return {}

        gate = torch.cat(self.gates)
        scale = torch.cat(self.scales)
        effective = torch.cat(self.effective_strengths)
        sorted_gate, _ = torch.sort(gate)
        std = gate.std(unbiased=False).item()

        return {
            'gate_count': int(gate.numel()),
            'gate_mean': gate.mean().item(),
            'gate_std': std,
            'gate_min': gate.min().item(),
            'gate_p10': _percentile(sorted_gate, 0.10),
            'gate_p25': _percentile(sorted_gate, 0.25),
            'gate_p50': _percentile(sorted_gate, 0.50),
            'gate_p75': _percentile(sorted_gate, 0.75),
            'gate_p90': _percentile(sorted_gate, 0.90),
            'gate_max': gate.max().item(),
            'gate_le_0.05_frac': (gate <= 0.05).float().mean().item(),
            'gate_le_0.10_frac': (gate <= 0.10).float().mean().item(),
            'gate_ge_0.90_frac': (gate >= 0.90).float().mean().item(),
            'gate_ge_0.95_frac': (gate >= 0.95).float().mean().item(),
            'cpca_residual_scale_first': scale[0].item(),
            'cpca_residual_scale_last': scale[-1].item(),
            'cpca_residual_scale_mean': scale.mean().item(),
            'cpca_effective_strength_mean': effective.mean().item(),
            'cpca_effective_strength_min': effective.min().item(),
            'cpca_effective_strength_max': effective.max().item(),
        }
