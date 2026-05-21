from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RevINStats:
    mean: torch.Tensor
    stdev: torch.Tensor


class RevIN(nn.Module):
    """Reversible instance normalization for a single target series."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    def _compute_stats(self, x: torch.Tensor) -> RevINStats:
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x - mean, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return RevINStats(mean=mean, stdev=stdev)

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, RevINStats]:
        stats = self._compute_stats(x)
        x_norm = (x - stats.mean) / stats.stdev
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm, stats

    def normalize_with_stats(self, x: torch.Tensor, stats: RevINStats) -> torch.Tensor:
        x_norm = (x - stats.mean) / stats.stdev
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm

    def denormalize(self, x: torch.Tensor, stats: RevINStats) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * stats.stdev + stats.mean

