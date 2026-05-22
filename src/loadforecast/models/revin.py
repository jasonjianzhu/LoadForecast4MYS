from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RevINStats:
    mean: torch.Tensor
    stdev: torch.Tensor


class RevIN(nn.Module):
    """Reversible instance normalisation with optional affine parameters.

    Set ``per_station_affine=True`` to learn separate affine scales per series;
    otherwise a single global affine is shared across all stations."""

    def __init__(self, num_features: int, num_series: int = 1, eps: float = 1e-5,
                 affine: bool = True, per_station_affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.num_series = num_series
        self.eps = eps
        self.affine = affine
        self.per_station_affine = per_station_affine
        if affine:
            n = num_series if per_station_affine else 1
            self.affine_weight = nn.Parameter(torch.ones(n, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(n, 1, num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    def _select_affine(self, series_id: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.affine or self.affine_weight is None or self.affine_bias is None:
            return torch.ones(1, 1, self.num_features), torch.zeros(1, 1, self.num_features)
        if not self.per_station_affine or series_id is None:
            return self.affine_weight, self.affine_bias
        weight = self.affine_weight[series_id]
        bias = self.affine_bias[series_id]
        if weight.dim() == 1:
            weight = weight.unsqueeze(1)
        if bias.dim() == 1:
            bias = bias.unsqueeze(1)
        return weight, bias

    def _compute_stats(self, x: torch.Tensor) -> RevINStats:
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x - mean, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return RevINStats(mean=mean, stdev=stdev)

    def normalize(self, x: torch.Tensor, series_id: torch.Tensor | None = None) -> tuple[torch.Tensor, RevINStats]:
        stats = self._compute_stats(x)
        x_norm = (x - stats.mean) / stats.stdev
        if self.affine:
            weight, bias = self._select_affine(series_id)
            x_norm = x_norm * weight + bias
        return x_norm, stats

    def normalize_with_stats(self, x: torch.Tensor, stats: RevINStats, series_id: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = (x - stats.mean) / stats.stdev
        if self.affine:
            weight, bias = self._select_affine(series_id)
            x_norm = x_norm * weight + bias
        return x_norm

    def denormalize(self, x: torch.Tensor, stats: RevINStats, series_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.affine:
            weight, bias = self._select_affine(series_id)
            x = (x - bias) / (weight + self.eps)
        return x * stats.stdev + stats.mean

