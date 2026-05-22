from __future__ import annotations

from dataclasses import dataclass

import torch

from loadforecast.models.revin import RevINStats


@dataclass
class ForwardOutput:
    prediction: torch.Tensor
    prediction_norm: torch.Tensor
    revin_stats: RevINStats
