from __future__ import annotations

import torch
from torch import nn

from loadforecast.config import ModelConfig
from loadforecast.models.common import ForwardOutput
from loadforecast.models.revin import RevIN, RevINStats


class ConvFFN(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModernTCNBlock(nn.Module):
    """ModernTCN-inspired residual block.

    This keeps the design pure-convolutional: large-kernel depthwise temporal
    mixing, a small-kernel auxiliary branch, and a pointwise channel mixer."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        large_kernel_size: int,
        small_kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        large_padding = large_kernel_size // 2
        small_padding = small_kernel_size // 2
        self.pre_norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.large_kernel = nn.Conv1d(
            channels,
            channels,
            kernel_size=large_kernel_size,
            padding=large_padding,
            groups=channels,
        )
        self.small_kernel = nn.Conv1d(
            channels,
            channels,
            kernel_size=small_kernel_size,
            padding=small_padding,
            groups=channels,
        )
        self.post_mix = nn.Conv1d(channels, channels, kernel_size=1)
        self.ffn_norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.ffn = ConvFFN(
            channels=channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        x = self.large_kernel(x) + self.small_kernel(x)
        x = self.post_mix(x)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return residual + x


class ModernTCNBackbone(nn.Module):
    def __init__(self, input_dim: int, pred_len: int, config: ModelConfig):
        super().__init__()
        self.pred_len = pred_len
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, config.d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                ModernTCNBlock(
                    channels=config.d_model,
                    hidden_channels=config.d_ff,
                    large_kernel_size=config.tcn_large_kernel_size,
                    small_kernel_size=config.tcn_small_kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.tcn_num_blocks)
            ]
        )
        self.head = nn.Conv1d(config.d_model, 1, kernel_size=1)

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        x = sequence_features.transpose(1, 2)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        prediction = self.head(x).transpose(1, 2)
        return prediction[:, -self.pred_len :, 0]


class ModernTCNForecastModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        exog_dim: int,
        num_series: int,
        config: ModelConfig,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.exog_dim = exog_dim
        self.num_series = num_series
        self.config = config

        self.revin = RevIN(
            num_features=1,
            num_series=num_series,
            eps=config.revin_eps,
            affine=config.revin_affine,
            per_station_affine=config.revin_per_station_affine,
        )
        self.series_embedding = nn.Embedding(num_series, config.series_id_embedding_dim)
        self.series_id_mode = config.series_id_mode
        if self.series_id_mode not in {"repeat", "token"}:
            raise ValueError(f"Unsupported series_id_mode: {self.series_id_mode}")

        repeated_exog_dim = exog_dim + config.series_id_embedding_dim
        # target + exog + availability mask
        input_dim = 1 + repeated_exog_dim + 1
        if self.series_id_mode == "token":
            input_dim += config.series_id_embedding_dim

        self.backbone = ModernTCNBackbone(
            input_dim=input_dim,
            pred_len=pred_len,
            config=config,
        )

    def _repeat_series_embedding(
        self,
        history_exog: torch.Tensor,
        future_exog: torch.Tensor,
        series_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        series_embedding = self.series_embedding(series_id)
        history_static = series_embedding.unsqueeze(1).expand(-1, history_exog.shape[1], -1)
        future_static = series_embedding.unsqueeze(1).expand(-1, future_exog.shape[1], -1)
        return (
            torch.cat([history_exog, history_static], dim=-1),
            torch.cat([future_exog, future_static], dim=-1),
        )

    def _token_series_embedding(
        self,
        history_exog: torch.Tensor,
        future_exog: torch.Tensor,
        series_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        series_embedding = self.series_embedding(series_id)
        history_token = series_embedding.unsqueeze(1).expand(-1, history_exog.shape[1], -1)
        future_token = series_embedding.unsqueeze(1).expand(-1, future_exog.shape[1], -1)
        return (
            torch.cat([history_exog, history_token], dim=-1),
            torch.cat([future_exog, future_token], dim=-1),
        )

    def _build_sequence_features(
        self,
        history_target_norm: torch.Tensor,
        history_exog: torch.Tensor,
        future_exog: torch.Tensor,
    ) -> torch.Tensor:
        history_mask = torch.ones(history_target_norm.shape[0], self.seq_len, 1, device=history_target_norm.device)
        future_mask = torch.zeros(history_target_norm.shape[0], self.pred_len, 1, device=history_target_norm.device)
        future_target_placeholder = torch.zeros(history_target_norm.shape[0], self.pred_len, 1, device=history_target_norm.device)

        history_features = torch.cat([history_target_norm, history_exog, history_mask], dim=-1)
        future_features = torch.cat([future_target_placeholder, future_exog, future_mask], dim=-1)
        return torch.cat([history_features, future_features], dim=1)

    def forward(
        self,
        history_target: torch.Tensor,
        history_exog: torch.Tensor,
        future_exog: torch.Tensor,
        series_id: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | ForwardOutput:
        history_target = history_target.unsqueeze(-1)
        history_norm, stats = self.revin.normalize(history_target, series_id=series_id)

        if self.series_id_mode == "repeat":
            history_exog, future_exog = self._repeat_series_embedding(history_exog, future_exog, series_id)
        else:
            history_exog, future_exog = self._token_series_embedding(history_exog, future_exog, series_id)

        sequence_features = self._build_sequence_features(history_norm, history_exog, future_exog)
        prediction_norm = self.backbone(sequence_features)
        prediction = self.revin.denormalize(
            prediction_norm.unsqueeze(-1),
            stats,
            series_id=series_id,
        ).squeeze(-1)

        if return_aux:
            return ForwardOutput(
                prediction=prediction,
                prediction_norm=prediction_norm,
                revin_stats=stats,
            )
        return prediction

    def normalize_future_target(
        self,
        future_target: torch.Tensor,
        stats: RevINStats,
        series_id: torch.Tensor,
    ) -> torch.Tensor:
        future_target = future_target.unsqueeze(-1)
        normalized = self.revin.normalize_with_stats(future_target, stats, series_id=series_id)
        return normalized.squeeze(-1)
