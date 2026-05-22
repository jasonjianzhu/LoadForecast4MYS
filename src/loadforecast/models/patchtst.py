from __future__ import annotations

import math

import torch
from torch import nn

from loadforecast.config import ModelConfig
from loadforecast.models.common import ForwardOutput
from loadforecast.models.revin import RevIN, RevINStats


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]


class PatchTSTBackbone(nn.Module):
    """PatchTST-style backbone adapted to future-known exogenous features.

    The official PatchTST emphasizes patching over long lookback windows and
    transformer processing over patch tokens. Here we keep that spirit while
    feeding a concatenated sequence of:
    - normalized history target
    - known exogenous features
    - an availability mask
    """

    def __init__(
        self,
        total_len: int,
        pred_len: int,
        input_dim: int,
        include_series_token: bool,
        config: ModelConfig,
    ):
        super().__init__()
        if config.patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if config.patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if total_len < config.patch_len:
            raise ValueError("total_len must be at least patch_len")
        if (total_len - config.patch_len) % config.patch_stride != 0:
            raise ValueError("total_len must align with patch_stride")

        self.total_len = total_len
        self.pred_len = pred_len
        self.patch_len = config.patch_len
        self.patch_stride = config.patch_stride
        self.include_series_token = include_series_token
        self.patch_num = (total_len - config.patch_len) // config.patch_stride + 1

        self.patch_embedding = nn.Linear(config.patch_len * input_dim, config.d_model, bias=False)
        self.position_embedding = PositionalEmbedding(config.d_model)
        if include_series_token:
            self.series_token_projection = nn.Linear(config.series_id_embedding_dim, config.d_model)
            self.series_token_norm = nn.LayerNorm(config.d_model)
        else:
            self.series_token_projection = None
            self.series_token_norm = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.e_layers)
        token_count = self.patch_num + (1 if include_series_token else 0)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(config.dropout),
            nn.Linear(token_count * config.d_model, pred_len),
        )

    def _patchify(self, sequence_features: torch.Tensor) -> torch.Tensor:
        patches = sequence_features.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.contiguous().reshape(
            sequence_features.shape[0],
            self.patch_num,
            self.patch_len * sequence_features.shape[-1],
        )
        tokens = self.patch_embedding(patches)
        tokens = tokens + self.position_embedding(tokens.shape[1])
        return tokens

    def forward(
        self,
        sequence_features: torch.Tensor,
        series_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self._patchify(sequence_features)
        if self.include_series_token:
            if series_embedding is None:
                raise ValueError("series_embedding is required when include_series_token=True")
            series_token = self.series_token_projection(series_embedding)
            series_token = self.series_token_norm(series_token).unsqueeze(1)
            tokens = torch.cat([series_token, tokens], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded)


class PatchTSTForecastModel(nn.Module):
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

        repeated_exog_dim = exog_dim + (config.series_id_embedding_dim if self.series_id_mode == "repeat" else 0)
        input_dim = 1 + repeated_exog_dim + 1
        self.backbone = PatchTSTBackbone(
            total_len=seq_len + pred_len,
            pred_len=pred_len,
            input_dim=input_dim,
            include_series_token=self.series_id_mode == "token",
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
        series_embedding = self.series_embedding(series_id)

        if self.series_id_mode == "repeat":
            history_exog, future_exog = self._repeat_series_embedding(history_exog, future_exog, series_id)
            series_token = None
        else:
            series_token = series_embedding

        sequence_features = self._build_sequence_features(history_norm, history_exog, future_exog)
        prediction_norm = self.backbone(sequence_features, series_embedding=series_token)
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
