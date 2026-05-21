from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from loadforecast.config import ModelConfig
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class FullAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, embed_dim = queries.shape
        scale = 1.0 / math.sqrt(embed_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", attention, values)


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attention_dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        head_dim = d_model // n_heads
        self.query_projection = nn.Linear(d_model, head_dim * n_heads)
        self.key_projection = nn.Linear(d_model, head_dim * n_heads)
        self.value_projection = nn.Linear(d_model, head_dim * n_heads)
        self.out_projection = nn.Linear(head_dim * n_heads, d_model)
        self.attention = FullAttention(attention_dropout=attention_dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        batch_size, query_len, _ = queries.shape
        _, key_len, _ = keys.shape

        queries = self.query_projection(queries).view(batch_size, query_len, self.n_heads, -1)
        keys = self.key_projection(keys).view(batch_size, key_len, self.n_heads, -1)
        values = self.value_projection(values).view(batch_size, key_len, self.n_heads, -1)

        out = self.attention(queries, keys, values)
        out = out.reshape(batch_size, query_len, -1)
        return self.out_projection(out)


class FlattenHead(nn.Module):
    def __init__(self, n_vars: int, hidden_features: int, target_window: int, dropout: float):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(hidden_features, target_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class EndogenousEmbedding(nn.Module):
    def __init__(self, n_vars: int, d_model: int, patch_len: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.global_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        n_vars = x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.value_embedding(x) + self.position_embedding(x)
        x = x.reshape(-1, n_vars, x.shape[-2], x.shape[-1])
        glb = self.global_token.repeat(x.shape[0], 1, 1, 1)
        x = torch.cat([x, glb], dim=2)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return self.dropout(x), n_vars


class ExogenousPatchEmbedding(nn.Module):
    def __init__(self, total_len: int, exog_dim: int, d_model: int, patch_len: int, dropout: float):
        super().__init__()
        if total_len % patch_len != 0:
            raise ValueError("total exogenous length must be divisible by patch_len")
        self.patch_len = patch_len
        self.exog_dim = exog_dim
        self.patch_num = total_len // patch_len
        self.value_embedding = nn.Linear(patch_len * exog_dim, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, history_exog: torch.Tensor, future_exog: torch.Tensor) -> torch.Tensor:
        x = torch.cat([history_exog, future_exog], dim=1)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.patch_num, self.patch_len * self.exog_dim)
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.self_attention = AttentionLayer(d_model=d_model, n_heads=n_heads, attention_dropout=dropout)
        self.cross_attention = AttentionLayer(d_model=d_model, n_heads=n_heads, attention_dropout=dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        batch_size, _, model_dim = cross.shape
        x = x + self.dropout(self.self_attention(x, x, x))
        x = self.norm1(x)

        x_global_original = x[:, -1, :].unsqueeze(1)
        x_global = x_global_original.reshape(batch_size, -1, model_dim)
        x_global_attn = self.dropout(self.cross_attention(x_global, cross, cross))
        x_global_attn = x_global_attn.reshape(x_global_attn.shape[0] * x_global_attn.shape[1], x_global_attn.shape[2]).unsqueeze(1)
        x_global = self.norm2(x_global_original + x_global_attn)

        y = x = torch.cat([x[:, :-1, :], x_global], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Encoder(nn.Module):
    def __init__(self, layers: list[EncoderLayer], d_model: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cross)
        return self.norm(x)


class TimeXerBackbone(nn.Module):
    """Local TimeXer variant with future-known exogenous inputs."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        exog_dim: int,
        config: ModelConfig,
    ):
        super().__init__()
        if seq_len % config.patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len")
        if (seq_len + pred_len) % config.patch_len != 0:
            raise ValueError("seq_len + pred_len must be divisible by patch_len")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.exog_dim = exog_dim
        self.patch_len = config.patch_len
        self.patch_num = seq_len // config.patch_len
        self.endogenous_embedding = EndogenousEmbedding(n_vars=1, d_model=config.d_model, patch_len=config.patch_len, dropout=config.dropout)
        self.exogenous_embedding = ExogenousPatchEmbedding(
            total_len=seq_len + pred_len,
            exog_dim=exog_dim,
            d_model=config.d_model,
            patch_len=config.patch_len,
            dropout=config.dropout,
        )
        self.encoder = Encoder(
            layers=[
                EncoderLayer(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            d_model=config.d_model,
        )
        self.head = FlattenHead(
            n_vars=1,
            hidden_features=config.d_model * (self.patch_num + 1),
            target_window=pred_len,
            dropout=config.dropout,
        )

    def forward(
        self,
        history_target: torch.Tensor,
        history_exog: torch.Tensor,
        future_exog: torch.Tensor,
    ) -> torch.Tensor:
        endo_embed, n_vars = self.endogenous_embedding(history_target.unsqueeze(1))
        exo_embed = self.exogenous_embedding(history_exog, future_exog)
        encoded = self.encoder(endo_embed, exo_embed)
        encoded = encoded.reshape(-1, n_vars, encoded.shape[-2], encoded.shape[-1])
        encoded = encoded.permute(0, 1, 3, 2)
        prediction = self.head(encoded)
        return prediction.permute(0, 2, 1).squeeze(-1)


@dataclass
class ForwardOutput:
    prediction: torch.Tensor
    prediction_norm: torch.Tensor
    revin_stats: RevINStats


class LoadForecastModel(nn.Module):
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

        self.revin = RevIN(num_features=1, eps=config.revin_eps, affine=config.revin_affine)
        self.series_embedding = nn.Embedding(num_series, config.series_id_embedding_dim)
        self.backbone = TimeXerBackbone(
            seq_len=seq_len,
            pred_len=pred_len,
            exog_dim=exog_dim + config.series_id_embedding_dim,
            config=config,
        )

    def _append_series_embedding(
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

    def forward(
        self,
        history_target: torch.Tensor,
        history_exog: torch.Tensor,
        future_exog: torch.Tensor,
        series_id: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | ForwardOutput:
        history_target = history_target.unsqueeze(-1)
        history_norm, stats = self.revin.normalize(history_target)
        history_exog, future_exog = self._append_series_embedding(history_exog, future_exog, series_id)
        prediction_norm = self.backbone(
            history_target=history_norm.squeeze(-1),
            history_exog=history_exog,
            future_exog=future_exog,
        )
        prediction = self.revin.denormalize(prediction_norm.unsqueeze(-1), stats).squeeze(-1)

        if return_aux:
            return ForwardOutput(prediction=prediction, prediction_norm=prediction_norm, revin_stats=stats)
        return prediction

    def normalize_future_target(self, future_target: torch.Tensor, stats: RevINStats) -> torch.Tensor:
        future_target = future_target.unsqueeze(-1)
        normalized = self.revin.normalize_with_stats(future_target, stats)
        return normalized.squeeze(-1)
