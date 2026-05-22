from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    data_dir: str = "data"
    time_column: str = "Time"
    target_column: str = "load(kW)"
    target_clip_min: float | None = None
    freq: str = "5min"
    seq_len: int = 2016
    pred_len: int = 288
    train_stride: int = 6
    val_days: int = 5
    test_days: int = 7
    holiday_country: str = "MY"
    default_holiday_subdiv: str = "SGR"
    station_holiday_subdiv_map: dict[str, str] = field(default_factory=dict)
    holiday_strict: bool = True
    pre_holiday_days: int = 1
    post_holiday_days: int = 1


@dataclass
class ModelConfig:
    backbone: str = "timexer"
    d_model: int = 192
    d_ff: int = 384
    e_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    patch_len: int = 12
    patch_stride: int = 6
    tcn_num_blocks: int = 6
    tcn_large_kernel_size: int = 25
    tcn_small_kernel_size: int = 5
    factor: int = 5
    activation: str = "gelu"
    series_id_embedding_dim: int = 4
    series_id_mode: str = "repeat"
    revin_affine: bool = False
    revin_per_station_affine: bool = False
    revin_eps: float = 1e-5
    base_loss: str = "pinball"
    huber_delta: float = 1.0
    loss_huber_weight: float = 0.0
    loss_mse_weight: float = 0.0
    pointwise_peak_focus_quantile: float | None = None
    pointwise_peak_focus_weight: float = 0.0
    underprediction_weight: float = 0.0
    pinball_tau: float = 0.65
    peak_top_k: int = 24
    peak_loss_weight: float = 0.3
    underprediction_topk_weight: float = 0.0
    daily_max_loss_weight: float = 0.1
    holiday_loss_weight: float = 1.0
    scenario_loss_weights: dict[str, float] = field(default_factory=dict)
    station_weight_clip_min: float = 0.9
    station_weight_clip_max: float = 1.15


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 16
    eval_batch_size: int = 32
    max_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 8
    early_stopping_metric: str = "nrmse"
    scheduler: str = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    min_learning_rate: float = 1e-5
    num_workers: int = 0
    train_sampler: str = "shuffle"
    station_balance_power: float = 0.5
    log_interval_steps: int = 0
    device: str = "auto"


@dataclass
class OutputConfig:
    root_dir: str = "outputs/timexer_final_affine_off"
    save_predictions: bool = True
    save_checkpoint_name: str = "timexer_revin_affine_off_best.pt"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            data=DataConfig(**payload.get("data", {})),
            model=ModelConfig(**payload.get("model", {})),
            train=TrainConfig(**payload.get("train", {})),
            output=OutputConfig(**payload.get("output", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
