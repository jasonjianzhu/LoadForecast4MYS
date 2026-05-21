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
    d_model: int = 192
    d_ff: int = 384
    e_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    patch_len: int = 12
    factor: int = 5
    activation: str = "gelu"
    series_id_embedding_dim: int = 2
    revin_affine: bool = False
    revin_eps: float = 1e-5
    huber_delta: float = 1.0
    loss_huber_weight: float = 0.5
    loss_mse_weight: float = 0.5
    peak_focus_quantile: float = 0.9
    peak_focus_weight: float = 1.0
    underprediction_weight: float = 0.25


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
