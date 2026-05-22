from __future__ import annotations

from collections import defaultdict
import json
import logging
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from loadforecast.config import ExperimentConfig
from loadforecast.data import ForecastWindowDataset
from loadforecast.metrics import compute_all_metrics, compute_peak_metrics
from loadforecast.models.common import ForwardOutput
from loadforecast.models.moderntcn import ModernTCNForecastModel
from loadforecast.models.patchtst import PatchTSTForecastModel
from loadforecast.models.timexer import LoadForecastModel
from loadforecast.plotting import generate_test_prediction_plots

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = float("inf")
        self.best_epoch = -1
        self.counter = 0

    def step(self, epoch: int, score: float) -> bool:
        if score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience


def make_dataloaders(dataset_map: dict[str, ForecastWindowDataset], cfg: ExperimentConfig) -> dict[str, DataLoader]:
    train_cfg = cfg.train
    train_dataset = dataset_map["train"]
    train_sampler = None
    shuffle = True
    if train_cfg.train_sampler == "station_balanced":
        station_counts: dict[int, int] = defaultdict(int)
        for spec in train_dataset.specs:
            station_counts[spec.station_id] += 1
        if station_counts:
            power = train_cfg.station_balance_power
            weights = [
                1.0 / float(station_counts[spec.station_id] ** power)
                for spec in train_dataset.specs
            ]
            train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False
    elif train_cfg.train_sampler != "shuffle":
        raise ValueError(f"Unsupported train_sampler: {train_cfg.train_sampler}")

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=train_cfg.num_workers,
        ),
        "val": DataLoader(
            dataset_map["val"],
            batch_size=train_cfg.eval_batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
        ),
        "test": DataLoader(
            dataset_map["test"],
            batch_size=train_cfg.eval_batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
        ),
    }


def build_model(dataset_map: dict[str, ForecastWindowDataset], cfg: ExperimentConfig) -> nn.Module:
    train_dataset = dataset_map["train"]
    if len(train_dataset.stores) == 0:
        raise ValueError("No stations were loaded.")
    exog_dim = train_dataset.stores[0].exog.shape[1]
    common_kwargs = {
        "seq_len": cfg.data.seq_len,
        "pred_len": cfg.data.pred_len,
        "exog_dim": exog_dim,
        "num_series": len(train_dataset.stores),
        "config": cfg.model,
    }
    if cfg.model.backbone == "timexer":
        return LoadForecastModel(**common_kwargs)
    if cfg.model.backbone == "moderntcn":
        return ModernTCNForecastModel(**common_kwargs)
    if cfg.model.backbone == "patchtst":
        return PatchTSTForecastModel(**common_kwargs)
    raise ValueError(f"Unsupported backbone: {cfg.model.backbone}")


def create_optimizer_and_scheduler(model: nn.Module, cfg: ExperimentConfig):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = None
    if cfg.train.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.train.scheduler_factor,
            patience=cfg.train.scheduler_patience,
            min_lr=cfg.train.min_learning_rate,
        )
    elif cfg.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.max_epochs,
            eta_min=cfg.train.min_learning_rate,
        )
    return optimizer, scheduler


def build_station_sample_weights(train_dataset: ForecastWindowDataset) -> torch.Tensor:
    """Return per-station weight vector so that stations with fewer training samples
    are not dominated by data-rich stations during gradient updates."""
    station_counts: dict[int, int] = defaultdict(int)
    for spec in train_dataset.specs:
        station_counts[spec.station_id] += 1
    if not station_counts:
        return torch.ones(len(train_dataset.stores))
    weights = torch.ones(len(train_dataset.stores), dtype=torch.float32)
    for station_id, count in station_counts.items():
        weights[station_id] = 1.0 / float(np.log1p(count))
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights


def build_peak_focus_weights(
    target: torch.Tensor,
    quantile: float | None,
    extra_weight: float,
) -> torch.Tensor:
    if quantile is None or extra_weight <= 0:
        return torch.ones_like(target)

    quantile = float(np.clip(quantile, 0.0, 1.0))
    num_steps = target.shape[1]
    kth_index = min(max(int(np.ceil(quantile * num_steps)), 1), num_steps)
    threshold = target.kthvalue(kth_index, dim=1, keepdim=True).values
    peak_mask = target >= threshold
    return 1.0 + extra_weight * peak_mask.float()


def huber_loss_pointwise(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    diff = prediction - target
    abs_diff = diff.abs()
    quadratic = torch.minimum(abs_diff, torch.full_like(abs_diff, delta))
    linear = abs_diff - quadratic
    return 0.5 * quadratic.pow(2) + delta * linear


def compute_pointwise_loss(
    model: LoadForecastModel,
    output: ForwardOutput,
    future_target: torch.Tensor,
    series_id: torch.Tensor,
    cfg: ExperimentConfig,
) -> torch.Tensor:
    target_norm = model.normalize_future_target(future_target, output.revin_stats, series_id=series_id)
    point_weights = build_peak_focus_weights(
        target=future_target,
        quantile=cfg.model.pointwise_peak_focus_quantile,
        extra_weight=cfg.model.pointwise_peak_focus_weight,
    )
    if cfg.model.base_loss == "pinball":
        diff = target_norm - output.prediction_norm
        tau = cfg.model.pinball_tau
        pointwise = torch.max(tau * diff, (tau - 1.0) * diff)
        return pointwise * point_weights
    if cfg.model.base_loss == "hybrid":
        huber = huber_loss_pointwise(
            prediction=output.prediction_norm,
            target=target_norm,
            delta=cfg.model.huber_delta,
        )
        mse = (output.prediction_norm - target_norm).pow(2)
        underprediction = torch.relu(target_norm - output.prediction_norm)
        pointwise = (
            cfg.model.loss_huber_weight * huber
            + cfg.model.loss_mse_weight * mse
            + cfg.model.underprediction_weight * underprediction
        )
        return pointwise * point_weights
    raise ValueError(f"Unsupported base_loss: {cfg.model.base_loss}")


def compute_peak_losses(
    model: LoadForecastModel,
    output: ForwardOutput,
    future_target: torch.Tensor,
    series_id: torch.Tensor,
    cfg: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_norm = model.normalize_future_target(future_target, output.revin_stats, series_id=series_id)
    top_k = max(1, min(cfg.model.peak_top_k, future_target.shape[1]))
    peak_indices = future_target.topk(top_k, dim=1).indices

    pred_topk = output.prediction_norm.gather(dim=1, index=peak_indices)
    true_topk = target_norm.gather(dim=1, index=peak_indices)
    peak_loss = huber_loss_pointwise(
        prediction=pred_topk,
        target=true_topk,
        delta=cfg.model.huber_delta,
    ).mean(dim=1)
    underprediction_topk = torch.relu(true_topk - pred_topk).mean(dim=1)

    pred_daily_max = output.prediction_norm.max(dim=1).values
    true_daily_max = target_norm.max(dim=1).values
    daily_max_loss = (pred_daily_max - true_daily_max).pow(2)
    return peak_loss, underprediction_topk, daily_max_loss


def compute_scenario_weights(
    scenario: list[str],
    device: torch.device,
    cfg: ExperimentConfig,
) -> torch.Tensor:
    if cfg.model.scenario_loss_weights:
        default_weight = 1.0
        weights = [
            float(cfg.model.scenario_loss_weights.get(s, default_weight))
            for s in scenario
        ]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    holiday_mask = torch.tensor(
        ["holiday" in s for s in scenario],
        dtype=torch.float32,
        device=device,
    )
    return 1.0 - (1.0 - cfg.model.holiday_loss_weight) * holiday_mask


def compute_training_objective(
    model: LoadForecastModel,
    output: ForwardOutput,
    future_target: torch.Tensor,
    series_id: torch.Tensor,
    scenario: list[str],
    station_weights: torch.Tensor,
    cfg: ExperimentConfig,
) -> torch.Tensor:
    pointwise = compute_pointwise_loss(
        model=model, output=output, future_target=future_target,
        series_id=series_id, cfg=cfg,
    )
    peak_loss, underprediction_topk, daily_max_loss = compute_peak_losses(
        model=model,
        output=output,
        future_target=future_target,
        series_id=series_id,
        cfg=cfg,
    )
    sample_loss = pointwise.mean(dim=1)
    sample_loss = (
        sample_loss
        + cfg.model.peak_loss_weight * peak_loss
        + cfg.model.underprediction_topk_weight * underprediction_topk
        + cfg.model.daily_max_loss_weight * daily_max_loss
    )

    station_weight = station_weights.to(series_id.device)[series_id]
    station_weight = station_weight.clamp(
        min=cfg.model.station_weight_clip_min,
        max=cfg.model.station_weight_clip_max,
    )
    sample_loss = sample_loss * station_weight

    scenario_weight = compute_scenario_weights(
        scenario=scenario,
        device=sample_loss.device,
        cfg=cfg,
    )
    sample_loss = sample_loss * scenario_weight

    return sample_loss.sum() / sample_loss.numel()


def train_one_epoch(
    model: LoadForecastModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: ExperimentConfig,
    station_weights: torch.Tensor,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for step_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)

        history_target = batch["history_target"].to(device)
        future_target = batch["future_target"].to(device)
        history_exog = batch["history_exog"].to(device)
        future_exog = batch["future_exog"].to(device)
        series_id = batch["series_id"].to(device)

        output = model(
            history_target=history_target,
            history_exog=history_exog,
            future_exog=future_exog,
            series_id=series_id,
            return_aux=True,
        )
        assert isinstance(output, ForwardOutput)
        loss = compute_training_objective(
            model=model, output=output, future_target=future_target,
            series_id=series_id, scenario=batch["scenario"],
            station_weights=station_weights, cfg=cfg,
        )
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        batch_size = history_target.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        if cfg.train.log_interval_steps > 0 and step_idx % cfg.train.log_interval_steps == 0:
            LOGGER.info(
                "train_step=%s/%s batch_loss=%.6f running_loss=%.6f",
                step_idx,
                len(loader),
                float(loss.item()),
                total_loss / max(total_count, 1),
            )

    return total_loss / max(total_count, 1)


def evaluate(
    model: LoadForecastModel,
    loader: DataLoader,
    device: torch.device,
    cfg: ExperimentConfig,
    station_weights: torch.Tensor,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            history_target = batch["history_target"].to(device)
            future_target = batch["future_target"].to(device)
            history_exog = batch["history_exog"].to(device)
            future_exog = batch["future_exog"].to(device)
            series_id = batch["series_id"].to(device)

            output = model(
                history_target=history_target,
                history_exog=history_exog,
                future_exog=future_exog,
                series_id=series_id,
                return_aux=True,
            )
            assert isinstance(output, ForwardOutput)
            loss = compute_training_objective(
                model=model, output=output, future_target=future_target,
                series_id=series_id, scenario=batch["scenario"],
                station_weights=station_weights, cfg=cfg,
            )
            batch_size = history_target.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            prediction_np = output.prediction.detach().cpu().numpy()
            target_np = future_target.detach().cpu().numpy()
            for index in range(batch_size):
                row = {
                    "series_name": batch["series_name"][index],
                    "forecast_start": batch["forecast_start"][index],
                    "scenario": batch["scenario"][index],
                    "split": batch["split"][index],
                    "y_true": target_np[index].tolist(),
                    "y_pred": prediction_np[index].tolist(),
                }
                prediction_rows.append(row)

    prediction_frame = pd.DataFrame(prediction_rows)
    metrics = summarize_metrics(prediction_frame)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics, prediction_frame


def summarize_metrics(prediction_frame: pd.DataFrame) -> dict[str, float]:
    if prediction_frame.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "nmae": float("nan"), "nrmse": float("nan"), "wape": float("nan"), "mape": float("nan")}

    y_true = np.concatenate(prediction_frame["y_true"].apply(np.asarray).to_list())
    y_pred = np.concatenate(prediction_frame["y_pred"].apply(np.asarray).to_list())
    return dict(compute_all_metrics(y_true, y_pred))


def summarize_by_group(prediction_frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, group in prediction_frame.groupby(group_column):
        metrics = summarize_metrics(group)
        metrics[group_column] = group_value
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_peak_by_series(prediction_frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-series peak diagnostics averaged over forecast days."""
    rows: list[dict[str, Any]] = []
    for series_name, group in prediction_frame.groupby("series_name"):
        peak_ratios: list[float] = []
        peak_errs: list[float] = []
        nonpeak_biases: list[float] = []
        for _, row in group.iterrows():
            y_true = np.asarray(row["y_true"], dtype=np.float64)
            y_pred = np.asarray(row["y_pred"], dtype=np.float64)
            peak = compute_peak_metrics(y_true, y_pred)
            peak_ratios.append(peak["peak_ratio"])
            peak_errs.append(peak["peak_err"])
            if not np.isnan(peak["nonpeak_bias"]):
                nonpeak_biases.append(peak["nonpeak_bias"])
        rows.append(
            {
                "series_name": series_name,
                "peak_ratio_mean": float(np.mean(peak_ratios)),
                "peak_err_mean": float(np.mean(peak_errs)),
                "nonpeak_bias_mean": float(np.mean(nonpeak_biases)) if nonpeak_biases else float("nan"),
            }
        )
    return rows


def save_prediction_frame(prediction_frame: pd.DataFrame, output_path: Path) -> None:
    exploded_rows: list[dict[str, Any]] = []
    for _, row in prediction_frame.iterrows():
        forecast_start = pd.Timestamp(row["forecast_start"])
        timestamps = pd.date_range(forecast_start, periods=len(row["y_true"]), freq="5min")
        for timestamp, y_true, y_pred in zip(timestamps, row["y_true"], row["y_pred"]):
            exploded_rows.append(
                {
                    "series_name": row["series_name"],
                    "scenario": row["scenario"],
                    "split": row["split"],
                    "timestamp": timestamp.isoformat(),
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                }
            )
    pd.DataFrame(exploded_rows).to_csv(output_path, index=False)


def build_epoch_checkpoint_name(base_name: str, epoch: int) -> str:
    path = Path(base_name)
    suffix = path.suffix or ".pt"
    stem = path.stem if path.suffix else path.name
    return f"{stem}_epoch{epoch}{suffix}"


def remove_legacy_training_outputs(output_dir: Path, checkpoint_base_name: str) -> None:
    legacy_files = (
        "config_snapshot.json",
        "experiment_config_snapshot.json",
    )
    for filename in legacy_files:
        path = output_dir / filename
        if path.exists():
            path.unlink()

    checkpoint_base = Path(checkpoint_base_name)
    pattern = re.compile(rf"^{re.escape(checkpoint_base.stem)}_epoch\d+{re.escape(checkpoint_base.suffix or '.pt')}$")
    for path in output_dir.glob("*"):
        if path.is_file() and pattern.match(path.name):
            path.unlink()


def train_and_evaluate(dataset_map: dict[str, ForecastWindowDataset], cfg: ExperimentConfig) -> Path:
    set_seed(cfg.train.seed)
    output_dir = Path(cfg.output.root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_training_outputs(output_dir, cfg.output.save_checkpoint_name)
    (output_dir / "experiment_config_snapshot.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    device = resolve_device(cfg.train.device)
    dataloaders = make_dataloaders(dataset_map, cfg)
    station_weights = build_station_sample_weights(dataset_map["train"])
    model = build_model(dataset_map, cfg).to(device)
    optimizer, scheduler = create_optimizer_and_scheduler(model, cfg)
    early_stopper = EarlyStopper(patience=cfg.train.early_stopping_patience)
    checkpoint_path = output_dir / ".checkpoint_tmp.pt"

    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, cfg.train.max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            station_weights=station_weights,
            grad_clip_norm=cfg.train.grad_clip_norm,
        )
        val_metrics, _ = evaluate(
            model=model, loader=dataloaders["val"], device=device,
            cfg=cfg, station_weights=station_weights,
        )
        score = float(val_metrics[cfg.train.early_stopping_metric])

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        LOGGER.info(
            "epoch=%s train_loss=%.6f val_loss=%.6f val_mae=%.6f val_nrmse=%.6f lr=%.6g",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["mae"],
            val_metrics["nrmse"],
            optimizer.param_groups[0]["lr"],
        )

        improved = early_stopper.step(epoch, score)
        if improved:
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, checkpoint_path)
            LOGGER.info("checkpoint_saved epoch=%s", epoch)

        if scheduler is not None:
            if cfg.train.scheduler == "plateau":
                scheduler.step(score)
            else:
                scheduler.step()

        if early_stopper.should_stop:
            LOGGER.info(
                "early_stopping epoch=%s best_epoch=%s best_%s=%.6f",
                epoch,
                early_stopper.best_epoch,
                cfg.train.early_stopping_metric,
                early_stopper.best_score,
            )
            break

    pd.DataFrame(history_rows).to_csv(output_dir / "epoch_training_history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    summary: dict[str, Any] = {"best_epoch": int(checkpoint["epoch"])}
    for split in ("val", "test"):
        metrics, frame = evaluate(
            model=model, loader=dataloaders[split], device=device,
            cfg=cfg, station_weights=station_weights,
        )
        summary[split] = metrics
        summary[f"{split}_by_scenario"] = summarize_by_group(frame, "scenario").to_dict(orient="records")
        summary[f"{split}_by_series"] = summarize_by_group(frame, "series_name").to_dict(orient="records")
        summary[f"{split}_peak_by_series"] = summarize_peak_by_series(frame)
        if cfg.output.save_predictions:
            filename = "validation_predictions.csv" if split == "val" else f"{split}_predictions.csv"
            save_prediction_frame(frame, output_dir / filename)

    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if cfg.output.save_predictions:
        generate_test_prediction_plots(
            output_dir / "test_predictions.csv",
            output_dir / "plots",
        )
    final_checkpoint_name = build_epoch_checkpoint_name(cfg.output.save_checkpoint_name, int(checkpoint["epoch"]))
    final_checkpoint_path = output_dir / final_checkpoint_name
    checkpoint_path.replace(final_checkpoint_path)
    return output_dir
