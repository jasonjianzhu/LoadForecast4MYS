from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from loadforecast.config import ExperimentConfig
from loadforecast.data import ForecastWindowDataset
from loadforecast.metrics import compute_all_metrics
from loadforecast.models.timexer import ForwardOutput, LoadForecastModel
from loadforecast.plotting import generate_test_prediction_plots


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
    return {
        "train": DataLoader(
            dataset_map["train"],
            batch_size=train_cfg.batch_size,
            shuffle=True,
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


def build_model(dataset_map: dict[str, ForecastWindowDataset], cfg: ExperimentConfig) -> LoadForecastModel:
    train_dataset = dataset_map["train"]
    if len(train_dataset.stores) == 0:
        raise ValueError("No stations were loaded.")
    exog_dim = train_dataset.stores[0].exog.shape[1]
    return LoadForecastModel(
        seq_len=cfg.data.seq_len,
        pred_len=cfg.data.pred_len,
        exog_dim=exog_dim,
        num_series=len(train_dataset.stores),
        config=cfg.model,
    )


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


def train_one_epoch(
    model: LoadForecastModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    delta: float,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    criterion = nn.HuberLoss(delta=delta)

    for batch in loader:
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
        target_norm = model.normalize_future_target(future_target, output.revin_stats)
        loss = criterion(output.prediction_norm, target_norm)
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        batch_size = history_target.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate(
    model: LoadForecastModel,
    loader: DataLoader,
    device: torch.device,
    delta: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    criterion = nn.HuberLoss(delta=delta)
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
            target_norm = model.normalize_future_target(future_target, output.revin_stats)
            loss = criterion(output.prediction_norm, target_norm)
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
            delta=cfg.model.huber_delta,
            grad_clip_norm=cfg.train.grad_clip_norm,
        )
        val_metrics, _ = evaluate(model=model, loader=dataloaders["val"], device=device, delta=cfg.model.huber_delta)
        score = float(val_metrics[cfg.train.early_stopping_metric])

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_nrmse={val_metrics['nrmse']:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}",
            flush=True,
        )

        improved = early_stopper.step(epoch, score)
        if improved:
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, checkpoint_path)
            print(f"checkpoint_saved epoch={epoch}", flush=True)

        if scheduler is not None:
            if cfg.train.scheduler == "plateau":
                scheduler.step(score)
            else:
                scheduler.step()

        if early_stopper.should_stop:
            print(
                f"early_stopping epoch={epoch} "
                f"best_epoch={early_stopper.best_epoch} "
                f"best_{cfg.train.early_stopping_metric}={early_stopper.best_score:.6f}",
                flush=True,
            )
            break

    pd.DataFrame(history_rows).to_csv(output_dir / "epoch_training_history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    summary: dict[str, Any] = {"best_epoch": int(checkpoint["epoch"])}
    for split in ("val", "test"):
        metrics, frame = evaluate(model=model, loader=dataloaders[split], device=device, delta=cfg.model.huber_delta)
        summary[split] = metrics
        summary[f"{split}_by_scenario"] = summarize_by_group(frame, "scenario").to_dict(orient="records")
        summary[f"{split}_by_series"] = summarize_by_group(frame, "series_name").to_dict(orient="records")
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
