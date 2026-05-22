from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from loadforecast.calendar import HolidaySets, build_calendar_features, classify_day_type, load_holiday_sets
from loadforecast.config import DataConfig


FEATURE_COLUMNS = [
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "is_federal_holiday",
    "is_state_holiday",
    "is_pre_holiday",
    "is_post_holiday",
]


@dataclass(frozen=True)
class WindowSpec:
    station_id: int
    forecast_start_idx: int
    split: str
    scenario: str


@dataclass
class StationStore:
    station_id: int
    station_name: str
    source_path: Path
    holiday_subdiv: str
    timestamps: pd.DatetimeIndex
    target: np.ndarray
    observed: np.ndarray
    exog: np.ndarray
    holiday_sets: HolidaySets
    full_days: list[pd.Timestamp]


class ForecastWindowDataset(Dataset):
    def __init__(self, stores: list[StationStore], specs: list[WindowSpec], seq_len: int, pred_len: int):
        self.stores = stores
        self.specs = specs
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> dict[str, object]:
        spec = self.specs[index]
        store = self.stores[spec.station_id]
        start = spec.forecast_start_idx
        hist_slice = slice(start - self.seq_len, start)
        fut_slice = slice(start, start + self.pred_len)

        history_target = torch.from_numpy(store.target[hist_slice]).float()
        future_target = torch.from_numpy(store.target[fut_slice]).float()
        history_exog = torch.from_numpy(store.exog[hist_slice]).float()
        future_exog = torch.from_numpy(store.exog[fut_slice]).float()

        forecast_start = store.timestamps[start]
        return {
            "history_target": history_target,
            "future_target": future_target,
            "history_exog": history_exog,
            "future_exog": future_exog,
            "series_id": torch.tensor(store.station_id, dtype=torch.long),
            "series_name": store.station_name,
            "forecast_start": forecast_start.isoformat(),
            "scenario": spec.scenario,
            "split": spec.split,
        }


def discover_data_files(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).glob("*.csv"))


def resolve_station_holiday_subdiv(station_name: str, cfg: DataConfig) -> str:
    return cfg.station_holiday_subdiv_map.get(station_name, cfg.default_holiday_subdiv)


def _clip_target_values(target: np.ndarray, clip_min: float | None) -> np.ndarray:
    if clip_min is None:
        return target
    return np.where(np.isnan(target), target, np.maximum(target, clip_min))


def load_station_store(path: Path, station_id: int, cfg: DataConfig) -> StationStore:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().replace("\ufeff", "") for column in frame.columns]
    station_name = path.stem
    holiday_subdiv = resolve_station_holiday_subdiv(station_name, cfg)

    frame[cfg.time_column] = pd.to_datetime(frame[cfg.time_column], errors="coerce")
    frame[cfg.target_column] = pd.to_numeric(frame[cfg.target_column], errors="coerce")
    frame = frame.dropna(subset=[cfg.time_column]).copy()
    frame = frame.sort_values(cfg.time_column)

    frame = frame[[cfg.time_column, cfg.target_column]]
    frame = frame.groupby(cfg.time_column, as_index=False)[cfg.target_column].mean()
    frame = frame.set_index(cfg.time_column)

    full_index = pd.date_range(
        start=frame.index.min().floor(cfg.freq),
        end=frame.index.max().floor(cfg.freq),
        freq=cfg.freq,
    )
    frame = frame.reindex(full_index)

    feature_frame = build_calendar_features(
        full_index,
        country=cfg.holiday_country,
        subdiv=holiday_subdiv,
        strict=cfg.holiday_strict,
        pre_holiday_days=cfg.pre_holiday_days,
        post_holiday_days=cfg.post_holiday_days,
    )
    holiday_sets = load_holiday_sets(
        years=sorted({timestamp.year for timestamp in full_index}),
        country=cfg.holiday_country,
        subdiv=holiday_subdiv,
        strict=cfg.holiday_strict,
    )

    target = frame[cfg.target_column].to_numpy(dtype=np.float32)
    target = _clip_target_values(target, cfg.target_clip_min)
    observed = ~np.isnan(target)
    day_counts = pd.Series(observed.astype(np.int16), index=full_index).groupby(full_index.normalize()).sum()
    expected_per_day = int(pd.Timedelta(days=1) / pd.Timedelta(cfg.freq))
    full_days = [day for day, count in day_counts.items() if count == expected_per_day]

    return StationStore(
        station_id=station_id,
        station_name=station_name,
        source_path=path,
        holiday_subdiv=holiday_subdiv,
        timestamps=full_index,
        target=target,
        observed=observed,
        exog=feature_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        holiday_sets=holiday_sets,
        full_days=full_days,
    )


def load_all_stations(cfg: DataConfig) -> list[StationStore]:
    paths = discover_data_files(cfg.data_dir)
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {cfg.data_dir!r}")
    return [load_station_store(path, station_id=index, cfg=cfg) for index, path in enumerate(paths)]


def _is_window_observed(mask: np.ndarray, start: int, length: int) -> bool:
    segment = mask[start : start + length]
    return bool(segment.size == length and segment.all())


def _scenario_for_day(day: pd.Timestamp, holiday_sets: HolidaySets) -> str:
    previous_day = pd.Timestamp(day).normalize() - pd.Timedelta(days=1)
    return f"{classify_day_type(previous_day, holiday_sets)}->{classify_day_type(day, holiday_sets)}"


def build_train_specs(store: StationStore, cfg: DataConfig, train_cutoff: pd.Timestamp) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    cutoff_idx = int(store.timestamps.get_indexer([train_cutoff], method="nearest")[0])
    for forecast_idx in range(cfg.seq_len, cutoff_idx - cfg.pred_len + 1, cfg.train_stride):
        hist_start = forecast_idx - cfg.seq_len
        if not _is_window_observed(store.observed, hist_start, cfg.seq_len):
            continue
        if not _is_window_observed(store.observed, forecast_idx, cfg.pred_len):
            continue
        forecast_day = store.timestamps[forecast_idx].normalize()
        specs.append(
            WindowSpec(
                station_id=store.station_id,
                forecast_start_idx=forecast_idx,
                split="train",
                scenario=_scenario_for_day(forecast_day, store.holiday_sets),
            )
        )
    return specs


def build_daily_eval_specs(store: StationStore, days: Iterable[pd.Timestamp], cfg: DataConfig, split: str) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    full_day_set = {pd.Timestamp(day).normalize() for day in store.full_days}
    for day in days:
        day = pd.Timestamp(day).normalize()
        required_history = pd.date_range(day - pd.Timedelta(days=7), periods=7, freq="D")
        required = list(required_history) + [day]
        if not all(candidate in full_day_set for candidate in required):
            continue

        forecast_idx = int(store.timestamps.get_indexer([day], method="nearest")[0])
        if store.timestamps[forecast_idx] != day:
            continue
        specs.append(
            WindowSpec(
                station_id=store.station_id,
                forecast_start_idx=forecast_idx,
                split=split,
                scenario=_scenario_for_day(day, store.holiday_sets),
            )
        )
    return specs


def build_datasets(cfg: DataConfig) -> dict[str, ForecastWindowDataset]:
    stores = load_all_stations(cfg)
    train_specs: list[WindowSpec] = []
    val_specs: list[WindowSpec] = []
    test_specs: list[WindowSpec] = []

    for store in stores:
        if len(store.full_days) < cfg.val_days + cfg.test_days + 7:
            raise ValueError(
                f"{store.station_name} does not have enough complete days for "
                f"{cfg.val_days} validation days and {cfg.test_days} test days."
            )
        test_days = store.full_days[-cfg.test_days :]
        val_days = store.full_days[-(cfg.test_days + cfg.val_days) : -cfg.test_days]
        train_cutoff = pd.Timestamp(val_days[0]).normalize()

        train_specs.extend(build_train_specs(store, cfg, train_cutoff=train_cutoff))
        val_specs.extend(build_daily_eval_specs(store, val_days, cfg, split="val"))
        test_specs.extend(build_daily_eval_specs(store, test_days, cfg, split="test"))

    return {
        "train": ForecastWindowDataset(stores, train_specs, cfg.seq_len, cfg.pred_len),
        "val": ForecastWindowDataset(stores, val_specs, cfg.seq_len, cfg.pred_len),
        "test": ForecastWindowDataset(stores, test_specs, cfg.seq_len, cfg.pred_len),
    }
