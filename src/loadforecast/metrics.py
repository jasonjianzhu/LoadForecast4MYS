from __future__ import annotations

import math
from typing import Mapping

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean(np.square(y_pred - y_true))))


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.max(y_true) - np.min(y_true))
    return mae(y_true, y_pred) / max(denom, 1e-6)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.max(y_true) - np.min(y_true))
    return rmse(y_true, y_pred) / max(denom, 1e-6)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = float(np.sum(np.abs(y_pred - y_true)))
    denominator = float(np.sum(np.abs(y_true)))
    return numerator / max(denominator, 1e-6)


def mape_clipped(y_true: np.ndarray, y_pred: np.ndarray, floor_kw: float = 1.0) -> float:
    denominator = np.maximum(np.abs(y_true), floor_kw)
    return float(np.mean(np.abs(y_pred - y_true) / denominator))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "nmae": nmae(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "mape": mape_clipped(y_true, y_pred),
    }

