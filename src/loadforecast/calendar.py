from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
except ImportError:  # pragma: no cover - dependency is optional at import time.
    holidays_lib = None


@dataclass(frozen=True)
class HolidaySets:
    federal: frozenset[pd.Timestamp]
    state_only: frozenset[pd.Timestamp]

    @property
    def union(self) -> frozenset[pd.Timestamp]:
        return self.federal | self.state_only


def load_holiday_sets(
    years: Iterable[int],
    country: str,
    subdiv: str,
    strict: bool = True,
) -> HolidaySets:
    if holidays_lib is None:
        if strict:
            raise RuntimeError(
                "The `holidays` package is required for Malaysia holiday features. "
                "Install it in the project virtual environment before training."
            )
        return HolidaySets(federal=frozenset(), state_only=frozenset())

    federal_calendar = holidays_lib.country_holidays(country, years=list(years))
    state_calendar = holidays_lib.country_holidays(country, subdiv=subdiv, years=list(years))

    federal = frozenset(pd.Timestamp(day) for day in federal_calendar.keys())
    state_all = frozenset(pd.Timestamp(day) for day in state_calendar.keys())
    state_only = frozenset(day for day in state_all if day not in federal)
    return HolidaySets(federal=federal, state_only=state_only)


def build_calendar_features(
    index: pd.DatetimeIndex,
    country: str,
    subdiv: str,
    strict: bool,
    pre_holiday_days: int,
    post_holiday_days: int,
) -> pd.DataFrame:
    years = sorted({timestamp.year for timestamp in index})
    holiday_sets = load_holiday_sets(years, country, subdiv, strict=strict)

    normalized = pd.DatetimeIndex(index.normalize())
    minute_of_day = index.hour * 60 + index.minute
    day_of_week = index.dayofweek

    holiday_union = holiday_sets.union
    pre_holiday_dates = set()
    post_holiday_dates = set()
    for holiday in holiday_union:
        for offset in range(1, pre_holiday_days + 1):
            pre_holiday_dates.add(holiday - pd.Timedelta(days=offset))
        for offset in range(1, post_holiday_days + 1):
            post_holiday_dates.add(holiday + pd.Timedelta(days=offset))

    features = pd.DataFrame(index=index)
    features["minute_of_day_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    features["minute_of_day_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    features["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    features["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    features["is_weekend"] = (day_of_week >= 5).astype(np.float32)
    features["is_federal_holiday"] = normalized.isin(holiday_sets.federal).astype(np.float32)
    features["is_state_holiday"] = normalized.isin(holiday_sets.state_only).astype(np.float32)
    features["is_pre_holiday"] = normalized.isin(pre_holiday_dates).astype(np.float32)
    features["is_post_holiday"] = normalized.isin(post_holiday_dates).astype(np.float32)
    return features.astype(np.float32)


def classify_day_type(day: pd.Timestamp, holiday_sets: HolidaySets) -> str:
    normalized = pd.Timestamp(day).normalize()
    if normalized in holiday_sets.union:
        return "holiday"
    if normalized.dayofweek >= 5:
        return "weekend"
    return "weekday"
