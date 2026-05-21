from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd

from loadforecast.metrics import compute_all_metrics


@dataclass
class SeriesPlotData:
    series_name: str
    timestamps: list[pd.Timestamp]
    y_true: list[float]
    y_pred: list[float]

    @property
    def short_name(self) -> str:
        name = self.series_name
        suffixes = [
            "_20260213_20260512_load",
            "_20260326_20260512(1)",
            "_20260206_20260512_load",
            "_20260210_20260512_load(PV&meter)",
        ]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name

    @property
    def safe_name(self) -> str:
        safe_name = (
            self.short_name.lower()
            .replace(" ", "_")
            .replace("&", "and")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
        )
        return "".join(ch for ch in safe_name if ch.isalnum() or ch == "_")

    @property
    def metrics(self) -> dict[str, float]:
        return dict(compute_all_metrics(pd.Series(self.y_true).to_numpy(), pd.Series(self.y_pred).to_numpy()))


def load_series_data(path: Path) -> list[SeriesPlotData]:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["series_name", "timestamp"])

    series_list: list[SeriesPlotData] = []
    for series_name, group in frame.groupby("series_name", sort=True):
        series_list.append(
            SeriesPlotData(
                series_name=series_name,
                timestamps=group["timestamp"].tolist(),
                y_true=group["y_true"].astype(float).tolist(),
                y_pred=group["y_pred"].astype(float).tolist(),
            )
        )
    return series_list


def _polyline(points: Iterable[tuple[float, float]], color: str, width: float) -> str:
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{coords}" />'


def _nice_ticks(ymin: float, ymax: float, count: int = 5) -> list[float]:
    if ymax <= ymin:
        return [ymin]
    span = ymax - ymin
    step = span / max(count - 1, 1)
    magnitude = 10 ** int(len(str(int(step))) - 1) if step >= 1 else 0.1
    for factor in (1, 2, 5, 10):
        candidate = factor * magnitude
        if candidate >= step:
            step = candidate
            break
    start = step * int(ymin // step)
    ticks = []
    value = start
    while value <= ymax + step:
        if value >= ymin - 1e-9:
            ticks.append(round(value, 6))
        value += step
    return ticks


def render_single_series_svg(data: SeriesPlotData, output_path: Path, width: int = 1600, height: int = 520) -> None:
    margin_left = 78
    margin_right = 22
    margin_top = 56
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    y_values = data.y_true + data.y_pred
    ymin = min(y_values)
    ymax = max(y_values)
    pad = max((ymax - ymin) * 0.08, 1.0)
    ymin -= pad
    ymax += pad
    ticks = _nice_ticks(ymin, ymax, count=6)

    x_count = len(data.timestamps)
    x_max = max(x_count - 1, 1)

    def x_to_px(index: int) -> float:
        return margin_left + plot_width * index / x_max

    def y_to_px(value: float) -> float:
        return margin_top + plot_height * (1.0 - (value - ymin) / (ymax - ymin))

    true_points = [(x_to_px(i), y_to_px(v)) for i, v in enumerate(data.y_true)]
    pred_points = [(x_to_px(i), y_to_px(v)) for i, v in enumerate(data.y_pred)]

    unique_days = sorted({timestamp.normalize() for timestamp in data.timestamps})
    day_to_first_idx = {
        day: next(i for i, timestamp in enumerate(data.timestamps) if timestamp.normalize() == day)
        for day in unique_days
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{margin_left}" y="28" font-size="22" font-family="Arial, sans-serif" fill="#111827">{escape(data.short_name)} - Test Set</text>',
        f'<text x="{margin_left}" y="48" font-size="12" font-family="Arial, sans-serif" fill="#6b7280">Actual vs Predicted, 5-minute resolution, 7-day test horizon</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#d1d5db" stroke-width="1" />',
    ]

    for tick in ticks:
        py = y_to_px(tick)
        parts.append(f'<line x1="{margin_left}" y1="{py:.2f}" x2="{margin_left + plot_width}" y2="{py:.2f}" stroke="#e5e7eb" stroke-width="1" />')
        parts.append(f'<text x="{margin_left - 10}" y="{py + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#6b7280">{tick:.0f}</text>')

    for day in unique_days:
        idx = day_to_first_idx[day]
        px = x_to_px(idx)
        parts.append(f'<line x1="{px:.2f}" y1="{margin_top}" x2="{px:.2f}" y2="{margin_top + plot_height}" stroke="#f3f4f6" stroke-width="1" />')
        parts.append(f'<text x="{px:.2f}" y="{height - 24}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif" fill="#6b7280">{day.strftime("%m-%d")}</text>')

    parts.append(_polyline(true_points, "#111827", 1.7))
    parts.append(_polyline(pred_points, "#2563eb", 1.4))

    metrics = data.metrics
    box_x = margin_left + 18
    box_y = margin_top + 16
    box_w = 320
    box_h = 68
    parts.append(f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="8" fill="white" fill-opacity="0.92" stroke="#d1d5db" stroke-width="1" />')
    metric_lines = [
        f"MAE {metrics['mae']:.2f}  |  RMSE {metrics['rmse']:.2f}",
        f"NMAE {metrics['nmae']:.4f}  |  NRMSE {metrics['nrmse']:.4f}",
        f"WAPE {metrics['wape']:.4f}  |  MAPE {metrics['mape']:.4f}",
    ]
    for idx, line in enumerate(metric_lines):
        parts.append(
            f'<text x="{box_x + 14}" y="{box_y + 20 + idx * 18}" font-size="12" font-family="Arial, sans-serif" fill="#374151">{escape(line)}</text>'
        )

    legend_x = width - 180
    legend_y = 26
    parts.extend(
        [
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="#111827" stroke-width="2.5" />',
            f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-size="12" font-family="Arial, sans-serif" fill="#374151">Actual</text>',
            f'<line x1="{legend_x}" y1="{legend_y + 22}" x2="{legend_x + 24}" y2="{legend_y + 22}" stroke="#2563eb" stroke-width="2.5" />',
            f'<text x="{legend_x + 32}" y="{legend_y + 26}" font-size="12" font-family="Arial, sans-serif" fill="#374151">Predicted</text>',
            f'<text x="22" y="{margin_top + plot_height / 2:.2f}" font-size="12" font-family="Arial, sans-serif" fill="#6b7280">kW</text>',
        ]
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def render_combined_svg(series_list: list[SeriesPlotData], output_path: Path) -> None:
    panel_width = 980
    panel_height = 360
    cols = 2
    rows = 2
    width = panel_width * cols
    height = panel_height * rows
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
    ]

    for idx, data in enumerate(series_list):
        col = idx % cols
        row = idx // cols
        x0 = col * panel_width
        y0 = row * panel_height
        margin_left = x0 + 62
        margin_right = x0 + panel_width - 18
        margin_top = y0 + 44
        margin_bottom = y0 + panel_height - 48
        plot_width = margin_right - margin_left
        plot_height = margin_bottom - margin_top

        y_values = data.y_true + data.y_pred
        ymin = min(y_values)
        ymax = max(y_values)
        pad = max((ymax - ymin) * 0.08, 1.0)
        ymin -= pad
        ymax += pad
        ticks = _nice_ticks(ymin, ymax, count=5)
        x_count = len(data.timestamps)
        x_max = max(x_count - 1, 1)

        def x_to_px(local_index: int) -> float:
            return margin_left + plot_width * local_index / x_max

        def y_to_px(value: float) -> float:
            return margin_top + plot_height * (1.0 - (value - ymin) / (ymax - ymin))

        parts.append(f'<text x="{margin_left}" y="{y0 + 24}" font-size="17" font-family="Arial, sans-serif" fill="#111827">{escape(data.short_name)}</text>')
        parts.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#d1d5db" stroke-width="1" />')

        for tick in ticks:
            py = y_to_px(tick)
            parts.append(f'<line x1="{margin_left}" y1="{py:.2f}" x2="{margin_right}" y2="{py:.2f}" stroke="#eef2f7" stroke-width="1" />')
            parts.append(f'<text x="{margin_left - 8}" y="{py + 4:.2f}" text-anchor="end" font-size="10" font-family="Arial, sans-serif" fill="#6b7280">{tick:.0f}</text>')

        unique_days = sorted({timestamp.normalize() for timestamp in data.timestamps})
        for day in unique_days:
            first_idx = next(i for i, timestamp in enumerate(data.timestamps) if timestamp.normalize() == day)
            px = x_to_px(first_idx)
            parts.append(f'<line x1="{px:.2f}" y1="{margin_top}" x2="{px:.2f}" y2="{margin_bottom}" stroke="#f8fafc" stroke-width="1" />')
            parts.append(f'<text x="{px:.2f}" y="{margin_bottom + 18}" text-anchor="middle" font-size="10" font-family="Arial, sans-serif" fill="#6b7280">{day.strftime("%m-%d")}</text>')

        true_points = [(x_to_px(i), y_to_px(v)) for i, v in enumerate(data.y_true)]
        pred_points = [(x_to_px(i), y_to_px(v)) for i, v in enumerate(data.y_pred)]
        parts.append(_polyline(true_points, "#111827", 1.4))
        parts.append(_polyline(pred_points, "#2563eb", 1.2))

        metrics = data.metrics
        box_x = margin_left + 12
        box_y = margin_top + 12
        box_w = 270
        box_h = 56
        parts.append(f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="8" fill="white" fill-opacity="0.92" stroke="#d1d5db" stroke-width="1" />')
        combined_lines = [
            f"MAE {metrics['mae']:.1f} | RMSE {metrics['rmse']:.1f}",
            f"NMAE {metrics['nmae']:.3f} | NRMSE {metrics['nrmse']:.3f}",
            f"WAPE {metrics['wape']:.3f} | MAPE {metrics['mape']:.3f}",
        ]
        for line_idx, line in enumerate(combined_lines):
            parts.append(
                f'<text x="{box_x + 10}" y="{box_y + 16 + line_idx * 14}" font-size="10" font-family="Arial, sans-serif" fill="#374151">{escape(line)}</text>'
            )

    legend_y = 20
    parts.extend(
        [
            f'<line x1="{width - 220}" y1="{legend_y}" x2="{width - 196}" y2="{legend_y}" stroke="#111827" stroke-width="2.5" />',
            f'<text x="{width - 188}" y="{legend_y + 4}" font-size="12" font-family="Arial, sans-serif" fill="#374151">Actual</text>',
            f'<line x1="{width - 120}" y1="{legend_y}" x2="{width - 96}" y2="{legend_y}" stroke="#2563eb" stroke-width="2.5" />',
            f'<text x="{width - 88}" y="{legend_y + 4}" font-size="12" font-family="Arial, sans-serif" fill="#374151">Predicted</text>',
        ]
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def generate_test_prediction_plots(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    series_list = load_series_data(input_path)
    for data in series_list:
        render_single_series_svg(data, output_dir / f"{data.safe_name}_test_curve.svg")
    render_combined_svg(series_list, output_dir / "test_curves_4_sites.svg")
