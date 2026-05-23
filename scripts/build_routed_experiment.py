from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loadforecast.metrics import compute_all_metrics, compute_peak_metrics
from loadforecast.plotting import generate_test_prediction_plots


DEFAULT_ROUTE_MAP = {
    "Simpli-GS Paperboard & Packaging Sdn Bhd_20260213_20260512_load": "peak",
    "Tamura_Electronics_20260210_20260512_load(PV&meter)": "peak",
    "Simpli-Plastone Technolngy Packaging Sdn Bhd_20260326_20260512(1)": "base",
    "Simpli-Quality-Coils_20260206_20260512_load": "base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a routed experiment by selecting per-station predictions from multiple runs."
    )
    parser.add_argument(
        "--base-input",
        type=str,
        default="outputs/experiments/patchtst_e04_balanced_sampler/test_predictions.csv",
        help="Exploded test prediction CSV for the baseline source run.",
    )
    parser.add_argument(
        "--peak-input",
        type=str,
        default="outputs/experiments/patchtst_e04_e02_peak_loss/test_predictions.csv",
        help="Exploded test prediction CSV for the peak-focused source run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments/patchtst_station_routed",
        help="Directory to write routed outputs into.",
    )
    parser.add_argument(
        "--route-map",
        type=str,
        default="",
        help="Optional JSON file mapping series_name to source label: base or peak.",
    )
    return parser.parse_args()


def load_route_map(path: str) -> dict[str, str]:
    if not path:
        return dict(DEFAULT_ROUTE_MAP)
    route_path = Path(path)
    data = json.loads(route_path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in data.items()}


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"series_name", "scenario", "split", "timestamp", "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values(["series_name", "timestamp"]).reset_index(drop=True)


def select_routed_predictions(
    source_frames: dict[str, pd.DataFrame],
    route_map: dict[str, str],
) -> pd.DataFrame:
    available_series = sorted(
        set().union(*(frame["series_name"].unique().tolist() for frame in source_frames.values()))
    )
    missing_routes = [series for series in available_series if series not in route_map]
    if missing_routes:
        raise ValueError(f"Missing route entries for series: {missing_routes}")

    routed_parts: list[pd.DataFrame] = []
    for series_name in available_series:
        source_name = route_map[series_name]
        if source_name not in source_frames:
            raise ValueError(f"Unknown source '{source_name}' for series '{series_name}'")
        selected = source_frames[source_name]
        series_frame = selected[selected["series_name"] == series_name].copy()
        if series_frame.empty:
            raise ValueError(f"No rows found for series '{series_name}' in source '{source_name}'")
        series_frame["routed_source"] = source_name
        routed_parts.append(series_frame)
    return pd.concat(routed_parts, ignore_index=True).sort_values(
        ["series_name", "timestamp"]
    ).reset_index(drop=True)


def summarize_frame(frame: pd.DataFrame) -> dict[str, float]:
    metrics = compute_all_metrics(
        frame["y_true"].to_numpy(dtype=float),
        frame["y_pred"].to_numpy(dtype=float),
    )
    return {key: float(value) for key, value in metrics.items()}


def summarize_by_group(frame: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_value, group in frame.groupby(group_column, sort=True):
        metrics = summarize_frame(group)
        metrics[group_column] = group_value
        rows.append(metrics)
    return rows


def summarize_peak_by_series(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for series_name, series_group in frame.groupby("series_name", sort=True):
        peak_ratios: list[float] = []
        peak_errs: list[float] = []
        nonpeak_biases: list[float] = []
        day_groups = series_group.groupby(series_group["timestamp"].dt.normalize(), sort=True)
        for _, day_group in day_groups:
            day_group = day_group.sort_values("timestamp")
            peak = compute_peak_metrics(
                day_group["y_true"].to_numpy(dtype=float),
                day_group["y_pred"].to_numpy(dtype=float),
            )
            peak_ratios.append(float(peak["peak_ratio"]))
            peak_errs.append(float(peak["peak_err"]))
            nonpeak_bias = float(peak["nonpeak_bias"])
            if pd.notna(nonpeak_bias):
                nonpeak_biases.append(nonpeak_bias)
        rows.append(
            {
                "series_name": series_name,
                "peak_ratio_mean": float(sum(peak_ratios) / max(len(peak_ratios), 1)),
                "peak_err_mean": float(sum(peak_errs) / max(len(peak_errs), 1)),
                "nonpeak_bias_mean": (
                    float(sum(nonpeak_biases) / len(nonpeak_biases)) if nonpeak_biases else float("nan")
                ),
            }
        )
    return rows


def build_summary(frame: pd.DataFrame, route_map: dict[str, str], source_paths: dict[str, str]) -> dict[str, Any]:
    return {
        "route_map": route_map,
        "source_predictions": source_paths,
        "test": summarize_frame(frame),
        "test_by_series": summarize_by_group(frame, "series_name"),
        "test_by_scenario": summarize_by_group(frame, "scenario"),
        "test_by_source": summarize_by_group(frame, "routed_source"),
        "test_peak_by_series": summarize_peak_by_series(frame),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_map = load_route_map(args.route_map)
    source_paths = {
        "base": args.base_input,
        "peak": args.peak_input,
    }
    source_frames = {
        source_name: load_predictions(Path(path))
        for source_name, path in source_paths.items()
    }
    routed = select_routed_predictions(source_frames=source_frames, route_map=route_map)

    test_csv_path = output_dir / "test_predictions.csv"
    routed.to_csv(test_csv_path, index=False)

    summary = build_summary(frame=routed, route_map=route_map, source_paths=source_paths)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "route_map.json").write_text(
        json.dumps(route_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    generate_test_prediction_plots(test_csv_path, output_dir / "plots")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
