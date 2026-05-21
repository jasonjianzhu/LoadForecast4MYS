from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loadforecast.plotting import generate_test_prediction_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot test predictions vs actual values.")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/timexer_final_affine_off/test_predictions.csv",
        help="Path to the test prediction CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/timexer_final_affine_off/plots",
        help="Directory to write SVG plots into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    generate_test_prediction_plots(input_path, output_dir)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
