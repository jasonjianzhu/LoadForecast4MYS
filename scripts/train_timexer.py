from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loadforecast.config import ExperimentConfig
from loadforecast.data import build_datasets
from loadforecast.trainer import train_and_evaluate

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TimeXer-based load forecasting model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/timexer_default.json",
        help="Path to the JSON config file.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    datasets = build_datasets(config.data)
    output_dir = train_and_evaluate(datasets, config)
    LOGGER.info("training_complete output_dir=%s", output_dir.resolve())


if __name__ == "__main__":
    main()
