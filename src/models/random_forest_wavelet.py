"""Wavelet-features CLI convenience wrapper around the generic RF trainer.

All reusable training/tuning logic lives in ``src.models.random_forest``;
this module only supplies the wavelet feature set's default paths and
labels so ``python -m src.models.random_forest_wavelet`` produces
``reports/experiments/wavelet_rf_v1/``. Nothing wavelet-specific is
duplicated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.models.random_forest import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPLIT_CSV,
    DEFAULT_SPLIT_METADATA_JSON,
    read_json_field,
    run_random_forest_experiment,
)

DEFAULT_FEATURE_CSV = "data/features/wavelet/wavelet_features.csv"
DEFAULT_FEATURE_METADATA_JSON = "data/features/wavelet/wavelet_metadata.json"
DEFAULT_EXPERIMENT_NAME = "wavelet_rf_v1"
FEATURE_SET_NAME = "wavelet"
RANDOM_SEED = 42


def run_wavelet_random_forest_experiment(
    *,
    feature_csv: Path | str = DEFAULT_FEATURE_CSV,
    feature_metadata_json: Path | str = DEFAULT_FEATURE_METADATA_JSON,
    split_csv: Path | str = DEFAULT_SPLIT_CSV,
    split_metadata_json: Path | str = DEFAULT_SPLIT_METADATA_JSON,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run the generic RF trainer with wavelet's feature set and defaults."""

    feature_version = read_json_field(feature_metadata_json, "feature_version")
    if feature_version == "unknown":
        feature_version = "v1"
    return run_random_forest_experiment(
        feature_csv,
        feature_set_name=FEATURE_SET_NAME,
        feature_version=feature_version,
        experiment_name=experiment_name,
        split_csv=split_csv,
        split_metadata_json=split_metadata_json,
        output_root=output_root,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-csv", type=Path, default=Path(DEFAULT_FEATURE_CSV))
    parser.add_argument(
        "--feature-metadata-json", type=Path, default=Path(DEFAULT_FEATURE_METADATA_JSON)
    )
    parser.add_argument("--split-csv", type=Path, default=Path(DEFAULT_SPLIT_CSV))
    parser.add_argument(
        "--split-metadata-json", type=Path, default=Path(DEFAULT_SPLIT_METADATA_JSON)
    )
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    outcome = run_wavelet_random_forest_experiment(
        feature_csv=args.feature_csv,
        feature_metadata_json=args.feature_metadata_json,
        split_csv=args.split_csv,
        split_metadata_json=args.split_metadata_json,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )
    print(json.dumps(outcome["result"]["metrics"], indent=2))
    print(json.dumps({"chosen_hyperparameters": outcome["best_params"]}, indent=2))


if __name__ == "__main__":
    main()
