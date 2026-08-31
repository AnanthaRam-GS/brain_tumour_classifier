"""Gabor-features CLI convenience wrapper around the generic SVM trainer.

All reusable training/tuning logic lives in ``src.models.svm``; this module
only supplies Gabor's default paths and labels so ``python -m
src.models.svm_gabor`` keeps working unchanged. Nothing Gabor-specific is
duplicated here -- a teammate assembling a master pipeline should import
``src.models.svm.run_svm_experiment`` directly for other feature sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.models.svm import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPLIT_CSV,
    DEFAULT_SPLIT_METADATA_JSON,
    read_json_field,
    run_svm_experiment,
)

DEFAULT_FEATURE_CSV = "data/features/gabor/gabor_features.csv"
DEFAULT_FEATURE_METADATA_JSON = "data/features/gabor/gabor_metadata.json"
DEFAULT_EXPERIMENT_NAME = "gabor_svm_v1"
FEATURE_SET_NAME = "gabor"
RANDOM_SEED = 42


def run_gabor_svm_experiment(
    *,
    feature_csv: Path | str = DEFAULT_FEATURE_CSV,
    feature_metadata_json: Path | str = DEFAULT_FEATURE_METADATA_JSON,
    split_csv: Path | str = DEFAULT_SPLIT_CSV,
    split_metadata_json: Path | str = DEFAULT_SPLIT_METADATA_JSON,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run the generic SVM trainer with Gabor's feature set and defaults."""

    return run_svm_experiment(
        feature_csv,
        feature_set_name=FEATURE_SET_NAME,
        feature_version=read_json_field(feature_metadata_json, "feature_version"),
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

    outcome = run_gabor_svm_experiment(
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
