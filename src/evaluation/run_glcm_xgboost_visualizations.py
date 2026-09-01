"""Generate figures for the completed GLCM + XGBoost experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.visualize_experiment import generate_experiment_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path("reports/experiments/glcm_xgboost_v1"))
    parser.add_argument("--features", type=Path, default=Path("data/features/glcm/glcm_features.csv"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    generate_experiment_visualizations(
        experiment_dir=args.experiment_dir,
        features=args.features,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
