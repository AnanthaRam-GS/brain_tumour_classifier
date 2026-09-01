"""Generic Random Forest trainer for Phase 1 handcrafted feature tables.

Feature-set agnostic, mirroring ``src/models/svm.py``: works with any single
feature CSV following the shared ``sample_id,patient_id,label,split,<features>``
schema. Built only on the shared ``src.models.training`` and
``src.evaluation.{metrics,results}`` contracts -- nothing feature-specific is
hardcoded here.

Methodology (mirrors ``src.models.svm.run_svm_experiment`` exactly):
hyperparameters are grid-searched by fitting each candidate on TRAIN only and
scoring on VALIDATION only (primary metric: macro F1). The frozen best
hyperparameters are then used to fit a FINAL model on TRAIN ONLY -- the model
is never retrained on train+validation. Test is evaluated exactly once, after
selection is frozen.

Random Forest does not strictly need feature scaling, but the shared
``src.models.training`` preprocessing contract (median impute + train-only
StandardScaler) is applied anyway for consistency with the rest of the Phase
1 framework; this is a documented, harmless no-op for tree splits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.evaluation.metrics import CLASS_ORDER
from src.evaluation.results import (
    build_prediction_table,
    build_result_dict,
    scores_from_estimator,
    write_experiment_results,
)
from src.features.feature_table import merge_feature_tables, read_feature_table
from src.models.training import (
    prepare_feature_matrices,
    train_estimator,
    transform_feature_matrices,
)

DEFAULT_SPLIT_CSV = "data/splits/patient_split.csv"
DEFAULT_SPLIT_METADATA_JSON = "data/splits/split_metadata.json"
DEFAULT_OUTPUT_ROOT = "reports/experiments"
RANDOM_SEED = 42
SELECTION_METRIC = "validation macro F1"
FIXED_CLASS_WEIGHT = "balanced"
FIXED_N_JOBS = 1  # fixed (not -1) for bit-for-bit reproducibility across reruns

# Curated subset of the 3x3x3x3x2=162 full grid (small/reasonable, similar in
# spirit and size to svm.py's 5x6=30-candidate grid): a sensible center point
# plus one-at-a-time and a few joint variations. Order is significant -- it
# is the deterministic tie-break order documented in configs/models/random_forest.yaml.
RF_CANDIDATE_GRID: tuple[dict[str, Any], ...] = (
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 10, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 4, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "log2"},
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 20, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "log2"},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "log2"},
    {"n_estimators": 500, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": None, "min_samples_split": 10, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "log2"},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "log2"},
    {"n_estimators": 500, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "log2"},
    {"n_estimators": 500, "max_depth": 20, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "log2"},
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 20, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "log2"},
    {"n_estimators": 500, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 4, "max_features": "log2"},
)


def git_commit() -> str:
    """Return the current commit hash, or "unknown" outside a git checkout."""

    try:
        repo_root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        return "unknown"


def read_json_field(path: Path | str, field: str) -> str:
    """Read one string field from a small metadata JSON file, defaulting to "unknown"."""

    file_path = Path(path)
    if not file_path.is_file():
        return "unknown"
    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return str(payload.get(field, "unknown"))


def load_combined_feature_table(
    feature_csv: Path | str | Sequence[Path | str],
    *,
    split_csv: Path | str = DEFAULT_SPLIT_CSV,
) -> pd.DataFrame:
    """Load one feature CSV, or merge several by ``sample_id`` via the shared contract."""

    paths = [feature_csv] if isinstance(feature_csv, (str, Path)) else list(feature_csv)
    if not paths:
        raise ValueError("feature_csv must contain at least one path")
    tables = [read_feature_table(path, canonical_split=split_csv) for path in paths]
    if len(tables) == 1:
        return tables[0]
    return merge_feature_tables(tables, canonical_split=split_csv)


def build_random_forest(*, seed: int = RANDOM_SEED, **params: Any) -> RandomForestClassifier:
    """Return a configured ``RandomForestClassifier`` with fixed, documented defaults."""

    return RandomForestClassifier(
        random_state=seed,
        class_weight=FIXED_CLASS_WEIGHT,
        n_jobs=FIXED_N_JOBS,
        bootstrap=True,
        criterion="gini",
        **params,
    )


def tune_random_forest_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    candidate_grid: tuple[dict[str, Any], ...] = RF_CANDIDATE_GRID,
    seed: int = RANDOM_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Grid-search RF hyperparameters, selecting by validation macro F1.

    Each candidate is fit on TRAIN only and scored on VALIDATION only; test
    is never used. Ties are broken by (1) higher validation balanced
    accuracy, (2) higher validation accuracy, (3) earliest position in
    ``candidate_grid`` (deterministic insertion order).
    """

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, int] | None = None
    for index, params in enumerate(candidate_grid):
        model = build_random_forest(seed=seed, **params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_macro_f1 = float(
            f1_score(y_val, val_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
        )
        val_balanced_accuracy = float(balanced_accuracy_score(y_val, val_pred))
        val_accuracy = float(accuracy_score(y_val, val_pred))
        trial = {
            **params,
            "val_macro_f1": val_macro_f1,
            "val_balanced_accuracy": val_balanced_accuracy,
            "val_accuracy": val_accuracy,
            "grid_index": index,
        }
        trials.append(trial)
        # Maximize (macro_f1, balanced_accuracy, accuracy, -grid_index) so
        # an earlier grid_index wins ties (since we compare > strictly and
        # keep the first-seen best under equal scores).
        key = (val_macro_f1, val_balanced_accuracy, val_accuracy)
        if best is None or key > best_key:  # type: ignore[operator]
            best = trial
            best_key = key
    assert best is not None
    return best, trials


def run_random_forest_experiment(
    feature_csv: Path | str | Sequence[Path | str],
    *,
    feature_set_name: str,
    feature_version: str,
    experiment_name: str,
    split_csv: Path | str = DEFAULT_SPLIT_CSV,
    split_metadata_json: Path | str = DEFAULT_SPLIT_METADATA_JSON,
    split_version: str | None = None,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    seed: int = RANDOM_SEED,
    candidate_grid: tuple[dict[str, Any], ...] = RF_CANDIDATE_GRID,
) -> dict[str, Any]:
    """Fit preprocessing/RF on train, tune hyperparameters on val, evaluate once on test.

    Mirrors ``src.models.svm.run_svm_experiment``'s train-only-then-freeze
    convention: the final model is trained on TRAIN ONLY using the frozen
    best hyperparameters, never retrained on train+validation.
    """

    table = load_combined_feature_table(feature_csv, split_csv=split_csv)
    matrices = prepare_feature_matrices(table, canonical_split=str(split_csv))
    preprocessed = transform_feature_matrices(matrices, imputation="median", scaling="standard")

    best_trial, trials = tune_random_forest_hyperparameters(
        preprocessed.X_train,
        matrices.y_train,
        preprocessed.X_val,
        matrices.y_val,
        candidate_grid=candidate_grid,
        seed=seed,
    )
    best_params = {
        key: best_trial[key]
        for key in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features")
    }

    final_rf = build_random_forest(seed=seed, **best_params)
    trained = train_estimator(
        final_rf,
        preprocessed.X_train,
        matrices.y_train,
        model_name="random_forest",
        feature_count=len(matrices.feature_columns),
    )

    y_test_pred = trained.estimator.predict(preprocessed.X_test)
    y_test_score, score_reason = scores_from_estimator(trained.estimator, preprocessed.X_test)
    predictions = build_prediction_table(
        matrices.test_metadata, matrices.y_test, y_test_pred, y_score=y_test_score
    )

    result = build_result_dict(
        experiment_name=experiment_name,
        feature_set=feature_set_name,
        model_name=trained.model_name,
        split_evaluated="test",
        feature_count=trained.feature_count,
        metadata=matrices.test_metadata,
        y_true=matrices.y_test,
        y_pred=y_test_pred,
        model_parameters=trained.model_parameters,
        preprocessing={
            "imputation": preprocessed.preprocessor.imputation,
            "scaling": preprocessed.preprocessor.scaling,
            "hyperparameter_selection_metric": SELECTION_METRIC,
            "hyperparameter_selection_note": (
                "n_estimators/max_depth/min_samples_split/min_samples_leaf/"
                "max_features grid-searched by fitting on train and scoring "
                "on validation only (macro F1 primary, tie-break: validation "
                "balanced accuracy, then validation accuracy, then earliest "
                "grid position); test was not used until final evaluation. "
                "The final model is trained on train only (not train+val), "
                "mirroring src.models.svm.run_svm_experiment's convention."
            ),
        },
        random_seed=seed,
        split_version=split_version or read_json_field(split_metadata_json, "split_version"),
        feature_version=feature_version,
        code_commit=git_commit(),
        y_score=y_test_score if y_test_score is not None else None,
    )

    output_dir = Path(output_root) / experiment_name
    written = write_experiment_results(result, predictions, output_dir)

    search_log = {
        "selection_metric": SELECTION_METRIC,
        "tie_break": [
            "higher validation balanced accuracy",
            "higher validation accuracy",
            "earliest position in the candidate grid (deterministic insertion order)",
        ],
        "candidate_grid": list(candidate_grid),
        "chosen_hyperparameters": best_params,
        "chosen_trial": best_trial,
        "all_trials": trials,
        "score_availability_reason": score_reason,
        "train_vs_trainval_convention": (
            "final model trained on train split only, using hyperparameters "
            "frozen from validation-only selection; test evaluated exactly "
            "once. This mirrors src.models.svm.run_svm_experiment."
        ),
    }
    search_log_path = output_dir / "hyperparameter_search.json"
    search_log_path.write_text(json.dumps(search_log, indent=2) + "\n", encoding="utf-8")
    written["hyperparameter_search"] = search_log_path

    # Feature importances (impurity-based / Gini decrease, as computed by
    # sklearn's RandomForestClassifier.feature_importances_).
    importances = np.asarray(trained.estimator.feature_importances_, dtype=float)
    importance_frame = pd.DataFrame(
        {"feature": matrices.feature_columns, "importance": importances}
    )
    importance_frame = importance_frame.sort_values(
        ["importance", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    importance_frame.insert(0, "rank", np.arange(1, len(importance_frame) + 1))
    importance_frame = importance_frame[["feature", "importance", "rank"]]
    importance_path = output_dir / "feature_importance.csv"
    importance_frame.to_csv(importance_path, index=False)
    written["feature_importance"] = importance_path

    return {
        "result": result,
        "written": written,
        "best_params": best_params,
        "trials": trials,
        "feature_importance": importance_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-csv", nargs="+", required=True, type=Path,
        help="One feature CSV, or several to merge by sample_id.",
    )
    parser.add_argument("--feature-set-name", required=True)
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split-csv", type=Path, default=Path(DEFAULT_SPLIT_CSV))
    parser.add_argument(
        "--split-metadata-json", type=Path, default=Path(DEFAULT_SPLIT_METADATA_JSON)
    )
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    feature_csv = args.feature_csv[0] if len(args.feature_csv) == 1 else args.feature_csv
    outcome = run_random_forest_experiment(
        feature_csv,
        feature_set_name=args.feature_set_name,
        feature_version=args.feature_version,
        experiment_name=args.experiment_name,
        split_csv=args.split_csv,
        split_metadata_json=args.split_metadata_json,
        output_root=args.output_root,
        seed=args.seed,
    )
    print(json.dumps(outcome["result"]["metrics"], indent=2))
    print(json.dumps({"chosen_hyperparameters": outcome["best_params"]}, indent=2))


if __name__ == "__main__":
    main()
