"""Prediction tables and result files for Phase 1 classical ML experiments."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import CLASS_NAMES, CLASS_ORDER, EvaluationError, evaluate_predictions

PREDICTION_COLUMNS = ["sample_id", "patient_id", "true_label", "predicted_label"]
REQUIRED_RESULT_FIELDS = {
    "experiment_name",
    "feature_set",
    "model_name",
    "split_evaluated",
    "feature_count",
    "sample_count",
    "patient_count",
    "class_order",
    "class_names",
    "metrics",
    "per_class_metrics",
    "confusion_matrix",
    "roc_auc",
    "model_parameters",
    "preprocessing",
    "random_seed",
    "split_version",
    "feature_version",
    "generated_at",
    "code_commit",
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".csv", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def scores_from_estimator(estimator: Any, X: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Return class-ordered scores when an estimator exposes probabilities."""

    if not hasattr(estimator, "predict_proba"):
        return None, "predict_proba unavailable"
    if not hasattr(estimator, "classes_"):
        return None, "estimator.classes_ unavailable"
    classes = [int(label) for label in estimator.classes_]
    if not set(CLASS_ORDER).issubset(classes):
        return None, f"estimator classes do not cover {CLASS_ORDER}: {classes}"
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    ordered = np.zeros((proba.shape[0], len(CLASS_ORDER)), dtype=float)
    for output_index, label in enumerate(CLASS_ORDER):
        ordered[:, output_index] = proba[:, classes.index(label)]
    return ordered, ""


def build_prediction_table(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_score: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build deterministic prediction rows with optional class-named probabilities."""

    frame = metadata.loc[:, ["sample_id", "patient_id"]].copy()
    frame["true_label"] = np.asarray(y_true, dtype=int)
    frame["predicted_label"] = np.asarray(y_pred, dtype=int)
    if y_score is not None:
        scores = np.asarray(y_score, dtype=float)
        if scores.shape != (len(frame), len(CLASS_ORDER)):
            raise EvaluationError(
                f"probability scores must have shape ({len(frame)}, {len(CLASS_ORDER)})"
            )
        for index, label in enumerate(CLASS_ORDER):
            frame[f"prob_{CLASS_NAMES[label]}"] = scores[:, index]
    return frame.sort_values("sample_id", key=lambda column: column.astype(int)).reset_index(drop=True)


def build_result_dict(
    *,
    experiment_name: str,
    feature_set: str,
    model_name: str,
    split_evaluated: str,
    feature_count: int,
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_parameters: dict[str, Any],
    preprocessing: dict[str, Any],
    random_seed: int | None,
    split_version: str,
    feature_version: str,
    code_commit: str = "unknown",
    y_score: np.ndarray | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable experiment result schema."""

    evaluation = evaluate_predictions(y_true, y_pred, y_score=y_score)
    result = {
        "experiment_name": experiment_name,
        "feature_set": feature_set,
        "model_name": model_name,
        "split_evaluated": split_evaluated,
        "feature_count": int(feature_count),
        "sample_count": int(len(y_true)),
        "patient_count": int(metadata["patient_id"].nunique()),
        "class_order": CLASS_ORDER,
        "class_names": {str(label): CLASS_NAMES[label] for label in CLASS_ORDER},
        "metrics": evaluation["metrics"],
        "per_class_metrics": evaluation["per_class_metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "roc_auc": evaluation["roc_auc"],
        "model_parameters": model_parameters,
        "preprocessing": preprocessing,
        "random_seed": random_seed,
        "split_version": split_version,
        "feature_version": feature_version,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
    }
    return validate_result_dict(result)


def validate_result_dict(result: dict[str, Any]) -> dict[str, Any]:
    """Validate the shared result dictionary schema."""

    missing = sorted(REQUIRED_RESULT_FIELDS.difference(result))
    if missing:
        raise EvaluationError(f"result missing required field(s): {', '.join(missing)}")
    if result["class_order"] != CLASS_ORDER:
        raise EvaluationError("result class_order must be [1, 2, 3]")
    matrix = np.asarray(result["confusion_matrix"])
    if matrix.shape != (3, 3):
        raise EvaluationError("confusion_matrix must be 3x3")
    json.dumps(result, allow_nan=False)
    return result


def write_experiment_results(
    result: dict[str, Any],
    predictions: pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Write metrics, predictions, confusion matrix, and metadata files."""

    validated = validate_result_dict(result)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(validated, root / "metrics.json")
    _atomic_csv(predictions, root / "predictions.csv")
    matrix = pd.DataFrame(
        validated["confusion_matrix"],
        index=[CLASS_NAMES[label] for label in CLASS_ORDER],
        columns=[CLASS_NAMES[label] for label in CLASS_ORDER],
    )
    matrix.index.name = "true_label"
    _atomic_csv(matrix.reset_index(), root / "confusion_matrix.csv")
    metadata = {
        key: validated[key]
        for key in (
            "experiment_name",
            "feature_set",
            "model_name",
            "split_evaluated",
            "feature_count",
            "sample_count",
            "patient_count",
            "class_order",
            "class_names",
            "model_parameters",
            "preprocessing",
            "random_seed",
            "split_version",
            "feature_version",
            "generated_at",
            "code_commit",
        )
    }
    _atomic_json(metadata, root / "experiment_metadata.json")
    return {
        "metrics": root / "metrics.json",
        "predictions": root / "predictions.csv",
        "confusion_matrix": root / "confusion_matrix.csv",
        "experiment_metadata": root / "experiment_metadata.json",
    }


def read_result_json(path: Path | str) -> dict[str, Any]:
    """Read and validate a result JSON file."""

    with Path(path).open(encoding="utf-8") as handle:
        return validate_result_dict(json.load(handle))
