"""XGBoost-specific helpers for the GLCM Review 1 experiment."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from xgboost import XGBClassifier

PROJECT_LABELS = [1, 2, 3]
INTERNAL_LABELS = [0, 1, 2]


class XGBoostModelError(ValueError):
    """Raised when the XGBoost model-specific contract is violated."""


def project_to_internal_labels(labels: np.ndarray) -> np.ndarray:
    """Map project labels 1/2/3 to XGBoost's zero-based labels 0/1/2."""

    values = np.asarray(labels, dtype=int)
    if not set(np.unique(values)).issubset(PROJECT_LABELS):
        raise XGBoostModelError("project labels must be in {1, 2, 3}")
    return values - 1


def internal_to_project_labels(labels: np.ndarray) -> np.ndarray:
    """Map zero-based XGBoost labels back to project labels 1/2/3."""

    values = np.asarray(labels, dtype=int)
    if not set(np.unique(values)).issubset(INTERNAL_LABELS):
        raise XGBoostModelError("internal labels must be in {0, 1, 2}")
    return values + 1


def load_xgboost_config(path: Path | str = "configs/models/xgboost.yaml") -> dict[str, Any]:
    """Load and validate the frozen XGBoost experiment config."""

    config_path = Path(path)
    if not config_path.is_file():
        raise XGBoostModelError(f"XGBoost config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise XGBoostModelError("XGBoost config must be a YAML mapping")
    return validate_xgboost_config(config)


def validate_xgboost_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate required model config fields and the small deterministic grid."""

    required = {
        "model_name",
        "model_version",
        "random_seed",
        "feature_set",
        "objective",
        "num_class",
        "eval_metric",
        "preprocessing",
        "fixed_parameters",
        "candidate_grid",
        "selection_policy",
        "final_retraining_policy",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise XGBoostModelError(f"XGBoost config missing field(s): {', '.join(missing)}")
    if config["model_name"] != "xgboost":
        raise XGBoostModelError("model_name must be xgboost")
    if config["feature_set"] != "glcm":
        raise XGBoostModelError("feature_set must be glcm")
    if config["objective"] != "multi:softprob":
        raise XGBoostModelError("objective must be multi:softprob")
    if config["num_class"] != 3:
        raise XGBoostModelError("num_class must be 3")
    if config["random_seed"] != 42:
        raise XGBoostModelError("random_seed must be 42")
    preprocessing = config["preprocessing"]
    if preprocessing.get("imputation") != "median" or preprocessing.get("scaling") is not None:
        raise XGBoostModelError("XGBoost preprocessing must use median imputation and no scaling")
    grid = config["candidate_grid"]
    for key in ("n_estimators", "max_depth", "learning_rate"):
        if key not in grid or not isinstance(grid[key], list) or not grid[key]:
            raise XGBoostModelError(f"candidate_grid.{key} must be a non-empty list")
    if len(candidate_parameter_grid(config)) != 8:
        raise XGBoostModelError("candidate grid must contain exactly 8 configurations")
    if config["selection_policy"].get("primary_metric") != "macro_f1":
        raise XGBoostModelError("primary selection metric must be macro_f1")
    if config["final_retraining_policy"] != "train_plus_validation":
        raise XGBoostModelError("final retraining policy must be train_plus_validation")
    return config


def candidate_parameter_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidate parameters in deterministic grid order."""

    grid = config["candidate_grid"]
    return [
        {
            **config["fixed_parameters"],
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
        }
        for n_estimators, max_depth, learning_rate in product(
            grid["n_estimators"], grid["max_depth"], grid["learning_rate"]
        )
    ]


def create_xgboost_estimator(config: dict[str, Any], parameters: dict[str, Any]) -> XGBClassifier:
    """Create an XGBClassifier from frozen common parameters plus one candidate."""

    return XGBClassifier(
        objective=config["objective"],
        num_class=config["num_class"],
        eval_metric=config["eval_metric"],
        random_state=config["random_seed"],
        **parameters,
    )


def fit_xgboost_estimator(
    estimator: XGBClassifier,
    X: np.ndarray,
    y_project: np.ndarray,
) -> XGBClassifier:
    """Fit XGBoost with zero-based internal labels."""

    estimator.fit(X, project_to_internal_labels(y_project))
    return estimator


def predict_project_labels(estimator: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Predict project labels 1/2/3 from an XGBoost estimator."""

    return internal_to_project_labels(estimator.predict(X))


def predict_project_proba(estimator: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return probabilities ordered for project classes [1, 2, 3]."""

    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = [int(label) for label in estimator.classes_]
    if classes != INTERNAL_LABELS:
        raise XGBoostModelError(f"unexpected XGBoost class order: {classes}")
    if probabilities.shape[1] != len(PROJECT_LABELS):
        raise XGBoostModelError("XGBoost probability output must have three columns")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(row_sums <= 0):
        raise XGBoostModelError("XGBoost probability output is invalid")
    return probabilities / row_sums


def save_xgboost_artifacts(
    *,
    estimator: XGBClassifier,
    preprocessor: Any,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Save the native XGBoost model and fitted preprocessing state."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "xgboost_model.json"
    imputer_path = root / "imputer.joblib"
    bundle_path = root / "model_bundle.joblib"
    estimator.save_model(model_path)
    joblib.dump(preprocessor.pipeline, imputer_path)
    joblib.dump({"estimator": estimator, "preprocessor": preprocessor}, bundle_path)
    return {"model_json": model_path, "imputer": imputer_path, "bundle": bundle_path}


def load_xgboost_bundle(path: Path | str) -> dict[str, Any]:
    """Load a saved joblib model bundle."""

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "estimator" not in bundle or "preprocessor" not in bundle:
        raise XGBoostModelError("invalid XGBoost model bundle")
    return bundle


def write_json(payload: dict[str, Any], path: Path | str) -> None:
    """Write JSON with strict finite values."""

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
