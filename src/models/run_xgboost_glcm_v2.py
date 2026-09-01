"""Run the weighted XGBoost improvement experiment for GLCM v2."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost
import yaml
from sklearn.metrics import log_loss

from src.evaluation.results import build_prediction_table, build_result_dict
from src.features.feature_table import read_feature_metadata, read_feature_table
from src.models.run_xgboost import json_safe, write_json, write_split_outputs
from src.models.training import FeatureMatrices, fit_feature_preprocessor, prepare_feature_matrices
from src.models.xgboost_model import (
    PROJECT_LABELS,
    create_xgboost_estimator,
    fit_xgboost_estimator,
    predict_project_labels,
    predict_project_proba,
    save_xgboost_artifacts,
)

WEIGHTING_MODES = (
    "none",
    "class_balanced",
    "patient_balanced",
    "class_and_patient_balanced",
)


class XGBoostGLCMV2Error(ValueError):
    """Raised when the GLCM v2 XGBoost experiment contract is violated."""


def get_code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def load_xgboost_glcm_v2_config(
    path: Path | str = "configs/models/xgboost_glcm_v2.yaml",
) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise XGBoostGLCMV2Error(f"config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise XGBoostGLCMV2Error("config must be a YAML mapping")
    return validate_xgboost_glcm_v2_config(config)


def validate_xgboost_glcm_v2_config(config: dict[str, Any]) -> dict[str, Any]:
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
        "weighting_modes",
        "selection_policy",
        "promotion_policy",
        "final_retraining_policy",
        "feature_version",
        "split_version",
        "split_seed",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise XGBoostGLCMV2Error(f"config missing field(s): {', '.join(missing)}")
    if config["model_name"] != "xgboost" or config["model_version"] != "v2":
        raise XGBoostGLCMV2Error("config must describe xgboost v2")
    if config["feature_set"] != "glcm_v2" or config["feature_version"] != "v2":
        raise XGBoostGLCMV2Error("config must consume glcm_v2 feature version v2")
    if config["objective"] != "multi:softprob" or int(config["num_class"]) != 3:
        raise XGBoostGLCMV2Error("XGBoost must use multi:softprob with num_class=3")
    preprocessing = config["preprocessing"]
    if preprocessing.get("imputation") != "median" or preprocessing.get("scaling") is not None:
        raise XGBoostGLCMV2Error("preprocessing must be median imputation with no scaling")
    if tuple(config["weighting_modes"]) != WEIGHTING_MODES:
        raise XGBoostGLCMV2Error(f"weighting_modes must be {WEIGHTING_MODES}")
    if len(candidate_parameter_grid(config)) != 108:
        raise XGBoostGLCMV2Error("GLCM v2 search must contain 108 weighted candidates")
    if config["selection_policy"].get("primary_metric") != "macro_f1":
        raise XGBoostGLCMV2Error("primary selection metric must be macro_f1")
    return config


def candidate_parameter_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["candidate_grid"]
    rows = []
    for weighting_mode, n_estimators, max_depth, learning_rate in product(
        config["weighting_modes"],
        grid["n_estimators"],
        grid["max_depth"],
        grid["learning_rate"],
    ):
        rows.append(
            {
                **config["fixed_parameters"],
                "weighting_mode": str(weighting_mode),
                "n_estimators": int(n_estimators),
                "max_depth": int(max_depth),
                "learning_rate": float(learning_rate),
            }
        )
    return rows


def compute_sample_weights(
    metadata: pd.DataFrame,
    y: np.ndarray,
    *,
    mode: str,
) -> np.ndarray | None:
    """Compute train-scope sample weights only from the provided rows."""

    if mode not in WEIGHTING_MODES:
        raise XGBoostGLCMV2Error(f"unsupported weighting mode: {mode}")
    if mode == "none":
        return None

    y_values = np.asarray(y, dtype=int)
    weights = np.ones(len(y_values), dtype=np.float64)
    if "class" in mode:
        class_counts = {label: int((y_values == label).sum()) for label in PROJECT_LABELS}
        total = float(len(y_values))
        class_weights = {
            label: total / (len(PROJECT_LABELS) * count)
            for label, count in class_counts.items()
            if count > 0
        }
        weights *= np.asarray([class_weights[int(label)] for label in y_values], dtype=np.float64)

    if "patient" in mode:
        patient_counts = metadata["patient_id"].astype(str).value_counts().to_dict()
        weights *= np.asarray(
            [1.0 / float(patient_counts[str(patient_id)]) for patient_id in metadata["patient_id"]],
            dtype=np.float64,
        )

    mean = float(weights.mean())
    if mean <= 0 or not np.isfinite(weights).all():
        raise XGBoostGLCMV2Error("computed sample weights are invalid")
    return weights / mean


def fit_preprocessor_for(
    matrices: FeatureMatrices,
    *,
    include_validation: bool,
    config: dict[str, Any],
):
    X_fit = matrices.X_train if not include_validation else np.vstack([matrices.X_train, matrices.X_val])
    return fit_feature_preprocessor(
        X_fit,
        feature_columns=matrices.feature_columns,
        imputation=config["preprocessing"]["imputation"],
        scaling=config["preprocessing"]["scaling"],
    )


def evaluate_candidate(
    *,
    candidate_id: int,
    parameters: dict[str, Any],
    config: dict[str, Any],
    matrices: FeatureMatrices,
) -> dict[str, Any]:
    """Train one weighted candidate on TRAIN only and evaluate on VALIDATION."""

    weighting_mode = parameters["weighting_mode"]
    model_parameters = {key: value for key, value in parameters.items() if key != "weighting_mode"}
    preprocessor = fit_preprocessor_for(matrices, include_validation=False, config=config)
    X_train = preprocessor.pipeline.transform(matrices.X_train)
    X_val = preprocessor.pipeline.transform(matrices.X_val)
    sample_weight = compute_sample_weights(
        matrices.train_metadata,
        matrices.y_train,
        mode=weighting_mode,
    )
    estimator = create_xgboost_estimator(config, model_parameters)
    estimator.fit(X_train, matrices.y_train - 1, sample_weight=sample_weight)
    y_pred = predict_project_labels(estimator, X_val)
    y_score = predict_project_proba(estimator, X_val)
    result = build_result_dict(
        experiment_name="glcm_xgboost_v2_candidate_validation",
        feature_set=config["feature_set"],
        model_name=config["model_name"],
        split_evaluated="val",
        feature_count=len(matrices.feature_columns),
        metadata=matrices.val_metadata,
        y_true=matrices.y_val,
        y_pred=y_pred,
        y_score=y_score,
        model_parameters=json_safe(estimator.get_params(deep=True)),
        preprocessing={**config["preprocessing"], "sample_weighting": weighting_mode},
        random_seed=config["random_seed"],
        split_version=config["split_version"],
        feature_version=config["feature_version"],
        code_commit=get_code_commit(),
    )
    result["metrics"]["log_loss"] = float(log_loss(matrices.y_val, y_score, labels=PROJECT_LABELS))
    return {
        "candidate_id": candidate_id,
        **parameters,
        "validation_accuracy": result["metrics"]["accuracy"],
        "validation_balanced_accuracy": result["metrics"]["balanced_accuracy"],
        "validation_macro_f1": result["metrics"]["macro_f1"],
        "validation_weighted_f1": result["metrics"]["weighted_f1"],
        "validation_meningioma_f1": result["per_class_metrics"]["meningioma"]["f1"],
        "validation_log_loss": result["metrics"]["log_loss"],
        "result": result,
        "predictions": build_prediction_table(matrices.val_metadata, matrices.y_val, y_pred, y_score=y_score),
    }


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        candidates,
        key=lambda row: (
            -row["validation_macro_f1"],
            -row["validation_balanced_accuracy"],
            -row["validation_meningioma_f1"],
            -row["validation_accuracy"],
            row["validation_log_loss"],
            row["max_depth"],
            row["n_estimators"],
            row["learning_rate"],
            WEIGHTING_MODES.index(row["weighting_mode"]),
            row["candidate_id"],
        ),
    )[0]


def load_v1_validation_baseline(
    path: Path | str = "reports/experiments/glcm_xgboost_v1/validation_metrics.json",
) -> dict[str, float]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "accuracy": float(payload["metrics"]["accuracy"]),
        "balanced_accuracy": float(payload["metrics"]["balanced_accuracy"]),
        "macro_f1": float(payload["metrics"]["macro_f1"]),
        "weighted_f1": float(payload["metrics"]["weighted_f1"]),
        "meningioma_f1": float(payload["per_class_metrics"]["meningioma"]["f1"]),
    }


def decide_promotion(
    *,
    v1: dict[str, float],
    v2: dict[str, float],
    minimum_macro_f1_gain: float = 0.01,
) -> tuple[str, str]:
    macro_gain = v2["macro_f1"] - v1["macro_f1"]
    if macro_gain >= minimum_macro_f1_gain:
        return "PROMOTE", f"validation macro_f1 improved by {macro_gain:.6f}"
    if (
        macro_gain > 0
        and v2["balanced_accuracy"] > v1["balanced_accuracy"]
        and v2["meningioma_f1"] > v1["meningioma_f1"]
    ):
        return "PROMOTE", (
            "macro_f1 improved slightly and both balanced_accuracy and validation "
            "meningioma_f1 improved"
        )
    return "KEEP_V1", f"validation macro_f1 gain {macro_gain:.6f} did not meet promotion rule"


def validate_glcm_v2_inputs(
    features: Path | str,
    feature_metadata: Path | str,
    *,
    strict_real_input: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    canonical_split = "data/splits/patient_split.csv" if strict_real_input else None
    frame = read_feature_table(features, canonical_split=canonical_split)
    metadata = read_feature_metadata(feature_metadata)
    feature_columns = [column for column in frame.columns if column not in ("sample_id", "patient_id", "label", "split")]
    if strict_real_input and len(frame) != 3064:
        raise XGBoostGLCMV2Error(f"expected 3064 GLCM v2 rows, got {len(frame)}")
    if len(feature_columns) != 84:
        raise XGBoostGLCMV2Error(f"expected 84 GLCM v2 features, got {len(feature_columns)}")
    if metadata["feature_columns"] != feature_columns:
        raise XGBoostGLCMV2Error("GLCM v2 feature metadata columns do not match feature CSV")
    values = frame[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise XGBoostGLCMV2Error("GLCM v2 feature table contains NaN or infinite values")
    return frame, metadata


def _validation_metric_summary(result: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(result["metrics"]["accuracy"]),
        "balanced_accuracy": float(result["metrics"]["balanced_accuracy"]),
        "macro_f1": float(result["metrics"]["macro_f1"]),
        "weighted_f1": float(result["metrics"]["weighted_f1"]),
        "meningioma_f1": float(result["per_class_metrics"]["meningioma"]["f1"]),
    }


def _retrain_and_test(
    *,
    selected: dict[str, Any],
    config: dict[str, Any],
    matrices: FeatureMatrices,
    output_root: Path,
    model_root: Path,
    experiment_name: str,
) -> dict[str, Any]:
    final_preprocessor = fit_preprocessor_for(matrices, include_validation=True, config=config)
    X_train_val_raw = np.vstack([matrices.X_train, matrices.X_val])
    y_train_val = np.concatenate([matrices.y_train, matrices.y_val])
    metadata_train_val = pd.concat([matrices.train_metadata, matrices.val_metadata], ignore_index=True)
    X_train_val = final_preprocessor.pipeline.transform(X_train_val_raw)
    X_test = final_preprocessor.pipeline.transform(matrices.X_test)
    weighting_mode = selected["weighting_mode"]
    sample_weight = compute_sample_weights(metadata_train_val, y_train_val, mode=weighting_mode)
    selected_parameters = {
        key: selected[key]
        for key in (
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "tree_method",
            "verbosity",
            "n_jobs",
            "n_estimators",
            "max_depth",
            "learning_rate",
        )
    }
    final_estimator = create_xgboost_estimator(config, selected_parameters)
    final_estimator.fit(X_train_val, y_train_val - 1, sample_weight=sample_weight)
    y_pred = predict_project_labels(final_estimator, X_test)
    y_score = predict_project_proba(final_estimator, X_test)
    result = build_result_dict(
        experiment_name=experiment_name,
        feature_set=config["feature_set"],
        model_name=config["model_name"],
        split_evaluated="test",
        feature_count=len(matrices.feature_columns),
        metadata=matrices.test_metadata,
        y_true=matrices.y_test,
        y_pred=y_pred,
        y_score=y_score,
        model_parameters=json_safe(final_estimator.get_params(deep=True)),
        preprocessing={
            **config["preprocessing"],
            "fit_scope": "train_plus_validation",
            "sample_weighting": weighting_mode,
        },
        random_seed=config["random_seed"],
        split_version=config["split_version"],
        feature_version=config["feature_version"],
        code_commit=get_code_commit(),
    )
    result["metrics"]["log_loss"] = float(log_loss(matrices.y_test, y_score, labels=PROJECT_LABELS))
    predictions = build_prediction_table(matrices.test_metadata, matrices.y_test, y_pred, y_score=y_score)
    write_split_outputs(output_dir=output_root, prefix="test", result=result, predictions=predictions)
    importance = pd.DataFrame(
        {"feature": matrices.feature_columns, "importance": final_estimator.feature_importances_}
    ).sort_values("importance", ascending=False, kind="mergesort")
    importance.to_csv(output_root / "feature_importance.csv", index=False)
    artifacts = save_xgboost_artifacts(
        estimator=final_estimator,
        preprocessor=final_preprocessor,
        output_dir=model_root,
    )
    joblib.load(artifacts["bundle"])
    return {
        "test_result": result,
        "test_predictions": predictions,
        "artifact_paths": {key: str(path) for key, path in artifacts.items()},
        "feature_importance_top10": importance.head(10).to_dict(orient="records"),
    }


def run_experiment(
    *,
    features: Path | str = "data/features/glcm_v2/glcm_v2_features.csv",
    feature_metadata: Path | str = "data/features/glcm_v2/glcm_v2_metadata.json",
    config_path: Path | str = "configs/models/xgboost_glcm_v2.yaml",
    v1_validation_metrics: Path | str = "reports/experiments/glcm_xgboost_v1/validation_metrics.json",
    experiment_name: str = "glcm_xgboost_v2",
    output_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    strict_real_input: bool = True,
) -> dict[str, Any]:
    start = time.perf_counter()
    config = load_xgboost_glcm_v2_config(config_path)
    frame, feature_meta = validate_glcm_v2_inputs(
        features,
        feature_metadata,
        strict_real_input=strict_real_input,
    )
    canonical_split = "data/splits/patient_split.csv" if strict_real_input else None
    matrices = prepare_feature_matrices(frame, canonical_split=canonical_split)
    if len(matrices.feature_columns) != 84:
        raise XGBoostGLCMV2Error("feature order/count for GLCM v2 must be 84")
    output_root = Path(output_dir) if output_dir is not None else Path("reports/experiments") / experiment_name
    model_root = Path(model_dir) if model_dir is not None else Path("models") / experiment_name
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = []
    grid = candidate_parameter_grid(config)
    for candidate_id, parameters in enumerate(grid, start=1):
        row = evaluate_candidate(
            candidate_id=candidate_id,
            parameters=parameters,
            config=config,
            matrices=matrices,
        )
        candidates.append(row)
        print(
            f"candidate {candidate_id}/{len(grid)} weighting={row['weighting_mode']} "
            f"macro_f1={row['validation_macro_f1']:.4f} "
            f"meningioma_f1={row['validation_meningioma_f1']:.4f}"
        )

    selected = select_best_candidate(candidates)
    candidate_table = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key not in {"result", "predictions"}}
            | {"selected": row["candidate_id"] == selected["candidate_id"]}
            for row in candidates
        ]
    )
    candidate_table.to_csv(output_root / "validation_candidates.csv", index=False)
    write_split_outputs(
        output_dir=output_root,
        prefix="validation",
        result=selected["result"],
        predictions=selected["predictions"],
    )

    v1 = load_v1_validation_baseline(v1_validation_metrics)
    v2 = _validation_metric_summary(selected["result"])
    status, reason = decide_promotion(
        v1=v1,
        v2=v2,
        minimum_macro_f1_gain=float(config["promotion_policy"]["minimum_macro_f1_gain"]),
    )
    comparison = {
        "promotion_status": status,
        "promotion_reason": reason,
        "policy": config["promotion_policy"],
        "v1_validation": v1,
        "v2_best_validation": v2,
        "differences": {key: float(v2[key] - v1[key]) for key in v1},
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected_weighting_mode": selected["weighting_mode"],
        "selected_hyperparameters": {
            "n_estimators": int(selected["n_estimators"]),
            "max_depth": int(selected["max_depth"]),
            "learning_rate": float(selected["learning_rate"]),
            "subsample": float(selected["subsample"]),
            "colsample_bytree": float(selected["colsample_bytree"]),
        },
    }
    write_json(comparison, output_root / "promotion_comparison.json")

    summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "feature_set": config["feature_set"],
        "feature_version": config["feature_version"],
        "feature_metadata_commit": feature_meta["code_commit"],
        "model_name": config["model_name"],
        "model_version": config["model_version"],
        "xgboost_version": xgboost.__version__,
        "code_commit": get_code_commit(),
        "split_version": config["split_version"],
        "split_seed": config["split_seed"],
        "random_seed": config["random_seed"],
        "candidate_count": len(candidates),
        "selection_policy": config["selection_policy"],
        "promotion_comparison": comparison,
        "test_evaluation_run": False,
        "elapsed_seconds": None,
    }

    if status == "PROMOTE":
        test_payload = _retrain_and_test(
            selected=selected,
            config=config,
            matrices=matrices,
            output_root=output_root,
            model_root=model_root,
            experiment_name=experiment_name,
        )
        summary.update(
            {
                "test_evaluation_run": True,
                "final_retraining_policy": "train_plus_validation",
                "artifact_paths": test_payload["artifact_paths"],
                "test_metrics": {
                    **test_payload["test_result"]["metrics"],
                    "roc_auc": test_payload["test_result"]["roc_auc"],
                },
                "feature_importance_top10": test_payload["feature_importance_top10"],
            }
        )

    summary["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    write_json(summary, output_root / "experiment_summary.json")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/features/glcm_v2/glcm_v2_features.csv"))
    parser.add_argument(
        "--feature-metadata",
        type=Path,
        default=Path("data/features/glcm_v2/glcm_v2_metadata.json"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/models/xgboost_glcm_v2.yaml"))
    parser.add_argument(
        "--v1-validation-metrics",
        type=Path,
        default=Path("reports/experiments/glcm_xgboost_v1/validation_metrics.json"),
    )
    parser.add_argument("--experiment-name", default="glcm_xgboost_v2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    run_experiment(
        features=args.features,
        feature_metadata=args.feature_metadata,
        config_path=args.config,
        v1_validation_metrics=args.v1_validation_metrics,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
