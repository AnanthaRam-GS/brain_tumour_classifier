"""Run the Review 1 GLCM + XGBoost experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import log_loss

from src.evaluation.results import build_prediction_table, build_result_dict, write_experiment_results
from src.features.feature_table import read_feature_metadata, read_feature_table
from src.models.training import FeatureMatrices, fit_feature_preprocessor, prepare_feature_matrices
from src.models.xgboost_model import (
    PROJECT_LABELS,
    candidate_parameter_grid,
    create_xgboost_estimator,
    fit_xgboost_estimator,
    load_xgboost_bundle,
    load_xgboost_config,
    predict_project_labels,
    predict_project_proba,
    save_xgboost_artifacts,
    write_json,
)


def get_code_commit() -> str:
    """Return current Git commit hash, or unknown outside Git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def json_safe(value: Any) -> Any:
    """Convert estimator parameters into strict JSON-compatible values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def validate_real_glcm_inputs(
    feature_path: Path | str,
    feature_metadata_path: Path | str,
    *,
    strict_real_input: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the real GLCM feature CSV and metadata before model training."""

    canonical_split = "data/splits/patient_split.csv" if strict_real_input else None
    frame = read_feature_table(feature_path, canonical_split=canonical_split)
    metadata = read_feature_metadata(feature_metadata_path)
    feature_columns = [
        column
        for column in frame.columns
        if column not in ("sample_id", "patient_id", "label", "split")
    ]
    if strict_real_input and len(frame) != 3064:
        raise ValueError(f"expected 3064 GLCM rows, got {len(frame)}")
    if strict_real_input and len(feature_columns) != 12:
        raise ValueError(f"expected 12 GLCM features, got {len(feature_columns)}")
    if metadata["feature_columns"] != feature_columns:
        raise ValueError("GLCM feature metadata columns do not match feature CSV")
    if strict_real_input and not str(metadata["code_commit"]).startswith("ad1415b3"):
        raise ValueError(f"unexpected GLCM feature commit: {metadata['code_commit']}")
    values = frame[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("GLCM feature table contains NaN or infinite values")
    return frame, metadata


def fit_preprocessor_for(matrices: FeatureMatrices, *, include_validation: bool, config: dict[str, Any]):
    """Fit median-only preprocessing on train or train+validation."""

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
    """Fit one candidate on train only and evaluate it on validation."""

    preprocessor = fit_preprocessor_for(matrices, include_validation=False, config=config)
    X_train = preprocessor.pipeline.transform(matrices.X_train)
    X_val = preprocessor.pipeline.transform(matrices.X_val)
    estimator = create_xgboost_estimator(config, parameters)
    fit_xgboost_estimator(estimator, X_train, matrices.y_train)
    y_pred = predict_project_labels(estimator, X_val)
    y_score = predict_project_proba(estimator, X_val)
    result = build_result_dict(
        experiment_name="candidate_validation",
        feature_set=config["feature_set"],
        model_name=config["model_name"],
        split_evaluated="val",
        feature_count=len(matrices.feature_columns),
        metadata=matrices.val_metadata,
        y_true=matrices.y_val,
        y_pred=y_pred,
        y_score=y_score,
        model_parameters=json_safe(estimator.get_params(deep=True)),
        preprocessing=config["preprocessing"],
        random_seed=config["random_seed"],
        split_version=config["split_version"],
        feature_version=config["feature_version"],
        code_commit=get_code_commit(),
    )
    return {
        "candidate_id": candidate_id,
        **parameters,
        "validation_accuracy": result["metrics"]["accuracy"],
        "validation_balanced_accuracy": result["metrics"]["balanced_accuracy"],
        "validation_macro_f1": result["metrics"]["macro_f1"],
        "validation_weighted_f1": result["metrics"]["weighted_f1"],
        "validation_log_loss": float(log_loss(matrices.y_val, y_score, labels=PROJECT_LABELS)),
        "result": result,
        "predictions": build_prediction_table(matrices.val_metadata, matrices.y_val, y_pred, y_score=y_score),
    }


def select_best_candidate(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select by macro F1, then deterministic frozen tie-breakers."""

    return sorted(
        candidate_rows,
        key=lambda row: (
            -row["validation_macro_f1"],
            -row["validation_balanced_accuracy"],
            -row["validation_accuracy"],
            row["validation_log_loss"],
            row["max_depth"],
            row["n_estimators"],
            row["learning_rate"],
            row["candidate_id"],
        ),
    )[0]


def write_split_outputs(
    *,
    output_dir: Path,
    prefix: str,
    result: dict[str, Any],
    predictions: pd.DataFrame,
) -> dict[str, Path]:
    """Write shared result files with validation/test prefixes."""

    paths = write_experiment_results(result, predictions, output_dir)
    renamed = {}
    for key, path in paths.items():
        target_name = {
            "metrics": f"{prefix}_metrics.json",
            "predictions": f"{prefix}_predictions.csv",
            "confusion_matrix": f"{prefix}_confusion_matrix.csv",
            "experiment_metadata": f"{prefix}_metadata.json",
        }[key]
        target = output_dir / target_name
        path.replace(target)
        renamed[key] = target
    return renamed


def run_experiment(
    *,
    features: Path | str = "data/features/glcm/glcm_features.csv",
    feature_metadata: Path | str = "data/features/glcm/glcm_metadata.json",
    config_path: Path | str = "configs/models/xgboost.yaml",
    experiment_name: str = "glcm_xgboost_v1",
    output_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    strict_real_input: bool = True,
) -> dict[str, Any]:
    """Run candidate selection, final retraining, test evaluation, and artifact writing."""

    start = time.perf_counter()
    config = load_xgboost_config(config_path)
    frame, feature_meta = validate_real_glcm_inputs(
        features,
        feature_metadata,
        strict_real_input=strict_real_input,
    )
    canonical_split = "data/splits/patient_split.csv" if strict_real_input else None
    matrices = prepare_feature_matrices(frame, canonical_split=canonical_split)
    output_root = Path(output_dir) if output_dir is not None else Path("reports/experiments") / experiment_name
    model_root = Path(model_dir) if model_dir is not None else Path("models") / experiment_name
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = []
    for candidate_id, parameters in enumerate(candidate_parameter_grid(config), start=1):
        row = evaluate_candidate(
            candidate_id=candidate_id,
            parameters=parameters,
            config=config,
            matrices=matrices,
        )
        candidates.append(row)
        print(
            f"candidate {candidate_id}/8 macro_f1={row['validation_macro_f1']:.4f} "
            f"balanced_accuracy={row['validation_balanced_accuracy']:.4f}"
        )

    selected = select_best_candidate(candidates)
    candidate_table = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"result", "predictions"}
            }
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

    final_preprocessor = fit_preprocessor_for(matrices, include_validation=True, config=config)
    X_train_val = final_preprocessor.pipeline.transform(np.vstack([matrices.X_train, matrices.X_val]))
    y_train_val = np.concatenate([matrices.y_train, matrices.y_val])
    X_test = final_preprocessor.pipeline.transform(matrices.X_test)
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
    fit_xgboost_estimator(final_estimator, X_train_val, y_train_val)
    test_pred = predict_project_labels(final_estimator, X_test)
    test_score = predict_project_proba(final_estimator, X_test)
    test_result = build_result_dict(
        experiment_name=experiment_name,
        feature_set=config["feature_set"],
        model_name=config["model_name"],
        split_evaluated="test",
        feature_count=len(matrices.feature_columns),
        metadata=matrices.test_metadata,
        y_true=matrices.y_test,
        y_pred=test_pred,
        y_score=test_score,
        model_parameters=json_safe(final_estimator.get_params(deep=True)),
        preprocessing={**config["preprocessing"], "fit_scope": "train_plus_validation"},
        random_seed=config["random_seed"],
        split_version=config["split_version"],
        feature_version=config["feature_version"],
        code_commit=get_code_commit(),
    )
    test_log_loss = float(log_loss(matrices.y_test, test_score, labels=PROJECT_LABELS))
    test_result["metrics"]["log_loss"] = test_log_loss
    test_predictions = build_prediction_table(matrices.test_metadata, matrices.y_test, test_pred, y_score=test_score)
    write_split_outputs(output_dir=output_root, prefix="test", result=test_result, predictions=test_predictions)

    importance = pd.DataFrame(
        {"feature": matrices.feature_columns, "importance": final_estimator.feature_importances_}
    ).sort_values("importance", ascending=False, kind="mergesort")
    importance.to_csv(output_root / "feature_importance.csv", index=False)

    artifacts = save_xgboost_artifacts(
        estimator=final_estimator,
        preprocessor=final_preprocessor,
        output_dir=model_root,
    )
    bundle = load_xgboost_bundle(artifacts["bundle"])
    reloaded_pred = predict_project_labels(
        bundle["estimator"],
        bundle["preprocessor"].pipeline.transform(matrices.X_test),
    )
    reload_matches = bool(np.array_equal(reloaded_pred, test_pred))

    validation_result = selected["result"]
    summary = {
        "experiment_name": experiment_name,
        "feature_set": config["feature_set"],
        "feature_version": config["feature_version"],
        "feature_implementation_commit": config["feature_implementation_commit"],
        "feature_metadata_commit": feature_meta["code_commit"],
        "model_name": config["model_name"],
        "model_version": config["model_version"],
        "xgboost_version": xgboost.__version__,
        "code_commit": get_code_commit(),
        "split_version": config["split_version"],
        "split_seed": config["split_seed"],
        "random_seed": config["random_seed"],
        "candidate_count": len(candidates),
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected_hyperparameters": selected_parameters,
        "selection_policy": config["selection_policy"],
        "final_retraining_policy": "train_plus_validation",
        "train_samples": int(len(matrices.y_train)),
        "val_samples": int(len(matrices.y_val)),
        "test_samples": int(len(matrices.y_test)),
        "train_patients": int(matrices.train_metadata["patient_id"].nunique()),
        "val_patients": int(matrices.val_metadata["patient_id"].nunique()),
        "test_patients": int(matrices.test_metadata["patient_id"].nunique()),
        "feature_count": len(matrices.feature_columns),
        "validation_metrics": {
            **validation_result["metrics"],
            "roc_auc": validation_result["roc_auc"],
            "log_loss": float(selected["validation_log_loss"]),
        },
        "test_metrics": {**test_result["metrics"], "roc_auc": test_result["roc_auc"]},
        "artifact_paths": {key: str(path) for key, path in artifacts.items()},
        "reload_reproduces_test_predictions": reload_matches,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    write_json(summary, output_root / "experiment_summary.json")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/features/glcm/glcm_features.csv"))
    parser.add_argument(
        "--feature-metadata",
        type=Path,
        default=Path("data/features/glcm/glcm_metadata.json"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/models/xgboost.yaml"))
    parser.add_argument("--experiment-name", default="glcm_xgboost_v1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    run_experiment(
        features=args.features,
        feature_metadata=args.feature_metadata,
        config_path=args.config,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
