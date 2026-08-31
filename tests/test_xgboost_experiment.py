from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.models.run_xgboost import run_experiment, select_best_candidate
from src.models.xgboost_model import load_xgboost_bundle

FEATURE_COLUMNS = [f"glcm_f_{index:02d}" for index in range(12)]


def _tiny_feature_table() -> pd.DataFrame:
    rows = []
    sample_id = 1
    for split, count in [("train", 18), ("val", 9), ("test", 9)]:
        for index in range(count):
            label = index % 3 + 1
            row = {
                "sample_id": str(sample_id),
                "patient_id": f"p{sample_id}",
                "label": label,
                "split": split,
            }
            for feature_index, column in enumerate(FEATURE_COLUMNS):
                row[column] = float(label) + feature_index * 0.01 + index * 0.001
            rows.append(row)
            sample_id += 1
    return pd.DataFrame(rows)


def _write_metadata(path: Path) -> None:
    payload = {
        "feature_set_name": "glcm",
        "feature_version": "v1",
        "algorithm": "gray_level_cooccurrence_matrix",
        "parameters": {},
        "input_representation": "native_roi_image_and_mask",
        "roi_policy": "phase1_roi_contract_v1",
        "source_split_version": "phase1_patient_split_v1",
        "sample_count": 36,
        "feature_count": 12,
        "feature_columns": FEATURE_COLUMNS,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "code_commit": "ad1415b3",
        "validation_summary": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_config(path: Path) -> None:
    path.write_text(
        """
model_name: xgboost
model_version: v1
random_seed: 42
feature_set: glcm
objective: multi:softprob
num_class: 3
eval_metric: mlogloss
preprocessing:
  imputation: median
  scaling: null
fixed_parameters:
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 1
  gamma: 0
  reg_alpha: 0
  reg_lambda: 1
  tree_method: hist
  verbosity: 0
  n_jobs: 1
candidate_grid:
  n_estimators: [2, 3]
  max_depth: [1, 2]
  learning_rate: [0.1, 0.2]
selection_policy:
  primary_metric: macro_f1
  tie_breakers:
    - balanced_accuracy
    - accuracy
    - lower_log_loss
    - lower_max_depth
    - fewer_estimators
    - lower_learning_rate
final_retraining_policy: train_plus_validation
feature_implementation_commit: ad1415b3
feature_version: v1
split_version: phase1_patient_split_v1
split_seed: 42
importance_type: estimator.feature_importances_
""".strip(),
        encoding="utf-8",
    )


def test_candidate_selection_is_deterministic_and_uses_macro_f1() -> None:
    rows = [
        {
            "candidate_id": 1,
            "validation_macro_f1": 0.8,
            "validation_balanced_accuracy": 0.9,
            "validation_accuracy": 0.9,
            "validation_log_loss": 0.2,
            "max_depth": 5,
            "n_estimators": 200,
            "learning_rate": 0.1,
        },
        {
            "candidate_id": 2,
            "validation_macro_f1": 0.8,
            "validation_balanced_accuracy": 0.9,
            "validation_accuracy": 0.9,
            "validation_log_loss": 0.2,
            "max_depth": 3,
            "n_estimators": 200,
            "learning_rate": 0.1,
        },
    ]
    assert select_best_candidate(rows)["candidate_id"] == 2


def test_experiment_runner_works_on_tiny_synthetic_dataset(tmp_path) -> None:
    features = tmp_path / "features.csv"
    feature_metadata = tmp_path / "feature_metadata.json"
    config = tmp_path / "xgboost.yaml"
    output_dir = tmp_path / "reports"
    model_dir = tmp_path / "models"
    _tiny_feature_table().to_csv(features, index=False)
    _write_metadata(feature_metadata)
    _write_config(config)

    summary = run_experiment(
        features=features,
        feature_metadata=feature_metadata,
        config_path=config,
        experiment_name="tiny",
        output_dir=output_dir,
        model_dir=model_dir,
        strict_real_input=False,
    )

    candidates = pd.read_csv(output_dir / "validation_candidates.csv")
    importance = pd.read_csv(output_dir / "feature_importance.csv")
    test_predictions = pd.read_csv(output_dir / "test_predictions.csv")
    assert len(candidates) == 8
    assert int(candidates["selected"].sum()) == 1
    assert len(test_predictions) == 9
    assert set(test_predictions["predicted_label"]).issubset({1, 2, 3})
    assert len(importance) == 12
    assert (output_dir / "experiment_summary.json").is_file()
    assert (model_dir / "xgboost_model.json").is_file()
    assert (model_dir / "imputer.joblib").is_file()
    bundle = load_xgboost_bundle(model_dir / "model_bundle.joblib")
    assert "estimator" in bundle
    assert joblib.load(model_dir / "imputer.joblib") is not None
    assert summary["final_retraining_policy"] == "train_plus_validation"
    assert summary["test_samples"] == 9
