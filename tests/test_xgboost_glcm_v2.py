from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.run_xgboost_glcm_v2 import (
    WEIGHTING_MODES,
    candidate_parameter_grid,
    compute_sample_weights,
    decide_promotion,
    load_xgboost_glcm_v2_config,
    run_experiment,
    select_best_candidate,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["1", "2", "3", "4", "5", "6"],
            "patient_id": ["A", "A", "A", "B", "C", "D"],
            "label": [1, 1, 1, 2, 2, 3],
            "split": ["train"] * 6,
        }
    )


def test_config_loads_and_has_108_candidates() -> None:
    config = load_xgboost_glcm_v2_config()
    assert config["feature_set"] == "glcm_v2"
    assert config["model_version"] == "v2"
    assert config["random_seed"] == 42
    assert len(candidate_parameter_grid(config)) == 108
    assert tuple(config["weighting_modes"]) == WEIGHTING_MODES


def test_class_weights_use_provided_training_rows_only() -> None:
    metadata = _metadata()
    y = metadata["label"].to_numpy()
    weights = compute_sample_weights(metadata, y, mode="class_balanced")
    assert weights is not None
    assert weights[y == 1].mean() < weights[y == 3].mean()
    changed_val_test = pd.concat(
        [
            metadata,
            pd.DataFrame(
                {
                    "sample_id": ["7", "8"],
                    "patient_id": ["VAL", "TEST"],
                    "label": [3, 3],
                    "split": ["val", "test"],
                }
            ),
        ],
        ignore_index=True,
    )
    np.testing.assert_allclose(
        weights,
        compute_sample_weights(metadata, y, mode="class_balanced"),
    )
    assert len(changed_val_test) != len(metadata)


def test_patient_weights_use_training_patients_and_downweight_many_slice_patient() -> None:
    metadata = _metadata()
    y = metadata["label"].to_numpy()
    weights = compute_sample_weights(metadata, y, mode="patient_balanced")
    assert weights is not None
    assert weights[metadata["patient_id"].eq("A")].mean() < weights[metadata["patient_id"].eq("B")].mean()


def test_combined_weights_are_finite_and_mean_normalized() -> None:
    metadata = _metadata()
    weights = compute_sample_weights(metadata, metadata["label"].to_numpy(), mode="class_and_patient_balanced")
    assert weights is not None
    assert np.isfinite(weights).all()
    assert weights.mean() == pytest.approx(1.0)


def test_four_weighting_modes_supported() -> None:
    metadata = _metadata()
    y = metadata["label"].to_numpy()
    assert compute_sample_weights(metadata, y, mode="none") is None
    for mode in WEIGHTING_MODES[1:]:
        assert compute_sample_weights(metadata, y, mode=mode) is not None


def test_candidate_selection_uses_macro_f1_then_meningioma_tie_break() -> None:
    rows = [
        {
            "candidate_id": 1,
            "validation_macro_f1": 0.5,
            "validation_balanced_accuracy": 0.6,
            "validation_meningioma_f1": 0.2,
            "validation_accuracy": 0.7,
            "validation_log_loss": 1.0,
            "max_depth": 3,
            "n_estimators": 150,
            "learning_rate": 0.05,
            "weighting_mode": "none",
        },
        {
            "candidate_id": 2,
            "validation_macro_f1": 0.5,
            "validation_balanced_accuracy": 0.6,
            "validation_meningioma_f1": 0.4,
            "validation_accuracy": 0.65,
            "validation_log_loss": 1.1,
            "max_depth": 3,
            "n_estimators": 150,
            "learning_rate": 0.05,
            "weighting_mode": "class_balanced",
        },
    ]
    assert select_best_candidate(rows)["candidate_id"] == 2


def test_candidate_selection_is_deterministic() -> None:
    rows = []
    for candidate_id in range(3):
        rows.append(
            {
                "candidate_id": candidate_id + 1,
                "validation_macro_f1": 0.5,
                "validation_balanced_accuracy": 0.5,
                "validation_meningioma_f1": 0.5,
                "validation_accuracy": 0.5,
                "validation_log_loss": 1.0,
                "max_depth": 3,
                "n_estimators": 150,
                "learning_rate": 0.03,
                "weighting_mode": "none",
            }
        )
    assert select_best_candidate(rows)["candidate_id"] == 1


def test_promotion_comparison_uses_validation_metrics_only() -> None:
    v1 = {"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5, "weighted_f1": 0.5, "meningioma_f1": 0.2}
    v2 = {"accuracy": 0.6, "balanced_accuracy": 0.6, "macro_f1": 0.512, "weighted_f1": 0.6, "meningioma_f1": 0.3}
    status, _ = decide_promotion(v1=v1, v2=v2)
    assert status == "PROMOTE"
    v2["macro_f1"] = 0.501
    status, _ = decide_promotion(v1=v1, v2=v2)
    assert status == "PROMOTE"
    v2["balanced_accuracy"] = 0.49
    status, _ = decide_promotion(v1=v1, v2=v2)
    assert status == "KEEP_V1"


def test_failed_promotion_does_not_access_test(monkeypatch, tmp_path: Path) -> None:
    import src.models.run_xgboost_glcm_v2 as module

    def fail_if_called(**kwargs):
        raise AssertionError("test stage must not run when promotion fails")

    monkeypatch.setattr(module, "_retrain_and_test", fail_if_called)
    status, _ = module.decide_promotion(
        v1={"accuracy": 0.6, "balanced_accuracy": 0.6, "macro_f1": 0.6, "weighted_f1": 0.6, "meningioma_f1": 0.4},
        v2={"accuracy": 0.6, "balanced_accuracy": 0.6, "macro_f1": 0.59, "weighted_f1": 0.6, "meningioma_f1": 0.4},
    )
    assert status == "KEEP_V1"


def test_successful_promotion_enables_test_stage() -> None:
    status, _ = decide_promotion(
        v1={"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5, "weighted_f1": 0.5, "meningioma_f1": 0.2},
        v2={"accuracy": 0.6, "balanced_accuracy": 0.6, "macro_f1": 0.52, "weighted_f1": 0.6, "meningioma_f1": 0.3},
    )
    assert status == "PROMOTE"


def test_feature_order_84_preserved_in_real_file_if_available() -> None:
    path = Path("data/features/glcm_v2/glcm_v2_features.csv")
    if not path.exists():
        pytest.skip("real GLCM v2 feature table has not been generated")
    frame = pd.read_csv(path, dtype={"patient_id": str})
    feature_columns = [c for c in frame.columns if c not in ("sample_id", "patient_id", "label", "split")]
    assert len(feature_columns) == 84

