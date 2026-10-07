from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.visualize_glcm_v2 import (
    aggregate_importance_by_angle,
    aggregate_importance_by_distance,
    aggregate_importance_by_property,
    best_macro_f1_by_weighting,
    generate_glcm_v2_visualizations,
    prediction_confidence_stats,
    roc_auc_by_class,
    row_normalize,
)


def _metrics(scale: float = 1.0) -> dict:
    return {
        "metrics": {
            "accuracy": 0.6 * scale,
            "balanced_accuracy": 0.55 * scale,
            "macro_f1": 0.5 * scale,
            "weighted_f1": 0.58 * scale,
        },
        "per_class_metrics": {
            "meningioma": {"precision": 0.4 * scale, "recall": 0.3 * scale, "f1": 0.34 * scale, "support": 2},
            "glioma": {"precision": 0.7 * scale, "recall": 0.8 * scale, "f1": 0.75 * scale, "support": 2},
            "pituitary": {"precision": 0.65 * scale, "recall": 0.6 * scale, "f1": 0.62 * scale, "support": 2},
        },
        "roc_auc": {"macro_ovr": 0.75 * scale, "weighted_ovr": 0.77 * scale},
    }


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "patient_id": ["A", "B", "C", "D", "E", "F"],
            "true_label": [1, 1, 2, 2, 3, 3],
            "predicted_label": [1, 2, 2, 2, 3, 1],
            "prob_meningioma": [0.8, 0.2, 0.1, 0.1, 0.05, 0.6],
            "prob_glioma": [0.1, 0.7, 0.8, 0.7, 0.15, 0.2],
            "prob_pituitary": [0.1, 0.1, 0.1, 0.2, 0.8, 0.2],
        }
    )


def _importance() -> pd.DataFrame:
    rows = []
    value = 1.0
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm"]:
        for distance in [1, 2, 4]:
            for angle in [0, 45, 90, 135]:
                rows.append({"feature": f"glcm_{prop}_d{distance}_a{angle}", "importance": value})
                value += 1.0
        for stat in ["mean", "std"]:
            rows.append({"feature": f"glcm_{prop}_{stat}", "importance": value})
            value += 1.0
    return pd.DataFrame(rows)


def _features() -> pd.DataFrame:
    importance = _importance()
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "sample_id": range(1, 13),
            "patient_id": [f"P{i}" for i in range(1, 13)],
            "label": [1, 2, 3] * 4,
            "split": ["train"] * 6 + ["val"] * 3 + ["test"] * 3,
        }
    )
    for index, feature in enumerate(importance["feature"]):
        frame[feature] = rng.normal(loc=index / 10, scale=0.1, size=len(frame))
    return frame


def _write_experiment(root: Path, scale: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "test_metrics.json").write_text(json.dumps(_metrics(scale)))
    _predictions().to_csv(root / "test_predictions.csv", index=False)
    pd.DataFrame(
        [[2, 0, 0], [0, 2, 0], [1, 0, 1]],
        columns=["Meningioma", "Glioma", "Pituitary"],
    ).assign(true_label=["Meningioma", "Glioma", "Pituitary"])[
        ["true_label", "Meningioma", "Glioma", "Pituitary"]
    ].to_csv(root / "test_confusion_matrix.csv", index=False)


def test_core_calculations() -> None:
    matrix = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    assert row_normalize(matrix)[0].sum() == 100
    importance = _importance()
    assert aggregate_importance_by_property(importance)["importance"].sum() == importance["importance"].sum()
    assert set(aggregate_importance_by_distance(importance)["distance"]) == {"d1", "d2", "d4"}
    assert set(aggregate_importance_by_angle(importance)["angle"]) == {"a0", "a45", "a90", "a135"}
    candidates = pd.DataFrame(
        {
            "weighting_mode": ["none", "none", "class_balanced"],
            "validation_macro_f1": [0.5, 0.7, 0.6],
        }
    )
    best = best_macro_f1_by_weighting(candidates)
    assert best.loc[best["weighting_mode"].eq("none"), "validation_macro_f1"].item() == 0.7


def test_roc_and_confidence_calculations() -> None:
    aucs = roc_auc_by_class(_predictions())
    assert set(aucs) == {"Meningioma", "Glioma", "Pituitary", "Macro"}
    assert all(0 <= value <= 1 for value in aucs.values())
    stats = prediction_confidence_stats(_predictions())
    assert stats["correct_median"] > 0
    assert stats["incorrect_median"] > 0


def test_generate_all_figures_and_manifest_without_modifying_sources(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    _write_experiment(v1, 1.0)
    _write_experiment(v2, 1.05)
    _importance().to_csv(v2 / "feature_importance.csv", index=False)
    candidates = pd.DataFrame(
        {
            "candidate_id": range(1, 5),
            "weighting_mode": ["none", "class_balanced", "patient_balanced", "class_and_patient_balanced"],
            "validation_macro_f1": [0.7, 0.6, 0.55, 0.58],
            "selected": [True, False, False, False],
        }
    )
    candidates.to_csv(v2 / "validation_candidates.csv", index=False)
    features = tmp_path / "features.csv"
    _features().to_csv(features, index=False)
    before = {path: path.stat().st_mtime_ns for path in [v1 / "test_metrics.json", v2 / "test_metrics.json", features]}

    summary = generate_glcm_v2_visualizations(
        v2_experiment_dir=v2,
        v1_experiment_dir=v1,
        v2_features=features,
        expected_test_count=6,
        expected_feature_rows=12,
        expected_feature_count=84,
    )

    assert summary["figure_count"] == 20
    manifest = json.loads(Path(summary["manifest_path"]).read_text())
    assert len(manifest) == 20
    for entry in manifest:
        figure = Path(summary["figure_dir"]) / entry["filename"]
        assert figure.is_file()
        assert figure.stat().st_size > 0
    after = {path: path.stat().st_mtime_ns for path in before}
    assert before == after
