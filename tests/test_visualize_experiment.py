from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.visualize_experiment import (
    VisualizationError,
    compute_roc_curves,
    generate_experiment_visualizations,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_per_class_metrics,
    plot_prediction_confidence,
    plot_roc_curves,
    plot_top_feature_distributions,
    plot_validation_vs_test,
    prediction_confidence,
    row_normalize_confusion_matrix,
)


FEATURES = [f"glcm_feature_{index:02d}" for index in range(12)]


def _metrics() -> dict:
    return {
        "metrics": {
            "accuracy": 0.6,
            "balanced_accuracy": 0.55,
            "macro_precision": 0.5,
            "macro_recall": 0.55,
            "macro_f1": 0.52,
            "weighted_precision": 0.58,
            "weighted_recall": 0.6,
            "weighted_f1": 0.57,
        },
        "per_class_metrics": {
            "meningioma": {"precision": 0.3, "recall": 0.2, "f1": 0.24, "support": 2},
            "glioma": {"precision": 0.7, "recall": 0.8, "f1": 0.74, "support": 2},
            "pituitary": {"precision": 0.6, "recall": 0.7, "f1": 0.64, "support": 2},
        },
        "confusion_matrix": [[1, 1, 0], [0, 2, 0], [1, 0, 1]],
        "roc_auc": {"macro_ovr": 0.7, "weighted_ovr": 0.72, "available": True, "reason": ""},
    }


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [str(i) for i in range(1, 7)],
            "patient_id": [f"p{i}" for i in range(1, 7)],
            "true_label": [1, 1, 2, 2, 3, 3],
            "predicted_label": [1, 2, 2, 2, 1, 3],
            "prob_meningioma": [0.8, 0.3, 0.1, 0.1, 0.5, 0.1],
            "prob_glioma": [0.1, 0.5, 0.8, 0.7, 0.2, 0.2],
            "prob_pituitary": [0.1, 0.2, 0.1, 0.2, 0.3, 0.7],
        }
    )


def _features() -> pd.DataFrame:
    rows = []
    for index in range(30):
        label = index % 3 + 1
        row = {"sample_id": str(index + 1), "patient_id": f"p{index + 1}", "label": label, "split": "test"}
        for feature_index, feature in enumerate(FEATURES):
            row[feature] = label + feature_index * 0.1 + index * 0.01
        rows.append(row)
    return pd.DataFrame(rows)


def _importance() -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": FEATURES, "importance": np.linspace(1.0, 0.1, len(FEATURES))}
    )


def _write_experiment(root, features_path) -> None:
    metrics = _metrics()
    for name in ["test_metrics.json", "validation_metrics.json"]:
        (root / name).write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame(metrics["confusion_matrix"], columns=["meningioma", "glioma", "pituitary"]).assign(
        true_label=["meningioma", "glioma", "pituitary"]
    )[["true_label", "meningioma", "glioma", "pituitary"]].to_csv(root / "test_confusion_matrix.csv", index=False)
    _predictions().to_csv(root / "test_predictions.csv", index=False)
    _importance().to_csv(root / "feature_importance.csv", index=False)
    pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "validation_macro_f1": [0.5, 0.6],
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.05],
            "selected": [False, True],
        }
    ).to_csv(root / "validation_candidates.csv", index=False)
    _features().to_csv(features_path, index=False)


def assert_nonempty(path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0


def test_normalized_confusion_matrix_calculation() -> None:
    normalized = row_normalize_confusion_matrix(np.array([[1, 1], [0, 2]]))
    assert normalized.tolist() == [[50.0, 50.0], [0.0, 100.0]]


def test_individual_plots_created_and_nonempty(tmp_path) -> None:
    metrics = _metrics()
    predictions = _predictions()
    importance = _importance()
    features = _features()
    assert_nonempty(plot_confusion_matrix(np.array(metrics["confusion_matrix"]), tmp_path / "cm.png"))
    assert_nonempty(plot_per_class_metrics(metrics, tmp_path / "per_class.png"))
    assert_nonempty(plot_validation_vs_test(metrics, metrics, tmp_path / "compare.png"))
    assert_nonempty(plot_feature_importance(importance, tmp_path / "importance.png"))
    assert_nonempty(plot_top_feature_distributions(features, importance, tmp_path / "dist.png"))
    roc_path, roc = plot_roc_curves(predictions, tmp_path / "roc.png")
    assert_nonempty(roc_path)
    assert set(roc) == {"Meningioma", "Glioma", "Pituitary", "Macro average"}
    confidence_path, stats = plot_prediction_confidence(predictions, tmp_path / "confidence.png")
    assert_nonempty(confidence_path)
    assert stats["correct_median"] > 0


def test_roc_class_mapping_and_confidence_values() -> None:
    predictions = _predictions()
    curves = compute_roc_curves(predictions)
    assert curves["Meningioma"]["auc"] > 0.5
    confidence = prediction_confidence(predictions)
    assert confidence["confidence"].tolist() == pytest.approx([0.8, 0.5, 0.8, 0.7, 0.5, 0.7])
    assert confidence["correct"].tolist() == [True, False, True, True, False, True]


def test_missing_probability_columns_handled_clearly() -> None:
    with pytest.raises(VisualizationError, match="missing probability"):
        prediction_confidence(_predictions().drop(columns=["prob_pituitary"]))


def test_generation_creates_output_dir_manifest_and_preserves_sources(tmp_path) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    features_path = tmp_path / "features.csv"
    _write_experiment(experiment_dir, features_path)
    source_predictions = pd.read_csv(experiment_dir / "test_predictions.csv")
    source_features = pd.read_csv(features_path)

    manifest = generate_experiment_visualizations(
        experiment_dir=experiment_dir,
        features=features_path,
        output_dir=experiment_dir / "figures",
        dpi=80,
    )

    figure_dir = experiment_dir / "figures"
    assert figure_dir.is_dir()
    assert len(manifest["figures"]) == 8
    manifest_json = json.loads((figure_dir / "figure_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest_json["figures"]) == 8
    for entry in manifest_json["figures"]:
        assert_nonempty(figure_dir / entry["filename"])
    pd.testing.assert_frame_equal(source_predictions, pd.read_csv(experiment_dir / "test_predictions.csv"))
    pd.testing.assert_frame_equal(source_features, pd.read_csv(features_path))
