import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.evaluation.metrics import CLASS_ORDER, evaluate_predictions
from src.evaluation.results import (
    EvaluationError,
    build_prediction_table,
    build_result_dict,
    read_result_json,
    scores_from_estimator,
    validate_result_dict,
    write_experiment_results,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["10", "1", "2"],
            "patient_id": ["P10", "P1", "P2"],
            "label": [3, 1, 2],
            "split": ["test", "test", "test"],
        }
    )


def test_metrics_are_correct_for_known_labels() -> None:
    y_true = np.array([1, 1, 2, 2, 3, 3])
    y_pred = np.array([1, 2, 2, 2, 3, 1])

    result = evaluate_predictions(y_true, y_pred)

    assert result["metrics"]["accuracy"] == pytest.approx(4 / 6)
    assert result["metrics"]["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert result["metrics"]["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert result["metrics"]["weighted_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_per_class_metrics_and_confusion_matrix_contract() -> None:
    result = evaluate_predictions(np.array([1, 2, 3]), np.array([1, 1, 1]))

    assert set(result["per_class_metrics"]) == {"meningioma", "glioma", "pituitary"}
    assert np.asarray(result["confusion_matrix"]).shape == (3, 3)
    assert result["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]


def test_absent_predicted_class_does_not_crash() -> None:
    result = evaluate_predictions(np.array([1, 2, 3]), np.array([1, 1, 1]))

    assert result["per_class_metrics"]["glioma"]["precision"] == 0.0
    assert result["per_class_metrics"]["pituitary"]["precision"] == 0.0


def test_prediction_table_preserves_metadata_order_and_predictions() -> None:
    table = build_prediction_table(
        _metadata(),
        np.array([3, 1, 2]),
        np.array([1, 1, 2]),
    )

    assert list(table.columns) == ["sample_id", "patient_id", "true_label", "predicted_label"]
    assert table["sample_id"].tolist() == ["1", "2", "10"]
    assert table["predicted_label"].tolist() == [1, 2, 1]


def test_probability_columns_map_estimator_classes() -> None:
    metadata = _metadata().sort_values("sample_id", key=lambda column: column.astype(int))
    scores = np.array([[0.2, 0.7, 0.1], [0.3, 0.1, 0.6], [0.8, 0.1, 0.1]])

    table = build_prediction_table(
        metadata,
        np.array([1, 2, 3]),
        np.array([2, 3, 1]),
        y_score=scores,
    )

    assert list(table.columns[-3:]) == ["prob_meningioma", "prob_glioma", "prob_pituitary"]
    assert table.loc[0, "prob_meningioma"] == pytest.approx(0.2)


def test_scores_from_estimator_uses_classes_mapping() -> None:
    estimator = DummyClassifier(strategy="prior")
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([3, 1, 2, 3, 1, 2])
    estimator.fit(X, y)

    scores, reason = scores_from_estimator(estimator, X[:2])

    assert reason == ""
    assert scores is not None
    assert scores.shape == (2, 3)
    assert list(estimator.classes_) == [1, 2, 3]


def test_estimator_without_probabilities_still_works() -> None:
    class NoProbabilityEstimator:
        classes_ = np.array([1, 2, 3])

    scores, reason = scores_from_estimator(NoProbabilityEstimator(), np.zeros((2, 1)))

    assert scores is None
    assert reason == "predict_proba unavailable"


def test_result_dictionary_json_serializable_and_required_fields_present() -> None:
    result = build_result_dict(
        experiment_name="dummy_exp",
        feature_set="dummy",
        model_name="dummy",
        split_evaluated="val",
        feature_count=2,
        metadata=_metadata(),
        y_true=np.array([3, 1, 2]),
        y_pred=np.array([3, 1, 1]),
        model_parameters={"strategy": "dummy"},
        preprocessing={"imputation": "median", "scaling": "standard"},
        random_seed=42,
        split_version="phase1_patient_split_v1",
        feature_version="dummy_v1",
        code_commit="unknown",
        generated_at="2026-08-31T00:00:00+00:00",
    )

    json.dumps(result, allow_nan=False)
    assert set(result).issuperset(
        {
            "experiment_name",
            "metrics",
            "per_class_metrics",
            "confusion_matrix",
            "roc_auc",
        }
    )
    assert np.asarray(result["confusion_matrix"]).shape == (3, 3)


def test_result_write_read_round_trip(tmp_path: Path) -> None:
    metadata = _metadata()
    y_true = np.array([3, 1, 2])
    y_pred = np.array([3, 1, 1])
    result = build_result_dict(
        experiment_name="dummy_exp",
        feature_set="dummy",
        model_name="dummy",
        split_evaluated="test",
        feature_count=2,
        metadata=metadata,
        y_true=y_true,
        y_pred=y_pred,
        model_parameters={},
        preprocessing={},
        random_seed=None,
        split_version="phase1_patient_split_v1",
        feature_version="dummy_v1",
        generated_at="2026-08-31T00:00:00+00:00",
    )
    predictions = build_prediction_table(metadata, y_true, y_pred)

    paths = write_experiment_results(result, predictions, tmp_path / "experiment")
    read = read_result_json(paths["metrics"])

    assert read == result
    assert paths["predictions"].is_file()
    assert paths["confusion_matrix"].is_file()
    assert paths["experiment_metadata"].is_file()


def test_invalid_metric_result_rejected() -> None:
    with pytest.raises(EvaluationError, match="class_order"):
        validate_result_dict(
            {
                "experiment_name": "x",
                "feature_set": "x",
                "model_name": "x",
                "split_evaluated": "val",
                "feature_count": 1,
                "sample_count": 1,
                "patient_count": 1,
                "class_order": [1, 2],
                "class_names": {},
                "metrics": {},
                "per_class_metrics": {},
                "confusion_matrix": [[1]],
                "roc_auc": {},
                "model_parameters": {},
                "preprocessing": {},
                "random_seed": None,
                "split_version": "x",
                "feature_version": "x",
                "generated_at": "x",
                "code_commit": "unknown",
            }
        )


def test_class_order_constant_is_fixed() -> None:
    assert CLASS_ORDER == [1, 2, 3]
