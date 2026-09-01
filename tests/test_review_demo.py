from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.feature_dataset import Phase1Sample
from src.demo.inference import (
    CLASS_NAMES,
    DemoInferenceError,
    DemoPrediction,
    class_name,
    extract_demo_features,
    predict_demo_sample,
)
from src.demo.pipeline_registry import get_pipeline_spec, list_pipeline_statuses
from src.demo.run_review_demo import run_cli
from src.demo.sample_selection import (
    DemoSampleSelectionError,
    get_test_sample_ids,
    select_test_sample,
)
from src.demo.visualize_demo import save_demo_outputs
from src.preprocessing.roi import prepare_tumor_roi


class IdentityPreprocessor:
    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns = feature_columns
        self.fit_called = False

    def fit(self, X, y=None):
        self.fit_called = True
        raise AssertionError("demo inference must not fit preprocessors")

    def transform(self, X):
        return np.asarray(X, dtype=float)


class ConstantEstimator:
    classes_ = np.array([1, 2, 3])

    def __init__(self, prediction: int = 2, probabilities=None) -> None:
        self.prediction = prediction
        self.probabilities = probabilities
        self.fit_called = False

    def fit(self, X, y):
        self.fit_called = True
        raise AssertionError("demo inference must not train estimators")

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.prediction)

    def predict_proba(self, X):
        if self.probabilities is None:
            raise AttributeError("probabilities unavailable")
        return np.tile(np.asarray(self.probabilities, dtype=float), (np.asarray(X).shape[0], 1))


class NoProbabilityEstimator:
    classes_ = np.array([1, 2, 3])

    def predict(self, X):
        return np.ones(np.asarray(X).shape[0], dtype=int)

    def fit(self, X, y):
        raise AssertionError("demo inference must not train estimators")


@pytest.fixture
def tiny_split(tmp_path: Path) -> Path:
    path = tmp_path / "patient_split.csv"
    pd.DataFrame(
        [
            {"sample_id": 1, "patient_id": "P1", "label": 1, "split": "train"},
            {"sample_id": 2, "patient_id": "P2", "label": 2, "split": "val"},
            {"sample_id": 3, "patient_id": "P3", "label": 3, "split": "test"},
            {"sample_id": 4, "patient_id": "P4", "label": 1, "split": "test"},
        ]
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def sample() -> Phase1Sample:
    image = np.zeros((32, 32), dtype=np.float32)
    image[10:22, 10:22] = np.linspace(0.1, 1.0, 144, dtype=np.float32).reshape(12, 12)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    return Phase1Sample(
        sample_id="3",
        patient_id="P3",
        label=3,
        split="test",
        image_normalized=image,
        tumor_mask=mask,
        image_raw=(image * 100).astype(np.float32),
        tumor_border=mask.copy(),
    )


def test_only_test_samples_can_be_selected(tiny_split: Path) -> None:
    assert get_test_sample_ids(tiny_split) == ["3", "4"]
    assert select_test_sample(sample_id=3, split_csv=tiny_split)["split"] == "test"
    with pytest.raises(DemoSampleSelectionError, match="belongs to train, not test"):
        select_test_sample(sample_id=1, split_csv=tiny_split)
    with pytest.raises(DemoSampleSelectionError, match="belongs to val, not test"):
        select_test_sample(sample_id=2, split_csv=tiny_split)


def test_seeded_selection_is_deterministic_and_random_stays_in_test(tiny_split: Path) -> None:
    first = select_test_sample(seed=42, split_csv=tiny_split)
    second = select_test_sample(seed=42, split_csv=tiny_split)
    assert first["sample_id"] == second["sample_id"]
    assert select_test_sample(random_selection=True, split_csv=tiny_split)["split"] == "test"


def test_class_mapping_and_prediction_schema() -> None:
    assert CLASS_NAMES == {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}
    assert class_name(1) == "Meningioma"
    prediction = DemoPrediction(
        pipeline_id="demo",
        feature_name="GLCM",
        model_name="XGBoost",
        sample_id="3",
        patient_id="P3",
        true_label=3,
        true_class_name="Pituitary",
        predicted_label=3,
        predicted_class_name="Pituitary",
        confidence=0.9,
        class_probabilities={"Meningioma": 0.05, "Glioma": 0.05, "Pituitary": 0.9},
        correct=True,
        feature_count=12,
        model_artifact="model.joblib",
    )
    assert prediction.correct is True


def test_glcm_feature_adapter_returns_ordered_12_feature_vector(sample: Phase1Sample) -> None:
    spec = get_pipeline_spec("glcm_xgboost")
    X, columns = extract_demo_features(sample, spec)
    assert X.shape == (1, 12)
    assert len(columns) == 12
    assert columns == list(columns)
    assert np.isfinite(X).all()


def test_missing_model_artifact_has_clear_error(sample: Phase1Sample, tmp_path: Path) -> None:
    spec = replace(
        get_pipeline_spec("gabor_svm"),
        status="ready",
        default_model_artifact=tmp_path / "missing.joblib",
    )
    with pytest.raises(DemoInferenceError, match="Model artifact not found"):
        predict_demo_sample(sample=sample, spec=spec)


def test_generic_sklearn_bundle_predicts_and_maps_probabilities(
    sample: Phase1Sample,
    tmp_path: Path,
) -> None:
    spec = replace(
        get_pipeline_spec("glcm_xgboost"),
        model_adapter="generic_sklearn",
        status="ready",
        default_model_artifact=tmp_path / "bundle.joblib",
    )
    _, columns = extract_demo_features(sample, spec)
    joblib.dump(
        {
            "estimator": ConstantEstimator(2, [0.1, 0.7, 0.2]),
            "preprocessor": IdentityPreprocessor(columns),
        },
        spec.default_model_artifact,
    )
    prediction = predict_demo_sample(sample=sample, spec=spec)
    assert prediction.predicted_label == 2
    assert prediction.confidence == pytest.approx(0.7)
    assert prediction.class_probabilities == {
        "Meningioma": pytest.approx(0.1),
        "Glioma": pytest.approx(0.7),
        "Pituitary": pytest.approx(0.2),
    }
    assert prediction.correct is False


def test_classifier_without_probabilities_is_supported(sample: Phase1Sample, tmp_path: Path) -> None:
    spec = replace(
        get_pipeline_spec("glcm_xgboost"),
        model_adapter="generic_sklearn",
        status="ready",
        default_model_artifact=tmp_path / "bundle.joblib",
    )
    _, columns = extract_demo_features(sample, spec)
    joblib.dump(
        {"estimator": NoProbabilityEstimator(), "preprocessor": IdentityPreprocessor(columns)},
        spec.default_model_artifact,
    )
    prediction = predict_demo_sample(sample=sample, spec=spec)
    assert prediction.predicted_label == 1
    assert prediction.confidence is None
    assert prediction.class_probabilities is None


def test_feature_order_mismatch_is_rejected(sample: Phase1Sample, tmp_path: Path) -> None:
    spec = replace(
        get_pipeline_spec("glcm_xgboost"),
        model_adapter="generic_sklearn",
        status="ready",
        default_model_artifact=tmp_path / "bundle.joblib",
    )
    _, columns = extract_demo_features(sample, spec)
    joblib.dump(
        {
            "estimator": ConstantEstimator(3, [0.1, 0.1, 0.8]),
            "preprocessor": IdentityPreprocessor(list(reversed(columns))),
        },
        spec.default_model_artifact,
    )
    with pytest.raises(DemoInferenceError, match="feature order"):
        predict_demo_sample(sample=sample, spec=spec)


def test_figure_and_json_metadata_are_written(sample: Phase1Sample, tmp_path: Path) -> None:
    roi = prepare_tumor_roi(sample)
    prediction = DemoPrediction(
        pipeline_id="glcm_xgboost",
        feature_name="GLCM",
        model_name="XGBoost",
        sample_id=sample.sample_id,
        patient_id=sample.patient_id,
        true_label=3,
        true_class_name="Pituitary",
        predicted_label=3,
        predicted_class_name="Pituitary",
        confidence=0.86,
        class_probabilities={"Meningioma": 0.04, "Glioma": 0.1, "Pituitary": 0.86},
        correct=True,
        feature_count=12,
        model_artifact="models/glcm_xgboost_v1/model_bundle.joblib",
    )
    before = sample.image_normalized.copy()
    outputs = save_demo_outputs(sample=sample, roi=roi, prediction=prediction, output_dir=tmp_path)
    assert outputs["figure"].is_file()
    assert outputs["figure"].stat().st_size > 0
    assert outputs["metadata"].is_file()
    assert json.loads(outputs["metadata"].read_text())["sample_id"] == "3"
    np.testing.assert_array_equal(sample.image_normalized, before)


def test_hog_knn_reports_not_implemented(sample: Phase1Sample) -> None:
    spec = get_pipeline_spec("hog_knn")
    assert spec.status == "not_implemented"
    with pytest.raises(DemoInferenceError, match="not implemented"):
        extract_demo_features(sample, spec)


def test_list_pipelines_output(capsys) -> None:
    assert run_cli(["--list-pipelines"]) == 0
    output = capsys.readouterr().out
    assert "GLCM + XGBoost" in output
    assert "HOG + KNN" in output
    assert any(spec.pipeline_id == "glcm_xgboost" for spec in list_pipeline_statuses())


def test_cli_reports_selection_errors_cleanly(monkeypatch, capsys, tiny_split: Path) -> None:
    monkeypatch.setattr(
        "src.demo.inference.select_test_sample",
        lambda **kwargs: select_test_sample(sample_id=1, split_csv=tiny_split),
    )
    assert run_cli(["--pipeline", "glcm_xgboost", "--sample-id", "1"]) == 1
    captured = capsys.readouterr()
    assert "belongs to train, not test" in captured.err
    assert "Traceback" not in captured.err
