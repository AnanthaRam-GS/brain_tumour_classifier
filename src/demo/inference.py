"""Single-sample inference adapters for the Review 1 live demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.data.feature_dataset import Phase1Sample, load_phase1_sample
from src.demo.pipeline_registry import PipelineSpec, get_pipeline_spec
from src.demo.sample_selection import select_test_sample
from src.features.config import load_feature_config
from src.features.gabor import extract_gabor_features
from src.features.glcm import GLCM_FEATURE_COLUMNS, extract_glcm_features
from src.features.lbp import extract_lbp_features
from src.features.wavelet import extract_wavelet_features
from src.models.xgboost_model import (
    predict_project_labels,
    predict_project_proba,
)
from src.preprocessing.roi import prepare_tumor_roi

CLASS_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}
PROJECT_CLASS_ORDER = [1, 2, 3]
PROBABILITY_KEYS = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}
METADATA_COLUMNS = ("sample_id", "patient_id", "label", "split")


class DemoInferenceError(RuntimeError):
    """Raised when demo inference cannot proceed."""


@dataclass(frozen=True)
class DemoPrediction:
    """Immutable single-image demo prediction result."""

    pipeline_id: str
    feature_name: str
    model_name: str
    sample_id: str
    patient_id: str
    true_label: int
    true_class_name: str
    predicted_label: int
    predicted_class_name: str
    confidence: float | None
    class_probabilities: dict[str, float] | None
    correct: bool
    feature_count: int
    model_artifact: str


def class_name(label: int) -> str:
    """Return display name for a project class label."""

    try:
        return CLASS_NAMES[int(label)]
    except KeyError as exc:
        raise DemoInferenceError(f"unknown class label: {label}") from exc


def _feature_vector_from_row(
    row: dict[str, Any],
    feature_columns: list[str],
) -> np.ndarray:
    values = np.asarray([row[column] for column in feature_columns], dtype=np.float64)
    if not np.isfinite(values).all():
        raise DemoInferenceError("feature vector contains NaN or infinite values")
    return values.reshape(1, -1)


def extract_demo_features(sample: Phase1Sample, spec: PipelineSpec) -> tuple[np.ndarray, list[str]]:
    """Extract one feature vector using the registered feature adapter."""

    if spec.feature_adapter == "glcm":
        row = extract_glcm_features(sample)
        feature_columns = list(GLCM_FEATURE_COLUMNS)
    elif spec.feature_adapter == "lbp":
        row = extract_lbp_features(sample)
        feature_columns = [f"lbp_{index}" for index in range(spec.expected_feature_count or 0)]
    elif spec.feature_adapter == "gabor":
        roi = prepare_tumor_roi(sample)
        features = extract_gabor_features(roi.roi_image_masked_resized, load_feature_config("gabor"))
        feature_columns = list(features)
        row = {**features}
    elif spec.feature_adapter == "wavelet":
        roi = prepare_tumor_roi(sample)
        features = extract_wavelet_features(
            roi.roi_image_masked_resized,
            load_feature_config("wavelet"),
        )
        feature_columns = list(features)
        row = {**features}
    elif spec.feature_adapter == "hog":
        raise DemoInferenceError("Pipeline not implemented yet: HOG + KNN.")
    else:
        raise DemoInferenceError(f"unsupported feature adapter: {spec.feature_adapter}")

    if spec.expected_feature_count is not None and len(feature_columns) != spec.expected_feature_count:
        raise DemoInferenceError(
            f"{spec.display_name} produced {len(feature_columns)} features; "
            f"expected {spec.expected_feature_count}."
        )
    missing = [column for column in feature_columns if column not in row]
    if missing:
        raise DemoInferenceError(f"feature extractor missing expected column(s): {missing[:5]}")
    return _feature_vector_from_row(row, feature_columns), feature_columns


def _transform_with_preprocessor(preprocessor: Any, X: np.ndarray) -> np.ndarray:
    if preprocessor is None:
        return X
    if hasattr(preprocessor, "pipeline"):
        return preprocessor.pipeline.transform(X)
    if hasattr(preprocessor, "transform"):
        return preprocessor.transform(X)
    raise DemoInferenceError("model bundle preprocessor does not provide transform(X)")


def _bundle_feature_columns(bundle: dict[str, Any]) -> list[str] | None:
    if "feature_columns" in bundle:
        return list(bundle["feature_columns"])
    preprocessor = bundle.get("preprocessor")
    if hasattr(preprocessor, "feature_columns"):
        return list(preprocessor.feature_columns)
    return None


def _load_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DemoInferenceError(
            f"Model artifact not found for requested pipeline.\n"
            f"Expected: {path}\n"
            "Action: Run/export the finalized trained model bundle first. "
            "The demo performs inference only and never retrains."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise DemoInferenceError("model artifact must be a joblib dictionary bundle")
    if "estimator" not in bundle:
        raise DemoInferenceError("model bundle is missing required estimator")
    return bundle


def _predict_with_xgboost(bundle: dict[str, Any], X: np.ndarray) -> tuple[int, dict[str, float] | None]:
    estimator = bundle["estimator"]
    X_ready = _transform_with_preprocessor(bundle.get("preprocessor"), X)
    predicted = int(predict_project_labels(estimator, X_ready)[0])
    proba = predict_project_proba(estimator, X_ready)[0]
    probabilities = {
        PROBABILITY_KEYS[label]: float(proba[index])
        for index, label in enumerate(PROJECT_CLASS_ORDER)
    }
    return predicted, probabilities


def _predict_with_generic_bundle(
    bundle: dict[str, Any],
    X: np.ndarray,
) -> tuple[int, dict[str, float] | None]:
    estimator = bundle["estimator"]
    X_ready = _transform_with_preprocessor(bundle.get("preprocessor"), X)
    predicted = int(np.asarray(estimator.predict(X_ready)).reshape(-1)[0])

    probabilities = None
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X_ready), dtype=np.float64)
        if proba.ndim == 2 and proba.shape[0] == 1 and np.isfinite(proba).all():
            classes = [int(value) for value in getattr(estimator, "classes_", PROJECT_CLASS_ORDER)]
            probabilities = {}
            for label in PROJECT_CLASS_ORDER:
                if label in classes:
                    probabilities[PROBABILITY_KEYS[label]] = float(proba[0, classes.index(label)])
    return predicted, probabilities


def predict_demo_sample(
    *,
    sample: Phase1Sample,
    spec: PipelineSpec,
    model_artifact: Path | None = None,
) -> DemoPrediction:
    """Run feature extraction and saved-model inference for one sample."""

    if spec.status == "not_implemented":
        raise DemoInferenceError(f"Pipeline not implemented yet: {spec.display_name}.")
    artifact = model_artifact or spec.default_model_artifact
    if artifact is None:
        raise DemoInferenceError(f"No model artifact path is configured for {spec.display_name}.")
    bundle = _load_bundle(Path(artifact))
    X, feature_columns = extract_demo_features(sample, spec)

    trained_columns = _bundle_feature_columns(bundle)
    if trained_columns is not None and trained_columns != feature_columns:
        raise DemoInferenceError("inference feature order does not match the trained model bundle")

    if spec.model_adapter == "xgboost":
        predicted, probabilities = _predict_with_xgboost(bundle, X)
    elif spec.model_adapter == "generic_sklearn":
        predicted, probabilities = _predict_with_generic_bundle(bundle, X)
    else:
        raise DemoInferenceError(f"unsupported model adapter: {spec.model_adapter}")

    confidence = None
    if probabilities is not None:
        confidence = probabilities.get(class_name(predicted))
    return DemoPrediction(
        pipeline_id=spec.pipeline_id,
        feature_name=spec.feature_name,
        model_name=spec.model_name,
        sample_id=sample.sample_id,
        patient_id=sample.patient_id,
        true_label=sample.label,
        true_class_name=class_name(sample.label),
        predicted_label=predicted,
        predicted_class_name=class_name(predicted),
        confidence=confidence,
        class_probabilities=probabilities,
        correct=predicted == sample.label,
        feature_count=len(feature_columns),
        model_artifact=str(artifact),
    )


def run_demo_inference(
    *,
    pipeline_id: str,
    sample_id: str | int | None = None,
    seed: int | None = None,
    random_selection: bool = False,
    split_csv: Path | str = "data/splits/patient_split.csv",
    samples_dir: Path | str = "data/processed/samples",
) -> tuple[Phase1Sample, Any, DemoPrediction]:
    """Select a TEST sample, prepare its ROI, and run one saved pipeline."""

    spec = get_pipeline_spec(pipeline_id)
    if spec.status == "not_implemented":
        raise DemoInferenceError(f"Pipeline not implemented yet: {spec.display_name}.")
    if spec.status == "model_artifact_required":
        raise DemoInferenceError(
            f"Model artifact not found for {spec.display_name}.\n"
            f"Expected: {spec.default_model_artifact}\n"
            "Action: Run/export the trained model first. The demo does not retrain."
        )
    selected = select_test_sample(
        sample_id=sample_id,
        seed=seed,
        random_selection=random_selection,
        split_csv=split_csv,
    )
    sample = load_phase1_sample(
        selected["sample_id"],
        split_csv=split_csv,
        samples_dir=samples_dir,
    )
    roi = prepare_tumor_roi(sample)
    prediction = predict_demo_sample(sample=sample, spec=spec)
    return sample, roi, prediction

