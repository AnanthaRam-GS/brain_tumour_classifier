"""Pipeline registry for Review 1 single-image inference demos."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class PipelineSpec:
    """Static and dynamic metadata for one demo pipeline."""

    pipeline_id: str
    display_name: str
    feature_name: str
    model_name: str
    feature_adapter: str
    model_adapter: str
    default_model_artifact: Path | None
    supports_probability: bool
    status: str
    expected_feature_count: int | None


_PIPELINES: dict[str, PipelineSpec] = {
    "glcm_xgboost": PipelineSpec(
        pipeline_id="glcm_xgboost",
        display_name="GLCM + XGBoost",
        feature_name="GLCM",
        model_name="XGBoost",
        feature_adapter="glcm",
        model_adapter="xgboost",
        default_model_artifact=Path("models/glcm_xgboost_v1/model_bundle.joblib"),
        supports_probability=True,
        status="implemented",
        expected_feature_count=12,
    ),
    "gabor_svm": PipelineSpec(
        pipeline_id="gabor_svm",
        display_name="Gabor + SVM",
        feature_name="Gabor",
        model_name="SVM",
        feature_adapter="gabor",
        model_adapter="generic_sklearn",
        default_model_artifact=Path("models/gabor_svm_v1/model_bundle.joblib"),
        supports_probability=True,
        status="implemented",
        expected_feature_count=60,
    ),
    "wavelet_rf": PipelineSpec(
        pipeline_id="wavelet_rf",
        display_name="Wavelet + Random Forest",
        feature_name="Wavelet",
        model_name="Random Forest",
        feature_adapter="wavelet",
        model_adapter="generic_sklearn",
        default_model_artifact=Path("models/wavelet_rf_v1/model_bundle.joblib"),
        supports_probability=True,
        status="implemented",
        expected_feature_count=28,
    ),
    "lbp_logistic_regression": PipelineSpec(
        pipeline_id="lbp_logistic_regression",
        display_name="LBP + Logistic Regression",
        feature_name="LBP",
        model_name="Logistic Regression",
        feature_adapter="lbp",
        model_adapter="generic_sklearn",
        default_model_artifact=Path("models/lbp_logistic_regression_v1/model_bundle.joblib"),
        supports_probability=True,
        status="implemented",
        expected_feature_count=10,
    ),
    "hog_knn": PipelineSpec(
        pipeline_id="hog_knn",
        display_name="HOG + KNN",
        feature_name="HOG",
        model_name="KNN",
        feature_adapter="hog",
        model_adapter="not_implemented",
        default_model_artifact=None,
        supports_probability=False,
        status="not_implemented",
        expected_feature_count=1764,
    ),
}


def available_pipeline_ids() -> list[str]:
    """Return registered pipeline IDs."""

    return list(_PIPELINES)


def get_pipeline_spec(pipeline_id: str) -> PipelineSpec:
    """Return a registry entry with current local artifact status."""

    if pipeline_id not in _PIPELINES:
        raise KeyError(f"unknown pipeline: {pipeline_id}")
    spec = _PIPELINES[pipeline_id]
    if spec.status == "not_implemented":
        return spec
    if spec.default_model_artifact is not None and spec.default_model_artifact.is_file():
        return replace(spec, status="ready")
    return replace(spec, status="model_artifact_required")


def list_pipeline_statuses() -> list[PipelineSpec]:
    """Return all registered pipelines with dynamic local statuses."""

    return [get_pipeline_spec(pipeline_id) for pipeline_id in available_pipeline_ids()]

