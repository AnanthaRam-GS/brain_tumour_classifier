import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.models.training import (
    TrainingContractError,
    prepare_feature_matrices,
    train_estimator,
    transform_feature_matrices,
)


class RecordingEstimator:
    def __init__(self) -> None:
        self.fit_rows = 0
        self.classes_ = np.array([1, 2, 3])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RecordingEstimator":
        self.fit_rows = len(X)
        self.seen_y_ = np.asarray(y).copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), 1, dtype=int)

    def get_params(self, deep: bool = True) -> dict:
        return {"recording": True, "deep": deep}


def _feature_table() -> pd.DataFrame:
    rows = []
    sample_id = 1
    for split in ("train", "val", "test"):
        for label in (1, 2, 3):
            rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": f"P{sample_id}",
                    "label": label,
                    "split": split,
                    "dummy_a": float(label),
                    "dummy_b": float(sample_id),
                }
            )
            sample_id += 1
    return pd.DataFrame(rows[::-1])


def test_metadata_columns_excluded_and_features_retained() -> None:
    matrices = prepare_feature_matrices(_feature_table(), canonical_split=None)

    assert matrices.feature_columns == ["dummy_a", "dummy_b"]
    assert matrices.X_train.shape == (3, 2)
    assert list(matrices.train_metadata.columns) == ["sample_id", "patient_id", "label", "split"]


def test_train_val_test_split_and_numeric_order_are_from_metadata() -> None:
    matrices = prepare_feature_matrices(_feature_table(), canonical_split=None)

    assert matrices.train_metadata["sample_id"].tolist() == ["1", "2", "3"]
    assert matrices.val_metadata["sample_id"].tolist() == ["4", "5", "6"]
    assert matrices.test_metadata["sample_id"].tolist() == ["7", "8", "9"]
    assert matrices.y_train.tolist() == [1, 2, 3]


def test_invalid_or_missing_split_rejected() -> None:
    frame = _feature_table()
    frame = frame[~frame["split"].eq("test")]

    with pytest.raises(TrainingContractError, match="missing split"):
        prepare_feature_matrices(frame, canonical_split=None)


def test_empty_feature_set_rejected() -> None:
    frame = _feature_table().loc[:, ["sample_id", "patient_id", "label", "split"]]

    with pytest.raises(Exception, match="feature"):
        prepare_feature_matrices(frame, canonical_split=None)


def test_median_imputer_and_scaler_fit_train_only() -> None:
    frame = _feature_table()
    frame.loc[frame["split"].eq("train"), "dummy_a"] = [1.0, 50.5, 100.0]
    frame.loc[frame["split"].eq("val"), "dummy_a"] = 10_000.0
    frame.loc[frame["split"].eq("test"), "dummy_a"] = -10_000.0
    matrices = prepare_feature_matrices(frame, canonical_split=None)

    transformed = transform_feature_matrices(matrices)
    imputer = transformed.preprocessor.pipeline.named_steps["imputer"]
    scaler = transformed.preprocessor.pipeline.named_steps["scaler"]

    assert imputer.statistics_[0] == pytest.approx(50.5)
    assert scaler.mean_[0] == pytest.approx(np.mean([1.0, 50.5, 100.0]))
    assert transformed.X_train.shape == matrices.X_train.shape
    assert transformed.X_val.shape == matrices.X_val.shape
    assert transformed.X_test.shape == matrices.X_test.shape


def test_preprocessing_disabled_mode_works() -> None:
    matrices = prepare_feature_matrices(_feature_table(), canonical_split=None)

    transformed = transform_feature_matrices(matrices, imputation=None, scaling=None)

    np.testing.assert_array_equal(transformed.X_train, matrices.X_train)
    np.testing.assert_array_equal(transformed.X_val, matrices.X_val)
    np.testing.assert_array_equal(transformed.X_test, matrices.X_test)


def test_estimator_fit_receives_train_only_and_predictions_work() -> None:
    matrices = prepare_feature_matrices(_feature_table(), canonical_split=None)
    estimator = RecordingEstimator()

    trained = train_estimator(
        estimator,
        matrices.X_train,
        matrices.y_train,
        model_name="recording",
        feature_count=len(matrices.feature_columns),
    )

    assert trained.estimator.fit_rows == len(matrices.y_train)
    assert trained.training_sample_count == len(matrices.y_train)
    assert trained.model_parameters["recording"] is True
    assert trained.estimator.predict(matrices.X_val).tolist() == [1, 1, 1]
    assert trained.estimator.predict(matrices.X_test).tolist() == [1, 1, 1]


def test_lightweight_sklearn_estimator_can_use_contract() -> None:
    matrices = prepare_feature_matrices(_feature_table(), canonical_split=None)
    estimator = DummyClassifier(strategy="most_frequent")

    trained = train_estimator(
        estimator,
        matrices.X_train,
        matrices.y_train,
        model_name="dummy",
        feature_count=len(matrices.feature_columns),
    )

    assert trained.estimator.predict(matrices.X_val).shape == matrices.y_val.shape
