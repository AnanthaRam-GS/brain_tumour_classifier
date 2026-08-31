"""Shared classical ML training utilities for Phase 1 feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.feature_table import METADATA_COLUMNS, VALID_LABELS, validate_feature_table

VALID_SPLITS = ("train", "val", "test")


class TrainingContractError(ValueError):
    """Raised when feature matrices or training contract inputs are invalid."""


@dataclass(frozen=True)
class FeatureMatrices:
    """Train/validation/test matrices derived only from feature-table metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    train_metadata: pd.DataFrame
    X_val: np.ndarray
    y_val: np.ndarray
    val_metadata: pd.DataFrame
    X_test: np.ndarray
    y_test: np.ndarray
    test_metadata: pd.DataFrame
    feature_columns: list[str]


@dataclass(frozen=True)
class FittedPreprocessor:
    """A train-fitted preprocessing pipeline and its declared options."""

    pipeline: Pipeline
    imputation: str | None
    scaling: str | None
    feature_columns: list[str]


@dataclass(frozen=True)
class PreprocessedMatrices:
    """Feature matrices transformed by a preprocessor fit only on training data."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    preprocessor: FittedPreprocessor


@dataclass(frozen=True)
class TrainedEstimator:
    """A fitted sklearn-style estimator with reproducibility metadata."""

    estimator: Any
    model_name: str
    model_parameters: dict[str, Any]
    training_sample_count: int
    feature_count: int


def _require_all_splits_and_classes(table: pd.DataFrame) -> None:
    missing_splits = [split for split in VALID_SPLITS if split not in set(table["split"])]
    if missing_splits:
        raise TrainingContractError(f"feature table missing split(s): {missing_splits}")
    for split in VALID_SPLITS:
        labels = set(table.loc[table["split"].eq(split), "label"])
        missing_labels = sorted(VALID_LABELS.difference(labels))
        if missing_labels:
            raise TrainingContractError(
                f"split {split} missing class label(s): {missing_labels}"
            )


def prepare_feature_matrices(
    feature_data: pd.DataFrame | str,
    *,
    canonical_split: pd.DataFrame | str | None = "data/splits/patient_split.csv",
) -> FeatureMatrices:
    """Prepare train/val/test arrays from a validated feature table.

    The split column is the only source of train/validation/test membership.
    Metadata columns are never included in ``X``.
    """

    if isinstance(feature_data, pd.DataFrame):
        table = validate_feature_table(feature_data, canonical_split=canonical_split)
    else:
        from src.features.feature_table import read_feature_table

        table = read_feature_table(feature_data, canonical_split=canonical_split)
    feature_columns = [column for column in table.columns if column not in METADATA_COLUMNS]
    if not feature_columns:
        raise TrainingContractError("feature table must contain feature columns")
    _require_all_splits_and_classes(table)

    split_payloads: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for split in VALID_SPLITS:
        subset = table[table["split"].eq(split)].copy()
        subset = subset.sort_values("sample_id", key=lambda column: column.astype(int))
        metadata = subset.loc[:, METADATA_COLUMNS].reset_index(drop=True)
        X = subset.loc[:, feature_columns].to_numpy(dtype=np.float64)
        y = subset["label"].to_numpy(dtype=np.int64)
        if not np.isfinite(X).all():
            raise TrainingContractError(f"{split} feature matrix contains NaN or infinite values")
        split_payloads[split] = (X, y, metadata)

    return FeatureMatrices(
        X_train=split_payloads["train"][0],
        y_train=split_payloads["train"][1],
        train_metadata=split_payloads["train"][2],
        X_val=split_payloads["val"][0],
        y_val=split_payloads["val"][1],
        val_metadata=split_payloads["val"][2],
        X_test=split_payloads["test"][0],
        y_test=split_payloads["test"][1],
        test_metadata=split_payloads["test"][2],
        feature_columns=feature_columns,
    )


def fit_feature_preprocessor(
    X_train: np.ndarray,
    *,
    feature_columns: list[str],
    imputation: str | None = "median",
    scaling: str | None = "standard",
) -> FittedPreprocessor:
    """Fit a configurable preprocessing pipeline using training features only."""

    steps = []
    if imputation == "median":
        steps.append(("imputer", SimpleImputer(strategy="median")))
    elif imputation is not None:
        raise TrainingContractError(f"unsupported imputation option: {imputation}")

    if scaling == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaling is not None:
        raise TrainingContractError(f"unsupported scaling option: {scaling}")

    if not steps:
        steps.append(("identity", "passthrough"))
    pipeline = Pipeline(steps)
    pipeline.fit(X_train)
    return FittedPreprocessor(
        pipeline=pipeline,
        imputation=imputation,
        scaling=scaling,
        feature_columns=list(feature_columns),
    )


def transform_feature_matrices(
    matrices: FeatureMatrices,
    *,
    imputation: str | None = "median",
    scaling: str | None = "standard",
) -> PreprocessedMatrices:
    """Fit preprocessing on train and transform train/val/test."""

    preprocessor = fit_feature_preprocessor(
        matrices.X_train,
        feature_columns=matrices.feature_columns,
        imputation=imputation,
        scaling=scaling,
    )
    return PreprocessedMatrices(
        X_train=preprocessor.pipeline.transform(matrices.X_train),
        X_val=preprocessor.pipeline.transform(matrices.X_val),
        X_test=preprocessor.pipeline.transform(matrices.X_test),
        preprocessor=preprocessor,
    )


def train_estimator(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_name: str,
    feature_count: int,
) -> TrainedEstimator:
    """Fit an already-created sklearn-style estimator on training data only."""

    if not hasattr(estimator, "fit"):
        raise TrainingContractError("estimator must provide fit(X, y)")
    estimator.fit(X_train, y_train)
    parameters = estimator.get_params(deep=True) if hasattr(estimator, "get_params") else {}
    return TrainedEstimator(
        estimator=estimator,
        model_name=model_name,
        model_parameters=parameters,
        training_sample_count=int(len(y_train)),
        feature_count=int(feature_count),
    )
