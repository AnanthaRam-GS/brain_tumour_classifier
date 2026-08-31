from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation.metrics import CLASS_ORDER
from src.evaluation.results import validate_result_dict
from src.models.random_forest import (
    FIXED_CLASS_WEIGHT,
    FIXED_N_JOBS,
    RANDOM_SEED,
    RF_CANDIDATE_GRID,
    build_random_forest,
    run_random_forest_experiment,
    tune_random_forest_hyperparameters,
)
from src.models.training import (
    prepare_feature_matrices,
    train_estimator,
    transform_feature_matrices,
)


def _synthetic_feature_table(
    *, per_split: int = 30, n_features: int = 6, seed: int = 0
) -> pd.DataFrame:
    """A small, controlled feature table with a real (learnable) signal per class."""

    rng = np.random.default_rng(seed)
    class_centers = {1: -2.0, 2: 0.0, 3: 2.0}
    rows = []
    sample_id = 1
    for split in ("train", "val", "test"):
        for _ in range(per_split):
            label = int(rng.choice(CLASS_ORDER))
            center = class_centers[label]
            features = rng.normal(loc=center, scale=1.0, size=n_features)
            row = {
                "sample_id": sample_id,
                "patient_id": f"P{sample_id}",
                "label": label,
                "split": split,
            }
            for index, value in enumerate(features):
                row[f"feat_{index}"] = float(value)
            rows.append(row)
            sample_id += 1
    frame = pd.DataFrame(rows)
    # Guarantee every split has all three classes (required by the shared contract).
    for split in ("train", "val", "test"):
        mask = frame["split"].eq(split)
        present = set(frame.loc[mask, "label"])
        missing = sorted(set(CLASS_ORDER) - present)
        idx = frame.index[mask].tolist()
        for i, label in enumerate(missing):
            frame.loc[idx[i], "label"] = label
    return frame


def _matrices(table: pd.DataFrame):
    matrices = prepare_feature_matrices(table, canonical_split=None)
    preprocessed = transform_feature_matrices(matrices)
    return matrices, preprocessed


# --- config YAML -----------------------------------------------------------


def test_config_yaml_loads_and_matches_grid() -> None:
    with open("configs/models/random_forest.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["random_state"] == RANDOM_SEED
    grid = config["hyperparameter_search"]["grid"]
    assert set(grid["n_estimators"]) == {100, 300, 500}
    assert set(grid["max_depth"]) == {None, 10, 20}
    assert set(grid["min_samples_split"]) == {2, 5, 10}
    assert set(grid["min_samples_leaf"]) == {1, 2, 4}
    assert set(grid["max_features"]) == {"sqrt", "log2"}
    assert config["fixed_parameters"]["class_weight"] == FIXED_CLASS_WEIGHT
    assert config["fixed_parameters"]["n_jobs"] == FIXED_N_JOBS


# --- model construction ------------------------------------------------


def test_build_random_forest_applies_fixed_and_custom_params() -> None:
    model = build_random_forest(seed=42, n_estimators=100, max_depth=10, max_features="sqrt")

    assert model.random_state == 42
    assert model.class_weight == FIXED_CLASS_WEIGHT
    assert model.n_jobs == FIXED_N_JOBS
    assert model.n_estimators == 100
    assert model.max_depth == 10
    assert model.max_features == "sqrt"
    assert model.bootstrap is True
    assert model.criterion == "gini"


# --- determinism ---------------------------------------------------------


def test_same_config_and_data_yields_identical_predictions() -> None:
    table = _synthetic_feature_table(seed=1)
    matrices, preprocessed = _matrices(table)

    params = {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}
    model_a = build_random_forest(seed=42, **params)
    model_a.fit(preprocessed.X_train, matrices.y_train)
    pred_a = model_a.predict(preprocessed.X_test)

    model_b = build_random_forest(seed=42, **params)
    model_b.fit(preprocessed.X_train, matrices.y_train)
    pred_b = model_b.predict(preprocessed.X_test)

    assert np.array_equal(pred_a, pred_b)
    assert np.array_equal(model_a.feature_importances_, model_b.feature_importances_)


# --- feature importance ---------------------------------------------------


def test_feature_importance_shape_and_validity() -> None:
    table = _synthetic_feature_table(seed=2, n_features=8)
    matrices, preprocessed = _matrices(table)

    model = build_random_forest(seed=42, n_estimators=50)
    model.fit(preprocessed.X_train, matrices.y_train)
    importances = model.feature_importances_

    assert len(importances) == len(matrices.feature_columns) == 8
    assert np.isfinite(importances).all()
    assert (importances >= 0).all()
    assert importances.sum() == pytest.approx(1.0)


# --- integration with training.py -----------------------------------------


def test_integrates_with_shared_training_api() -> None:
    table = _synthetic_feature_table(seed=3)
    matrices, preprocessed = _matrices(table)

    model = build_random_forest(seed=42, n_estimators=50)
    trained = train_estimator(
        model, preprocessed.X_train, matrices.y_train, model_name="random_forest", feature_count=6
    )

    assert trained.model_name == "random_forest"
    assert trained.feature_count == 6
    assert trained.training_sample_count == len(matrices.y_train)
    preds = trained.estimator.predict(preprocessed.X_test)
    assert preds.shape == matrices.y_test.shape


# --- validation-only selection: test never touched during search ----------


def test_hyperparameter_search_never_uses_test_split() -> None:
    table = _synthetic_feature_table(seed=4)
    matrices, preprocessed = _matrices(table)

    small_grid = (
        {"n_estimators": 20, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 20, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "log2"},
    )

    # By design: tune_random_forest_hyperparameters only accepts train/val
    # arrays -- there is no parameter through which test data could reach it.
    import inspect

    signature = inspect.signature(tune_random_forest_hyperparameters)
    assert set(signature.parameters) >= {"X_train", "y_train", "X_val", "y_val"}
    assert "X_test" not in signature.parameters and "y_test" not in signature.parameters

    best, trials = tune_random_forest_hyperparameters(
        preprocessed.X_train, matrices.y_train, preprocessed.X_val, matrices.y_val,
        candidate_grid=small_grid, seed=42,
    )
    assert len(trials) == 2
    assert best["n_estimators"] in {20}


def test_tie_break_prefers_earlier_grid_index_on_equal_scores() -> None:
    # Two identical-behaving candidates (only differing in an irrelevant
    # parameter that yields the same predictions) should resolve to the
    # first one in grid order when scores tie exactly.
    table = _synthetic_feature_table(seed=5, per_split=40)
    matrices, preprocessed = _matrices(table)

    identical_grid = (
        {"n_estimators": 30, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 30, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    )
    best, trials = tune_random_forest_hyperparameters(
        preprocessed.X_train, matrices.y_train, preprocessed.X_val, matrices.y_val,
        candidate_grid=identical_grid, seed=42,
    )
    assert trials[0]["val_macro_f1"] == trials[1]["val_macro_f1"]
    assert best["grid_index"] == 0


# --- prediction shape / class order ----------------------------------------


def test_prediction_shape_and_class_order_in_confusion_matrix() -> None:
    table = _synthetic_feature_table(seed=6)
    matrices, preprocessed = _matrices(table)

    model = build_random_forest(seed=42, n_estimators=50)
    model.fit(preprocessed.X_train, matrices.y_train)
    preds = model.predict(preprocessed.X_test)

    assert preds.shape == (len(matrices.y_test),)
    assert set(np.unique(preds)).issubset(set(CLASS_ORDER))

    from src.evaluation.metrics import evaluate_predictions

    evaluation = evaluate_predictions(matrices.y_test, preds)
    matrix = np.asarray(evaluation["confusion_matrix"])
    assert matrix.shape == (3, 3)


# --- end-to-end smoke test --------------------------------------------------


def test_end_to_end_smoke(tmp_path) -> None:
    table = _synthetic_feature_table(seed=7, per_split=25, n_features=5)
    csv_path = tmp_path / "features.csv"
    table.to_csv(csv_path, index=False)

    split_csv_path = tmp_path / "split.csv"
    table[["sample_id", "patient_id", "label", "split"]].to_csv(split_csv_path, index=False)

    small_grid = (
        {"n_estimators": 20, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 20, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "log2"},
    )

    outcome = run_random_forest_experiment(
        csv_path,
        feature_set_name="synthetic",
        feature_version="test",
        experiment_name="synthetic_rf_smoke",
        split_csv=split_csv_path,
        split_metadata_json=tmp_path / "does_not_exist.json",
        split_version="synthetic_v1",
        output_root=tmp_path / "reports",
        seed=42,
        candidate_grid=small_grid,
    )

    result = outcome["result"]
    validate_result_dict(result)  # raises on schema violation
    assert result["model_name"] == "random_forest"
    assert result["split_evaluated"] == "test"
    assert result["feature_count"] == 5

    written = outcome["written"]
    assert written["metrics"].is_file()
    assert written["predictions"].is_file()
    assert written["confusion_matrix"].is_file()
    assert written["experiment_metadata"].is_file()
    assert written["hyperparameter_search"].is_file()
    assert written["feature_importance"].is_file()

    importance = outcome["feature_importance"]
    assert list(importance.columns) == ["feature", "importance", "rank"]
    assert len(importance) == 5
    assert importance["rank"].tolist() == list(range(1, 6))
    assert (importance["importance"] >= 0).all()
    assert np.isfinite(importance["importance"]).all()


def test_result_dict_validates_against_shared_schema() -> None:
    table = _synthetic_feature_table(seed=8, per_split=20)
    matrices, preprocessed = _matrices(table)

    model = build_random_forest(seed=42, n_estimators=30)
    trained = train_estimator(
        model, preprocessed.X_train, matrices.y_train, model_name="random_forest", feature_count=6
    )
    preds = trained.estimator.predict(preprocessed.X_test)

    from src.evaluation.results import build_result_dict

    result = build_result_dict(
        experiment_name="unit_test_rf",
        feature_set="synthetic",
        model_name=trained.model_name,
        split_evaluated="test",
        feature_count=trained.feature_count,
        metadata=matrices.test_metadata,
        y_true=matrices.y_test,
        y_pred=preds,
        model_parameters=trained.model_parameters,
        preprocessing={"imputation": "median", "scaling": "standard"},
        random_seed=42,
        split_version="synthetic_v1",
        feature_version="test",
    )
    validate_result_dict(result)  # no exception == valid
