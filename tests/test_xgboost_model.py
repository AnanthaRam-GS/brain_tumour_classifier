from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost

from src.evaluation.results import build_prediction_table
from src.models.xgboost_model import (
    PROJECT_LABELS,
    candidate_parameter_grid,
    create_xgboost_estimator,
    fit_xgboost_estimator,
    internal_to_project_labels,
    load_xgboost_config,
    predict_project_labels,
    predict_project_proba,
    project_to_internal_labels,
)


def test_xgboost_dependency_imports() -> None:
    assert xgboost.__version__


def test_config_loads_and_model_objective_multiclass() -> None:
    config = load_xgboost_config()
    assert config["model_name"] == "xgboost"
    assert config["objective"] == "multi:softprob"
    assert config["num_class"] == 3
    assert config["random_seed"] == 42
    assert config["preprocessing"] == {"imputation": "median", "scaling": None}
    assert len(candidate_parameter_grid(config)) == 8


def test_label_mapping_round_trip() -> None:
    project = np.array([1, 2, 3, 1])
    internal = project_to_internal_labels(project)
    assert internal.tolist() == [0, 1, 2, 0]
    assert internal_to_project_labels(internal).tolist() == project.tolist()
    with pytest.raises(ValueError):
        project_to_internal_labels(np.array([0]))
    with pytest.raises(ValueError):
        internal_to_project_labels(np.array([3]))


def test_estimator_predictions_and_probabilities_are_project_ordered() -> None:
    config = load_xgboost_config()
    params = {**candidate_parameter_grid(config)[0], "n_estimators": 3, "n_jobs": 1}
    estimator = create_xgboost_estimator(config, params)
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
            [2.0, 2.0],
            [2.1, 2.0],
        ]
    )
    y = np.array([1, 1, 2, 2, 3, 3])
    fit_xgboost_estimator(estimator, X, y)
    pred = predict_project_labels(estimator, X)
    proba = predict_project_proba(estimator, X)
    assert set(pred).issubset(PROJECT_LABELS)
    assert proba.shape == (6, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_prediction_table_probability_columns_use_project_names() -> None:
    metadata = pd.DataFrame({"sample_id": ["1", "2"], "patient_id": ["a", "b"]})
    table = build_prediction_table(
        metadata,
        np.array([1, 2]),
        np.array([1, 3]),
        y_score=np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]),
    )
    assert list(table.columns) == [
        "sample_id",
        "patient_id",
        "true_label",
        "predicted_label",
        "prob_meningioma",
        "prob_glioma",
        "prob_pituitary",
    ]
