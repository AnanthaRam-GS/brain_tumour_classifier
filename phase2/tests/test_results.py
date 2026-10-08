import json

import numpy as np
import pandas as pd
import pytest

from btdl.evaluation.predict import Predictions
from btdl.evaluation.results import (
    EVALUATION_FILES,
    EvaluationValidationError,
    validate_evaluation_dir,
    write_evaluation,
)


def _synthetic_predictions(n=20, seed=0):
    rng = np.random.default_rng(seed)
    sample_ids = [str(i + 1) for i in range(n)]  # real sample_ids exist in roi_geometry.csv
    patient_ids = [f"p{i % 5}" for i in range(n)]
    y_true_idx = rng.integers(0, 3, size=n)
    probs = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    for i in range(n):
        probs[i, y_true_idx[i]] += 0.6
    probs = probs / probs.sum(axis=1, keepdims=True)
    return Predictions(
        sample_ids=tuple(sample_ids), patient_ids=tuple(patient_ids), y_true_idx=y_true_idx, probs=probs
    )


def _run_metadata():
    return {
        "model_name": "toy",
        "model_version": "v1",
        "seed": 42,
        "lr": 1e-3,
        "best_epoch": 5,
        "git_commit": "abc123",
        "contract_dir_sha256": "deadbeef",
        "contract_conformant": True,
    }


def test_write_then_validate_passes(tmp_path):
    predictions = _synthetic_predictions()
    write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val", n_resamples=50)
    validate_evaluation_dir(tmp_path)  # must not raise


def test_all_files_written(tmp_path):
    predictions = _synthetic_predictions()
    write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val", n_resamples=50)
    for name in EVALUATION_FILES:
        assert (tmp_path / name).is_file()


def test_metrics_json_contains_split_schema_and_run_identifiers(tmp_path):
    predictions = _synthetic_predictions()
    result = write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val", n_resamples=50)
    assert result["split"] == "val"
    assert result["results_schema_version"] == "1.0.0"
    assert result["contract_conformant"] is True
    assert result["run"]["model_name"] == "toy"
    assert result["run"]["seed"] == 42
    assert "roc_curves" not in result  # moved to its own file


def test_predictions_csv_has_required_columns(tmp_path):
    predictions = _synthetic_predictions()
    write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val", n_resamples=50)
    frame = pd.read_csv(tmp_path / "predictions.csv")
    required = {
        "sample_id",
        "patient_id",
        "true_label",
        "true_name",
        "pred_label",
        "pred_name",
        "prob_meningioma",
        "prob_glioma",
        "prob_pituitary",
        "correct",
        "crop_side",
        "size_tertile",
    }
    assert required.issubset(frame.columns)
    assert len(frame) == 20


def test_confusion_matrix_csv_has_header_and_index(tmp_path):
    predictions = _synthetic_predictions()
    write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val", n_resamples=50)
    frame = pd.read_csv(tmp_path / "confusion_matrix.csv", index_col=0)
    assert frame.shape == (3, 3)
    assert list(frame.index) == ["meningioma", "glioma", "pituitary"]
    assert list(frame.columns) == ["meningioma", "glioma", "pituitary"]


@pytest.mark.parametrize("missing_file", list(EVALUATION_FILES))
def test_validation_fails_when_a_file_is_missing(tmp_path, missing_file):
    predictions = _synthetic_predictions()
    write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val", n_resamples=50)
    (tmp_path / missing_file).unlink()
    with pytest.raises(EvaluationValidationError):
        validate_evaluation_dir(tmp_path)


def test_validation_fails_when_metrics_json_is_corrupted(tmp_path):
    predictions = _synthetic_predictions()
    write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val", n_resamples=50)
    with (tmp_path / "metrics.json").open("w") as handle:
        json.dump({"accuracy": 0.5}, handle)  # missing required keys
    with pytest.raises(EvaluationValidationError):
        validate_evaluation_dir(tmp_path)


def test_validation_fails_when_predictions_csv_missing_column(tmp_path):
    predictions = _synthetic_predictions()
    write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val", n_resamples=50)
    frame = pd.read_csv(tmp_path / "predictions.csv")
    frame = frame.drop(columns=["crop_side"])
    frame.to_csv(tmp_path / "predictions.csv", index=False)
    with pytest.raises(EvaluationValidationError):
        validate_evaluation_dir(tmp_path)


def test_validation_fails_when_confusion_matrix_wrong_shape(tmp_path):
    predictions = _synthetic_predictions()
    write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val", n_resamples=50)
    pd.DataFrame([[1, 2], [3, 4]]).to_csv(tmp_path / "confusion_matrix.csv")
    with pytest.raises(EvaluationValidationError):
        validate_evaluation_dir(tmp_path)


def test_validation_fails_when_roc_curves_json_missing_a_class(tmp_path):
    predictions = _synthetic_predictions()
    write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val", n_resamples=50)
    with (tmp_path / "roc_curves.json").open() as handle:
        roc_curves = json.load(handle)
    del roc_curves["meningioma"]
    with (tmp_path / "roc_curves.json").open("w") as handle:
        json.dump(roc_curves, handle)
    with pytest.raises(EvaluationValidationError):
        validate_evaluation_dir(tmp_path)


def test_bootstrap_ci_json_records_settings_used(tmp_path):
    predictions = _synthetic_predictions()
    write_evaluation(
        tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val",
        n_resamples=37, bootstrap_seed=5, alpha=0.1,
    )
    with (tmp_path / "bootstrap_ci.json").open() as handle:
        bootstrap = json.load(handle)
    assert bootstrap["settings"] == {"n_resamples": 37, "seed": 5, "alpha": 0.1}
    assert "metrics" in bootstrap
    for stats in bootstrap["metrics"].values():
        assert stats["n_valid"] <= 37


@pytest.mark.slow
def test_write_evaluation_defaults_come_from_contract(tmp_path):
    from btdl import config

    predictions = _synthetic_predictions(n=6)
    write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val")
    with (tmp_path / "bootstrap_ci.json").open() as handle:
        bootstrap = json.load(handle)
    evaluation_cfg = config.load_contract("evaluation")
    assert bootstrap["settings"]["n_resamples"] == evaluation_cfg["bootstrap"]["n_resamples"]
    assert bootstrap["settings"]["seed"] == evaluation_cfg["bootstrap"]["seed"]
    assert bootstrap["settings"]["alpha"] == evaluation_cfg["bootstrap"]["alpha"]


@pytest.mark.slow
def test_validate_evaluation_dir_require_contract_passes_for_contract_defaults(tmp_path):
    predictions = _synthetic_predictions(n=6)
    write_evaluation(tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val")
    validate_evaluation_dir(tmp_path, require_contract=True)  # must not raise


def test_validate_evaluation_dir_require_contract_fails_for_custom_settings(tmp_path):
    predictions = _synthetic_predictions()
    write_evaluation(
        tmp_path, predictions=predictions, run_metadata=_run_metadata(), split="val", n_resamples=50
    )
    with pytest.raises(EvaluationValidationError, match="bootstrap"):
        validate_evaluation_dir(tmp_path, require_contract=True)
