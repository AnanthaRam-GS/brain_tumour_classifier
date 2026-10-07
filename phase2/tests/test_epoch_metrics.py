import numpy as np
import pytest

from btdl.evaluation.metrics import MetricsInputError, epoch_metrics


def test_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    probs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    metrics = epoch_metrics(y_true, probs)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["log_loss"] == pytest.approx(0.0, abs=1e-6)


def test_all_wrong_predictions():
    y_true = np.array([0, 1, 2])
    probs = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    metrics = epoch_metrics(y_true, probs)
    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["macro_f1"] == pytest.approx(0.0)


def test_missing_class_in_y_true_gets_zero_division_handling():
    # Only classes 0 and 1 present; class 2 never appears as true label.
    y_true = np.array([0, 0, 1, 1])
    probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.05, 0.9, 0.05],
        ]
    )
    metrics = epoch_metrics(y_true, probs)
    assert metrics["accuracy"] == pytest.approx(1.0)
    # class 2 contributes f1=0 (zero_division=0) to the macro average
    assert 0.0 < metrics["macro_f1"] < 1.0


def test_known_log_loss_value():
    y_true = np.array([0])
    probs = np.array([[0.5, 0.25, 0.25]])
    metrics = epoch_metrics(y_true, probs)
    assert metrics["log_loss"] == pytest.approx(-np.log(0.5), abs=1e-6)


def test_rejects_wrong_probs_shape():
    with pytest.raises(MetricsInputError):
        epoch_metrics(np.array([0, 1]), np.array([[0.5, 0.5], [0.5, 0.5]]))


def test_rejects_mismatched_n():
    with pytest.raises(MetricsInputError):
        epoch_metrics(np.array([0, 1, 2]), np.array([[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]]))


def test_rejects_non_finite_probs():
    probs = np.array([[0.5, 0.5, np.nan]])
    with pytest.raises(MetricsInputError):
        epoch_metrics(np.array([0]), probs)


def test_rejects_rows_not_summing_to_one():
    probs = np.array([[0.5, 0.5, 0.5]])  # sums to 1.5
    with pytest.raises(MetricsInputError):
        epoch_metrics(np.array([0]), probs)


def test_accepts_rows_within_tolerance():
    probs = np.array([[0.33334, 0.33333, 0.33333]])  # sums to ~1.0, within 1e-5
    metrics = epoch_metrics(np.array([0]), probs)
    assert "accuracy" in metrics


def test_rejects_out_of_range_label():
    probs = np.array([[0.3, 0.3, 0.4]])
    with pytest.raises(MetricsInputError):
        epoch_metrics(np.array([3]), probs)
