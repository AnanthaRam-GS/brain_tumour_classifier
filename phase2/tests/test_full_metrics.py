import numpy as np
import pytest

from btdl.evaluation.metrics import MetricsInputError, full_metrics


def _hand_case():
    # 2 samples per class, all correct except one meningioma misclassified as glioma.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],  # true=0 (meningioma), predicted=1 (glioma)
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8],
            [0.2, 0.3, 0.5],
        ]
    )
    return y_true, probs


def test_n_and_basic_fields():
    y_true, probs = _hand_case()
    metrics = full_metrics(y_true, probs)
    assert metrics["n"] == 6
    assert metrics["accuracy"] == pytest.approx(5 / 6)


def test_confusion_matrix_orientation_rows_true_cols_predicted():
    y_true, probs = _hand_case()
    metrics = full_metrics(y_true, probs)
    cm = np.array(metrics["confusion_matrix"])
    # true=0 row: one predicted 0, one predicted 1 (misclassified), zero predicted 2.
    assert list(cm[0]) == [1, 1, 0]
    # true=1 row: both predicted 1.
    assert list(cm[1]) == [0, 2, 0]
    # true=2 row: both predicted 2.
    assert list(cm[2]) == [0, 0, 2]


def test_per_class_keyed_by_class_name():
    y_true, probs = _hand_case()
    metrics = full_metrics(y_true, probs)
    assert set(metrics["per_class"].keys()) == {"meningioma", "glioma", "pituitary"}
    assert metrics["per_class"]["pituitary"]["support"] == 2
    assert metrics["per_class"]["pituitary"]["recall"] == pytest.approx(1.0)
    # meningioma: 1 of 2 true samples correctly predicted
    assert metrics["per_class"]["meningioma"]["recall"] == pytest.approx(0.5)


def test_argmax_tie_broken_by_lowest_index():
    y_true = np.array([0])
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])  # exact 3-way tie
    metrics = full_metrics(y_true, probs)
    cm = np.array(metrics["confusion_matrix"])
    # predicted class must be index 0 (lowest index wins the tie)
    assert cm[0, 0] == 1


def test_roc_auc_per_class_known_value():
    # Perfectly separable binary-style signal for class 0 vs rest.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.05, 0.05, 0.9],
        ]
    )
    metrics = full_metrics(y_true, probs)
    assert metrics["roc_auc"]["per_class"]["meningioma"] == pytest.approx(1.0)
    assert metrics["roc_auc"]["macro_ovr"] == pytest.approx(1.0)


def test_roc_auc_undefined_when_class_absent():
    # Class 2 never appears in y_true -> per-class AUC for pituitary is undefined.
    y_true = np.array([0, 0, 1, 1])
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
        ]
    )
    metrics = full_metrics(y_true, probs)
    assert metrics["roc_auc"]["per_class"]["pituitary"] is None
    assert metrics["roc_curves"]["pituitary"] == {"fpr": [], "tpr": [], "thresholds": []}


def test_weighted_vs_macro_differ_with_imbalanced_support():
    y_true = np.array([0, 0, 0, 0, 1, 2])  # class 0 heavily overrepresented
    probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],  # misclassified
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    metrics = full_metrics(y_true, probs)
    assert metrics["weighted_f1"] != pytest.approx(metrics["macro_f1"])


def test_macro_and_weighted_ovr_none_when_a_class_is_absent():
    # Class 2 (pituitary) never appears in y_true.
    y_true = np.array([0, 0, 1, 1])
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
        ]
    )
    metrics = full_metrics(y_true, probs)
    # sklearn's roc_auc_score(multi_class="ovr") warns and returns NaN rather
    # than raising when a class is absent; that must surface as None, not NaN
    # (NaN would be inconsistent with per_class's None and isn't clean JSON).
    assert metrics["roc_auc"]["macro_ovr"] is None
    assert metrics["roc_auc"]["weighted_ovr"] == pytest.approx(1.0)  # only uses present classes


def test_rejects_invalid_input_same_as_epoch_metrics():
    with pytest.raises(MetricsInputError):
        full_metrics(np.array([0, 1]), np.array([[0.5, 0.5], [0.5, 0.5]]))
