"""Parity between btdl's full_metrics and Phase 1's evaluate_predictions.

Phase 1 (src.evaluation.metrics) is imported here ONLY, inside a fixture
that removes every newly-imported `src`-prefixed module from sys.modules
on teardown, so this test does not leave Phase 1 registered for others.
"""

import sys

import numpy as np
import pytest
from sklearn.metrics import log_loss as sk_log_loss

from btdl import config
from btdl.contracts import CLASS_ORDER_INDEX, index_to_label
from btdl.evaluation.metrics import full_metrics


def _phase1_evaluate_predictions_context():
    """Plain generator (not a pytest fixture) so its add/cleanup symmetry can
    be driven and inspected directly in a unit test, independent of pytest's
    fixture-wrapping machinery."""

    repo_root = config.repo_root()
    repo_root_str = str(repo_root)
    added_path = repo_root_str not in sys.path
    if added_path:
        sys.path.insert(0, repo_root_str)

    pre_existing_modules = set(sys.modules.keys())
    try:
        from src.evaluation.metrics import evaluate_predictions
    except Exception as exc:
        if added_path:
            sys.path.remove(repo_root_str)
        raise RuntimeError(f"Phase 1 (src.evaluation.metrics) unavailable: {exc}") from exc

    yield evaluate_predictions

    newly_imported = set(sys.modules.keys()) - pre_existing_modules
    for name in newly_imported:
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    if added_path and repo_root_str in sys.path:
        sys.path.remove(repo_root_str)


@pytest.fixture
def phase1_evaluate_predictions():
    try:
        yield from _phase1_evaluate_predictions_context()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _synthetic_predictions(seed=0, n=60):
    rng = np.random.default_rng(seed)
    y_true_idx = rng.integers(0, 3, size=n)
    # Biased-random probs so predictions aren't degenerate.
    probs = rng.dirichlet(alpha=[1.0, 1.0, 1.0], size=n)
    # Nudge probability mass toward the true class for a realistic mix of
    # correct/incorrect predictions.
    for i in range(n):
        probs[i, y_true_idx[i]] += 0.6
    probs = probs / probs.sum(axis=1, keepdims=True)
    return y_true_idx, probs


def test_full_metrics_matches_phase1_evaluate_predictions(phase1_evaluate_predictions):
    evaluate_predictions = phase1_evaluate_predictions
    y_true_idx, probs = _synthetic_predictions()

    btdl_result = full_metrics(y_true_idx, probs)

    y_true_labels = np.array([index_to_label(int(idx)) for idx in y_true_idx])
    y_pred_labels = np.array([index_to_label(int(idx)) for idx in probs.argmax(axis=1)])
    phase1_result = evaluate_predictions(y_true_labels, y_pred_labels, y_score=probs)

    assert btdl_result["accuracy"] == pytest.approx(phase1_result["metrics"]["accuracy"])
    assert btdl_result["balanced_accuracy"] == pytest.approx(phase1_result["metrics"]["balanced_accuracy"])
    assert btdl_result["macro_precision"] == pytest.approx(phase1_result["metrics"]["macro_precision"])
    assert btdl_result["macro_recall"] == pytest.approx(phase1_result["metrics"]["macro_recall"])
    assert btdl_result["macro_f1"] == pytest.approx(phase1_result["metrics"]["macro_f1"])
    assert btdl_result["weighted_precision"] == pytest.approx(phase1_result["metrics"]["weighted_precision"])
    assert btdl_result["weighted_recall"] == pytest.approx(phase1_result["metrics"]["weighted_recall"])
    assert btdl_result["weighted_f1"] == pytest.approx(phase1_result["metrics"]["weighted_f1"])

    for name in ("meningioma", "glioma", "pituitary"):
        btdl_class = btdl_result["per_class"][name]
        phase1_class = phase1_result["per_class_metrics"][name]
        assert btdl_class["precision"] == pytest.approx(phase1_class["precision"])
        assert btdl_class["recall"] == pytest.approx(phase1_class["recall"])
        assert btdl_class["f1"] == pytest.approx(phase1_class["f1"])
        assert btdl_class["support"] == phase1_class["support"]

    assert btdl_result["confusion_matrix"] == phase1_result["confusion_matrix"]

    assert phase1_result["roc_auc"]["available"] is True
    assert btdl_result["roc_auc"]["macro_ovr"] == pytest.approx(phase1_result["roc_auc"]["macro_ovr"])
    assert btdl_result["roc_auc"]["weighted_ovr"] == pytest.approx(phase1_result["roc_auc"]["weighted_ovr"])

    expected_log_loss = sk_log_loss(y_true_idx, probs, labels=list(CLASS_ORDER_INDEX))
    assert btdl_result["log_loss"] == pytest.approx(expected_log_loss)


def test_fixture_removes_exactly_what_it_newly_imported():
    """The phase1_evaluate_predictions fixture's own cleanup is symmetric:
    whatever `src`-prefixed modules IT newly imports, it removes again on
    teardown. (A global "sys.modules has no src" assertion isn't meaningful
    here: pytest's collection phase imports every test module up front,
    including test_phase1_parity.py's module-level Phase 1 import, before
    any test body runs -- that pollution is outside this fixture's control
    and is exactly why test_environment.py's equivalent check runs in a
    subprocess instead.)"""

    generator = _phase1_evaluate_predictions_context()
    before = set(sys.modules.keys())
    try:
        next(generator)  # run up to the yield (imports Phase 1)
    except RuntimeError:
        pytest.skip("Phase 1 (src.evaluation.metrics) unavailable")
    during = set(sys.modules.keys())
    newly_added = during - before
    assert any(name == "src" or name.startswith("src.") for name in newly_added)

    with pytest.raises(StopIteration):
        next(generator)  # run teardown
    after = set(sys.modules.keys())

    # Every src-prefixed module THIS fixture call added is gone again.
    leaked = {name for name in newly_added if name in after and (name == "src" or name.startswith("src."))}
    assert leaked == set()


GLCM_V2_PREDICTIONS_RELPATH = "reports/experiments/glcm_xgboost_v2/test_predictions.csv"
GLCM_V2_SUMMARY_RELPATH = "reports/experiments/glcm_xgboost_v2/experiment_summary.json"

GLCM_V2_EXPECTED = {
    "accuracy": 0.64859,
    "balanced_accuracy": 0.59073,
    "macro_f1": 0.58912,
    "weighted_f1": 0.62898,
    "log_loss": 0.86964,
    "roc_auc_macro_ovr": 0.80715,
}
GLCM_V2_EXPECTED_CONFUSION = [[33, 54, 30], [17, 187, 18], [19, 37, 103]]


@pytest.mark.data
def test_reproduces_glcm_v2_published_test_metrics():
    """Already-published Phase 1 predictions (not Phase 2 touching the test
    split): read-only reproduction of a fixed, previously-reported result."""

    import pandas as pd

    repo_root = config.repo_root()
    predictions_path = repo_root / GLCM_V2_PREDICTIONS_RELPATH
    summary_path = repo_root / GLCM_V2_SUMMARY_RELPATH
    if not predictions_path.is_file() or not summary_path.is_file():
        pytest.skip("Phase 1 GLCM v2 experiment artifacts not present on disk")

    from btdl.contracts import label_to_index

    predictions = pd.read_csv(predictions_path)
    y_true_labels = predictions["true_label"].to_numpy()
    y_true_idx = np.array([label_to_index(int(label)) for label in y_true_labels])
    probs = predictions[["prob_meningioma", "prob_glioma", "prob_pituitary"]].to_numpy()

    result = full_metrics(y_true_idx, probs)

    assert result["accuracy"] == pytest.approx(GLCM_V2_EXPECTED["accuracy"], abs=1e-5)
    assert result["balanced_accuracy"] == pytest.approx(GLCM_V2_EXPECTED["balanced_accuracy"], abs=1e-5)
    assert result["macro_f1"] == pytest.approx(GLCM_V2_EXPECTED["macro_f1"], abs=1e-5)
    assert result["weighted_f1"] == pytest.approx(GLCM_V2_EXPECTED["weighted_f1"], abs=1e-5)
    assert result["log_loss"] == pytest.approx(GLCM_V2_EXPECTED["log_loss"], abs=1e-5)
    assert result["roc_auc"]["macro_ovr"] == pytest.approx(GLCM_V2_EXPECTED["roc_auc_macro_ovr"], abs=1e-5)
    assert result["confusion_matrix"] == GLCM_V2_EXPECTED_CONFUSION
