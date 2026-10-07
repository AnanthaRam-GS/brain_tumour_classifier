import numpy as np
import pandas as pd
import pytest

from btdl.evaluation.predict import Predictions
from btdl.evaluation.stratified import classify_tertile, stratified_report


def test_classify_tertile_boundaries():
    edges = [50.0, 100.0]
    assert classify_tertile(10, edges) == "small"
    assert classify_tertile(50, edges) == "small"  # <= low -> small
    assert classify_tertile(51, edges) == "medium"
    assert classify_tertile(100, edges) == "medium"  # <= high -> medium
    assert classify_tertile(101, edges) == "large"


def _make_predictions_and_geometry():
    sample_ids = ["1", "2", "3", "4", "5", "6"]
    y_true_idx = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],  # misclassified: true=0, predicted=1
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.1, 0.8],
        ]
    )
    predictions = Predictions(
        sample_ids=tuple(sample_ids),
        patient_ids=tuple(f"p{i}" for i in sample_ids),
        y_true_idx=y_true_idx,
        probs=probs,
    )
    # crop_side: samples 1,2 small; 3,4 medium; 5,6 large
    geometry = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "crop_side": [10, 20, 60, 70, 150, 160],
        }
    )
    tertiles = {"edges": [50.0, 100.0]}
    return predictions, geometry, tertiles


def test_stratified_report_counts_and_metrics():
    predictions, geometry, tertiles = _make_predictions_and_geometry()
    report = stratified_report(predictions, geometry, tertiles)

    assert report["tertiles"]["small"]["n"] == 2
    assert report["tertiles"]["medium"]["n"] == 2
    assert report["tertiles"]["large"]["n"] == 2

    # small tertile: samples 1 (correct) and 2 (misclassified as glioma) -> accuracy 0.5
    assert report["tertiles"]["small"]["accuracy"] == pytest.approx(0.5)
    # medium and large tertiles are both perfectly classified
    assert report["tertiles"]["medium"]["accuracy"] == pytest.approx(1.0)
    assert report["tertiles"]["large"]["accuracy"] == pytest.approx(1.0)


def test_stratified_report_class_tertile_counts():
    predictions, geometry, tertiles = _make_predictions_and_geometry()
    report = stratified_report(predictions, geometry, tertiles)
    counts = report["class_tertile_counts"]
    # true class 0 (meningioma): both samples (1,2) are in the small tertile
    assert counts["meningioma"]["small"] == 2
    assert counts["meningioma"]["medium"] == 0
    assert counts["meningioma"]["large"] == 0
    # true class 1 (glioma): both in medium
    assert counts["glioma"]["medium"] == 2
    # true class 2 (pituitary): both in large
    assert counts["pituitary"]["large"] == 2


def test_empty_tertile_reports_none_metrics():
    predictions, geometry, _ = _make_predictions_and_geometry()
    # Tertile edges chosen so "large" is empty.
    tertiles = {"edges": [1000.0, 2000.0]}
    report = stratified_report(predictions, geometry, tertiles)
    assert report["tertiles"]["large"]["n"] == 0
    assert report["tertiles"]["large"]["accuracy"] is None
    assert report["tertiles"]["large"]["macro_f1"] is None
