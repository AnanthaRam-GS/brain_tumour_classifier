import pandas as pd
import pytest

from btdl import config
from btdl.comparison.baseline import BaselineAlignmentError, load_baseline_test_predictions


def _write_study_with_baseline(tmp_path, baseline_relpath):
    study_path = tmp_path / "phase2" / "configs" / "study.yaml"
    study_path.parent.mkdir(parents=True, exist_ok=True)
    study_path.write_text(
        "study_version: '1.0.0'\nmodels:\n  - tiny_cnn\n"
        f"baseline_reference: {baseline_relpath}\n"
    )


def _write_baseline_csv(tmp_path, baseline_dir_relpath, sample_ids, patient_ids, labels):
    baseline_dir = tmp_path / baseline_dir_relpath
    baseline_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "patient_id": patient_ids,
            "true_label": labels,
            "predicted_label": labels,
            "prob_meningioma": [0.8 if lbl == 1 else 0.1 for lbl in labels],
            "prob_glioma": [0.8 if lbl == 2 else 0.1 for lbl in labels],
            "prob_pituitary": [0.8 if lbl == 3 else 0.1 for lbl in labels],
        }
    )
    frame.to_csv(baseline_dir / "test_predictions.csv", index=False)
    (baseline_dir / "experiment_summary.json").write_text("{}")


def test_exact_id_match_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    _write_study_with_baseline(tmp_path, "reports/experiments/fake_baseline/experiment_summary.json")
    _write_baseline_csv(
        tmp_path, "reports/experiments/fake_baseline", ["1", "2", "3"], ["p1", "p1", "p2"], [1, 2, 3]
    )

    result = load_baseline_test_predictions({"1", "2", "3"})
    assert result["name"] == "glcm_xgboost_v2"
    assert set(result["sample_ids"]) == {"1", "2", "3"}
    assert result["probs"].shape == (3, 3)


def test_missing_ids_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    _write_study_with_baseline(tmp_path, "reports/experiments/fake_baseline/experiment_summary.json")
    _write_baseline_csv(
        tmp_path, "reports/experiments/fake_baseline", ["1", "2"], ["p1", "p1"], [1, 2]
    )

    with pytest.raises(BaselineAlignmentError, match="does not exactly match"):
        load_baseline_test_predictions({"1", "2", "3"})  # baseline is missing sample "3"


def test_extra_ids_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    _write_study_with_baseline(tmp_path, "reports/experiments/fake_baseline/experiment_summary.json")
    _write_baseline_csv(
        tmp_path, "reports/experiments/fake_baseline", ["1", "2", "3", "4"], ["p1", "p1", "p2", "p2"], [1, 2, 3, 1]
    )

    with pytest.raises(BaselineAlignmentError, match="does not exactly match"):
        load_baseline_test_predictions({"1", "2", "3"})  # baseline has an extra sample "4"


def test_missing_predictions_file_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    _write_study_with_baseline(tmp_path, "reports/experiments/fake_baseline/experiment_summary.json")
    (tmp_path / "reports" / "experiments" / "fake_baseline").mkdir(parents=True)

    with pytest.raises(BaselineAlignmentError, match="not found"):
        load_baseline_test_predictions({"1", "2", "3"})
