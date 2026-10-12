"""Synthetic end-to-end test of the walkthrough notebooks' gated demo
path: btdl.walkthrough.run_official_final_test(), followed by the
read-only reporting helpers that depend on real test-mode data
(split_results("test"), baseline_comparison(), example_predictions(),
test_access_status()) -- all against a fake repo, a registered model with
3 frozen seeds, and synthetic Phase 1 baseline predictions aligned to the
fake test split. The real repo's test_access_log.csv is never touched by
this file (ABSOLUTE RULE: real final_test is never run on the real test
split anywhere in this project's test suite)."""

import json
import subprocess

import pandas as pd
import pytest

from btdl import config, walkthrough
from btdl.cli.final_test import FinalTestError, check_final_test_readiness
from btdl.contracts import CLASS_NAMES_BY_INDEX
from btdl.training.device import select_device

from tests.test_final_test_cli import _fully_ready_for_export

DEVICE = select_device("cpu")


def _write_baseline_predictions(tmp_path, test_sample_rows):
    """test_sample_rows: list of (sample_id, patient_id, true_label) for
    the fake repo's test split -- writes a baseline test_predictions.csv
    at the location fake_repo's study.yaml["baseline_reference"] implies
    (reports/experiments/fake_baseline/), with an EXACT sample_id match
    and plausible (mostly-correct) class probabilities."""

    baseline_dir = tmp_path / "reports" / "experiments" / "fake_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample_id, patient_id, true_label in test_sample_rows:
        probs = {name: 0.05 for name in CLASS_NAMES_BY_INDEX}
        true_name = CLASS_NAMES_BY_INDEX[{1: 0, 2: 1, 3: 2}[true_label]]
        probs[true_name] = 1.0 - 0.05 * (len(CLASS_NAMES_BY_INDEX) - 1)
        row = {"sample_id": sample_id, "patient_id": patient_id, "true_label": true_label}
        for name in CLASS_NAMES_BY_INDEX:
            row[f"prob_{name}"] = probs[name]
        rows.append(row)
    pd.DataFrame(rows).to_csv(baseline_dir / "test_predictions.csv", index=False)


@pytest.fixture
def registered_model_with_frozen_seeds(trained_run, monkeypatch):
    """tiny_cnn, fully registered frozen for all 3 of training.yaml's
    final_seeds (reusing test_final_test_cli.py's own
    _fully_ready_for_export fixture helper -- evaluate+freeze+efficiency+
    rehearsal, duplicated to every final_seed, registered, committed),
    plus baseline predictions aligned to the fake test split's 2 samples
    -- ready for run_official_final_test(confirm=True)."""

    tmp_path, device, final_seeds, lr = _fully_ready_for_export(trained_run, monkeypatch)

    split_index = __import__("btdl.data.split", fromlist=["load_split"]).load_split()
    test_rows = [
        (sample_id, row["patient_id"], row["label"])
        for sample_id, row in split_index.items()
        if row["split"] == "test"
    ]
    _write_baseline_predictions(tmp_path, test_rows)

    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline predictions"], cwd=tmp_path, check=True, capture_output=True)

    return tmp_path, device, sorted(final_seeds), lr


def test_run_official_final_test_refuses_without_confirm(registered_model_with_frozen_seeds):
    tmp_path, device, final_seeds, lr = registered_model_with_frozen_seeds

    with pytest.raises(walkthrough.WalkthroughError, match="confirm=True"):
        walkthrough.run_official_final_test("tiny_cnn", confirm=False, device=device)

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    assert not log_path.is_file()
    results_dir = tmp_path / "phase2" / "artifacts" / "results" / "tiny_cnn"
    assert not results_dir.exists()


@pytest.mark.slow
def test_run_official_final_test_confirm_true_tests_and_exports_all_seeds(registered_model_with_frozen_seeds):
    tmp_path, device, final_seeds, lr = registered_model_with_frozen_seeds

    result = walkthrough.run_official_final_test("tiny_cnn", confirm=True, device=device)

    assert set(result["per_seed"].keys()) == set(final_seeds)
    for seed in final_seeds:
        status = result["per_seed"][seed]
        assert status["status"] == "tested"
        assert status["export_status"] == "exported"

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    log_frame = pd.read_csv(log_path)
    assert len(log_frame) == 3
    assert set(log_frame["seed"]) == set(final_seeds)

    for seed in final_seeds:
        assert (tmp_path / "phase2" / "artifacts" / "results" / "tiny_cnn" / f"seed{seed}" / "eval" / "test").is_dir()

    # All 3 final_test calls (real mode) and all 3 export_results calls
    # ran back to back inside run_official_final_test with no commit
    # between them -- had the clean-tree gate on seed 2 or 3 tripped over
    # seed 1's own appended log row (or export_results required a clean
    # tree at all), this call would have raised instead of returning
    # "tested"/"exported" for every seed, as already asserted above. The
    # repo being uncommitted right now is simply evidence nothing
    # committed on our behalf -- that step is left to the presenter.
    status_lines = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout
    assert "test_access_log.csv" in status_lines


@pytest.mark.slow
def test_reporting_helpers_work_after_official_final_test(registered_model_with_frozen_seeds):
    tmp_path, device, final_seeds, lr = registered_model_with_frozen_seeds
    walkthrough.run_official_final_test("tiny_cnn", confirm=True, device=device)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "official final test"], cwd=tmp_path, check=True, capture_output=True)

    split_result = walkthrough.split_results("tiny_cnn", "test")
    assert split_result["available"] is True
    assert set(split_result["per_seed"].keys()) == set(final_seeds)

    baseline_result = walkthrough.baseline_comparison("tiny_cnn")
    assert baseline_result["available"] is True
    assert baseline_result["model"] == "tiny_cnn"
    for name in walkthrough.BASELINE_COMPARISON_METRICS:
        assert name in baseline_result["headline"]
        assert name in baseline_result["paired_vs_baseline"]

    examples = walkthrough.example_predictions("tiny_cnn", "test", n=2)
    assert examples["available"] is True
    assert len(examples["examples"]) > 0

    access_status = walkthrough.test_access_status("tiny_cnn")
    for seed in final_seeds:
        assert access_status["per_seed"][seed]["logged"] is True
        assert access_status["per_seed"][seed]["exported"] is True


@pytest.mark.slow
def test_second_call_runs_nothing_new_and_reports_already_tested(registered_model_with_frozen_seeds):
    tmp_path, device, final_seeds, lr = registered_model_with_frozen_seeds
    first = walkthrough.run_official_final_test("tiny_cnn", confirm=True, device=device)
    assert all(status["status"] == "tested" for status in first["per_seed"].values())

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    log_bytes_before = log_path.read_bytes()

    second = walkthrough.run_official_final_test("tiny_cnn", confirm=True, device=device)
    assert all(status["status"] == "already tested" for status in second["per_seed"].values())
    for seed, status in second["per_seed"].items():
        assert "already tested at" in status["detail"]

    assert log_path.read_bytes() == log_bytes_before


@pytest.mark.slow
def test_check_only_after_testing_reports_already_logged_gate_as_fail(registered_model_with_frozen_seeds):
    from btdl.runs import run_dir_for

    tmp_path, device, final_seeds, lr = registered_model_with_frozen_seeds
    walkthrough.run_official_final_test("tiny_cnn", confirm=True, device=device)

    seed = final_seeds[0]
    run_dir = run_dir_for("tiny_cnn", lr, seed, smoke=False)
    results = check_final_test_readiness(run_dir, repo_root=tmp_path)
    gates = {r["name"]: r["ok"] for r in results}
    assert gates["not_already_logged"] is False
