"""Synthetic end-to-end test of the walkthrough notebooks' gated demo
path: btdl.walkthrough.run_official_final_test(), followed by the
read-only reporting helpers that depend on real test-mode data
(split_results("test"), phase1_comparison_doc(), example_predictions(),
test_access_status()) -- all against a fake repo and a registered model
with 3 frozen seeds. The real repo's test_access_log.csv is never touched
by this file (ABSOLUTE RULE: real final_test is never run on the real
test split anywhere in this project's test suite)."""

import json
import subprocess

import pandas as pd
import pytest

from btdl import config, walkthrough
from btdl.cli.final_test import FinalTestError, check_final_test_readiness
from btdl.training.device import select_device

from tests.test_final_test_cli import _fully_ready_for_export

DEVICE = select_device("cpu")


def _write_phase1_comparison_doc(tmp_path, model_name="tiny_cnn"):
    """Writes a minimal hand-written-style Phase 1 vs Phase 2 comparison
    doc under phase2/docs/, matching phase1_comparison_doc()'s naming
    convention -- "<phase1_model>_vs_<model>.md"."""

    docs_dir = tmp_path / "phase2" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / f"fake_baseline_vs_{model_name}.md").write_text(
        f"# Fake baseline vs {model_name}\n\n| metric | fake baseline | {model_name} |\n"
        f"|---|---|---|\n| macro_f1 | 0.50 | 0.95 |\n"
    )


@pytest.fixture
def registered_model_with_frozen_seeds(trained_run, monkeypatch):
    """tiny_cnn, fully registered frozen for all 3 of training.yaml's
    final_seeds (reusing test_final_test_cli.py's own
    _fully_ready_for_export fixture helper -- evaluate+freeze+efficiency+
    rehearsal, duplicated to every final_seed, registered, committed),
    plus a Phase 1 comparison doc -- ready for
    run_official_final_test(confirm=True)."""

    tmp_path, device, final_seeds, lr = _fully_ready_for_export(trained_run, monkeypatch)

    _write_phase1_comparison_doc(tmp_path)

    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "phase 1 comparison doc"], cwd=tmp_path, check=True, capture_output=True)

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

    comparison_doc = walkthrough.phase1_comparison_doc("tiny_cnn")
    assert comparison_doc["available"] is True
    assert "tiny_cnn" in comparison_doc["content"]

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
