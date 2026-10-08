import csv
import dataclasses
import json
import subprocess

import pytest

from btdl import config
from btdl.cli.final_test import FinalTestError, run_final_test
from btdl.cli.freeze import run_freeze
from btdl.evaluation.reload_check import ReloadEquivalenceError
from btdl.evaluation.results import validate_evaluation_dir
from btdl.models.registry import MODEL_REGISTRY


def _frozen_run(trained_run):
    tmp_path, run_dir, model, device = trained_run
    run_freeze(run_dir)
    return tmp_path, run_dir, model, device


def _enable_real_mode(monkeypatch, model_name="tiny_cnn"):
    """Temporarily mark the model as NOT reference_only, so the real
    (non-rehearsal) final_test path can be exercised -- tiny_cnn is
    reference_only=True in the actual registry and can only ever rehearse."""

    original_spec = MODEL_REGISTRY[model_name]
    fake_spec = dataclasses.replace(original_spec, reference_only=False)
    monkeypatch.setitem(MODEL_REGISTRY, model_name, fake_spec)


def _write_matching_selection(tmp_path, run_dir, model_name="tiny_cnn"):
    with (run_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    selection_path = tmp_path / "phase2" / "artifacts" / "selection" / f"{model_name}.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({"selected_lr": metadata["lr"]}))
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add selection"], cwd=tmp_path, check=True, capture_output=True)


# ---- base gates (apply in both real and rehearsal mode) -----------------------------


def test_final_test_requires_confirm_flag(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    with pytest.raises(FinalTestError, match="confirm-final-test"):
        run_final_test(run_dir, confirm_final_test=False, device=device, batch_size=2)


def test_final_test_requires_frozen_run(trained_run):
    tmp_path, run_dir, model, device = trained_run  # not frozen
    with pytest.raises(FinalTestError, match="not frozen"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_refuses_mismatched_frozen_sha256(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    frozen_path = run_dir / "FROZEN.json"
    with frozen_path.open() as handle:
        frozen = json.load(handle)
    frozen["best_pt_sha256"] = "deadbeef" * 8
    with frozen_path.open("w") as handle:
        json.dump(frozen, handle)

    with pytest.raises(FinalTestError, match="sha256"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_refuses_contract_mismatch(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    frozen_path = run_dir / "FROZEN.json"
    with frozen_path.open() as handle:
        frozen = json.load(handle)
    frozen["contract_dir_sha256"] = "deadbeef" * 8
    with frozen_path.open("w") as handle:
        json.dump(frozen, handle)

    with pytest.raises(FinalTestError, match="contract"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_refuses_non_conformant_metadata(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    metadata["contract_conformant"] = False
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(FinalTestError, match="contract_conformant"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_refuses_dirty_working_tree(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    (tmp_path / "uncommitted_file.txt").write_text("not committed")

    with pytest.raises(FinalTestError, match="clean"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


# ---- C6: real-mode-only gates -----------------------------------------------------------


def test_real_mode_refuses_reference_only_model(trained_run):
    # tiny_cnn is reference_only=True in the real registry -- real mode must
    # always refuse it, regardless of anything else.
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    with pytest.raises(FinalTestError, match="reference_only"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_real_mode_refuses_seed_not_in_final_seeds(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)

    metadata_path = run_dir / "metadata.json"
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    metadata["seed"] = 999  # not in final_seeds
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle)
    # run_dir is under phase2/runs/, which is gitignored -- editing a file
    # inside it never dirties the tree, so no commit is needed (or possible).

    with pytest.raises(FinalTestError, match="final_seeds"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_real_mode_refuses_missing_selection_file(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)

    with pytest.raises(FinalTestError, match="selection"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_real_mode_refuses_lr_not_matching_selection(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)

    selection_path = tmp_path / "phase2" / "artifacts" / "selection" / "tiny_cnn.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({"selected_lr": 9.99}))  # does not match run's lr
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add mismatched selection"], cwd=tmp_path, check=True, capture_output=True)

    with pytest.raises(FinalTestError, match="does not match"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


# ---- C6 success path + log row ---------------------------------------------------------


@pytest.mark.slow
def test_real_mode_succeeds_and_appends_one_log_row(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)

    metrics = run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)
    assert metrics["split"] == "test"
    validate_evaluation_dir(run_dir / "eval" / "test")

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    assert log_path.is_file()
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["model_name"] == "tiny_cnn"


@pytest.mark.slow
def test_real_mode_refuses_repeated_run(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)

    run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)

    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "log test access"], cwd=tmp_path, check=True, capture_output=True)

    with pytest.raises(FinalTestError, match="already"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1  # the refused second call appended nothing


# ---- C7: rehearsal -----------------------------------------------------------------------


@pytest.mark.slow
def test_rehearsal_succeeds_without_selection_or_real_mode(trained_run):
    # tiny_cnn is reference_only and has no selection file -- rehearsal must
    # still succeed (selection + reference_only gates are waived).
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    metrics = run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)
    assert metrics["split"] == "val"
    validate_evaluation_dir(run_dir / "eval" / "rehearsal_val")


@pytest.mark.slow
def test_rehearsal_writes_no_log_row(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    if log_path.is_file():
        with log_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 0


def test_rehearsal_still_requires_frozen(trained_run):
    tmp_path, run_dir, model, device = trained_run  # not frozen
    with pytest.raises(FinalTestError, match="not frozen"):
        run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)


def test_rehearsal_still_requires_clean_tree(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    (tmp_path / "uncommitted_file.txt").write_text("not committed")
    with pytest.raises(FinalTestError, match="clean"):
        run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)


@pytest.mark.slow
def test_rehearsal_can_be_run_multiple_times(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)
    # Should not raise: rehearsal has no "already logged" gate.
    run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)


# ---- C8: reload equivalence --------------------------------------------------------------


def test_reload_mismatch_is_detected(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    val_predictions_path = run_dir / "val_predictions.csv"
    import pandas as pd

    frame = pd.read_csv(val_predictions_path)
    # Corrupt one probability column so it can no longer match a fresh reload.
    prob_columns = [c for c in frame.columns if c.startswith("prob_")]
    frame.loc[0, prob_columns[0]] = 0.0
    frame.to_csv(val_predictions_path, index=False)
    # run_dir is gitignored -- no commit needed (or possible) for this edit.

    with pytest.raises(ReloadEquivalenceError):
        run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)
