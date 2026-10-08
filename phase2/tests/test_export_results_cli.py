import dataclasses
import json
import subprocess

import pytest

from btdl.cli.efficiency import run_efficiency
from btdl.cli.evaluate import run_evaluate
from btdl.cli.export_results import ExportResultsError, run_export_results
from btdl.cli.final_test import run_final_test
from btdl.cli.freeze import run_freeze
from btdl.evaluation.results import validate_evaluation_dir
from btdl.models.registry import MODEL_REGISTRY


def _enable_real_mode(monkeypatch, model_name="tiny_cnn"):
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


def _final_tested_run(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = trained_run
    # trained_run calls fit() directly (not via cli.train.run_train), so
    # eval/val/ -- normally written automatically by run_train -- doesn't
    # exist yet; export_results requires it.
    run_evaluate(run_dir, split="val", device=device, batch_size=2)
    run_freeze(run_dir)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)
    run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)
    run_efficiency(run_dir, device=device)
    return tmp_path, run_dir


@pytest.mark.slow
def test_export_results_copies_expected_files_and_validates(trained_run, monkeypatch):
    tmp_path, run_dir = _final_tested_run(trained_run, monkeypatch)
    with (run_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)

    dest_dir = run_export_results(run_dir)
    assert dest_dir == tmp_path / "phase2" / "artifacts" / "results" / "tiny_cnn" / f"seed{metadata['seed']}"

    for name in ("metadata.json", "history.csv", "FROZEN.json", "efficiency.json"):
        assert (dest_dir / name).is_file()
    assert (dest_dir / "eval" / "val").is_dir()
    assert (dest_dir / "eval" / "test").is_dir()
    validate_evaluation_dir(dest_dir / "eval" / "val", require_contract=True)
    validate_evaluation_dir(dest_dir / "eval" / "test", require_contract=True)


@pytest.mark.slow
def test_export_results_never_copies_checkpoints(trained_run, monkeypatch):
    tmp_path, run_dir = _final_tested_run(trained_run, monkeypatch)
    dest_dir = run_export_results(run_dir)
    assert not (dest_dir / "best.pt").exists()
    assert not (dest_dir / "last.pt").exists()
    assert list(dest_dir.rglob("*.pt")) == []


@pytest.mark.slow
def test_export_results_refuses_to_overwrite(trained_run, monkeypatch):
    tmp_path, run_dir = _final_tested_run(trained_run, monkeypatch)
    run_export_results(run_dir)
    with pytest.raises(ExportResultsError, match="already exists"):
        run_export_results(run_dir)


def test_export_results_refuses_without_any_final_test(trained_run):
    tmp_path, run_dir, model, device = trained_run
    run_freeze(run_dir)
    with pytest.raises(ExportResultsError, match="eval/test"):
        run_export_results(run_dir)


@pytest.mark.slow
def test_export_results_refuses_rehearsal_only_run(trained_run):
    tmp_path, run_dir, model, device = trained_run
    run_freeze(run_dir)
    run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)
    with pytest.raises(ExportResultsError, match="eval/test"):
        run_export_results(run_dir)


def test_export_results_refuses_missing_run_dir(tmp_path):
    with pytest.raises(ExportResultsError, match="eval/test"):
        run_export_results(tmp_path / "nonexistent_run")
