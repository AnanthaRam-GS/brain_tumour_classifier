import dataclasses
import json
import shutil
import subprocess

import pytest

from btdl import config
from btdl.cli.efficiency import run_efficiency
from btdl.cli.evaluate import run_evaluate
from btdl.cli.export_results import ExportResultsError, run_export_results
from btdl.cli.final_test import run_final_test
from btdl.cli.freeze import run_freeze
from btdl.cli.register_frozen import run_register_frozen
from btdl.evaluation.results import validate_evaluation_dir
from btdl.models.registry import MODEL_REGISTRY
from btdl.runs import run_dir_for


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


def _make_unblinding_ready(tmp_path, run_dir, device, model_name="tiny_cnn"):
    """D18: see test_final_test_cli.py's identical helper -- duplicates the
    single trained run to the other final_seeds' run_dirs so
    register_frozen (which requires every final_seed) can succeed without
    training two more models."""

    rehearsal_dir = run_dir / "eval" / "rehearsal_val"
    if not rehearsal_dir.is_dir():
        run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)

    with (run_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    lr = metadata["lr"]
    own_seed = metadata["seed"]

    training_cfg = config.load_contract("training")
    for seed in training_cfg["final_seeds"]:
        if seed == own_seed:
            continue
        target_dir = run_dir_for(model_name, lr, seed, smoke=False)
        if target_dir.exists():
            continue
        shutil.copytree(run_dir, target_dir)
        meta_path = target_dir / "metadata.json"
        with meta_path.open() as handle:
            meta = json.load(handle)
        meta["seed"] = seed
        with meta_path.open("w") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

    run_register_frozen(model_name)

    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "register frozen"], cwd=tmp_path, check=True, capture_output=True)


def _final_tested_run(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = trained_run
    # trained_run calls fit() directly (not via cli.train.run_train), so
    # eval/val/ -- normally written automatically by run_train -- doesn't
    # exist yet; export_results requires it.
    run_evaluate(run_dir, split="val", device=device, batch_size=2)
    run_freeze(run_dir)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)
    _make_unblinding_ready(tmp_path, run_dir, device)
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
def test_export_results_refuses_sha_mismatch_vs_registration(trained_run, monkeypatch):
    # D18: export_results re-checks the frozen registration independently
    # of final_test's own (earlier) check.
    tmp_path, run_dir = _final_tested_run(trained_run, monkeypatch)
    with (run_dir / "metadata.json").open() as handle:
        own_seed = json.load(handle)["seed"]

    frozen_registration_path = tmp_path / "phase2" / "artifacts" / "frozen" / "tiny_cnn.json"
    with frozen_registration_path.open() as handle:
        registration = json.load(handle)
    for entry in registration["seeds"]:
        if entry["seed"] == own_seed:
            entry["best_pt_sha256"] = "deadbeef" * 8
    with frozen_registration_path.open("w") as handle:
        json.dump(registration, handle)
    # run_dir is gitignored; artifacts/frozen/ is tracked, but export_results
    # doesn't check git cleanliness itself, so no commit is required here.

    with pytest.raises(ExportResultsError, match="does not match the frozen"):
        run_export_results(run_dir)


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
