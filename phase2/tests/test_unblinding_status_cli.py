import dataclasses
import json
import shutil
import subprocess

import pytest

from btdl import config
from btdl.cli.final_test import run_final_test
from btdl.cli.freeze import run_freeze
from btdl.cli.register_frozen import run_register_frozen
from btdl.cli.unblinding_status import compute_status
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


def _duplicate_to_all_final_seeds(run_dir, model_name="tiny_cnn"):
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


def _register_tiny_cnn(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = trained_run
    run_freeze(run_dir)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)
    run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)
    _duplicate_to_all_final_seeds(run_dir)
    run_register_frozen("tiny_cnn")
    return tmp_path


def test_not_ready_with_no_registrations(fake_repo):
    tmp_path, manifest, specs = fake_repo
    status = compute_status()
    assert status["ready"] is False
    assert len(status["models"]) == 1  # conftest's study.yaml lists only tiny_cnn
    row = status["models"][0]
    assert row["model_name"] == "tiny_cnn"
    assert row["selection_present"] is False
    assert row["frozen_present"] is False
    assert row["ready"] is False


def test_not_ready_with_selection_but_no_registration(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = trained_run
    run_freeze(run_dir)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)

    status = compute_status()
    assert status["ready"] is False
    row = status["models"][0]
    assert row["selection_present"] is True
    assert row["frozen_present"] is False


@pytest.mark.slow
def test_ready_once_fully_registered(trained_run, monkeypatch):
    _register_tiny_cnn(trained_run, monkeypatch)
    status = compute_status()
    assert status["ready"] is True
    row = status["models"][0]
    assert row["frozen_present"] is True
    assert row["contract_matches"] is True
    assert row["seeds_registered"] == status["final_seeds"]
    assert row["ready"] is True


@pytest.mark.slow
def test_not_ready_if_contract_hash_stale(trained_run, monkeypatch):
    tmp_path = _register_tiny_cnn(trained_run, monkeypatch)
    frozen_path = tmp_path / "phase2" / "artifacts" / "frozen" / "tiny_cnn.json"
    with frozen_path.open() as handle:
        registration = json.load(handle)
    registration["contract_dir_sha256"] = "deadbeef" * 8
    with frozen_path.open("w") as handle:
        json.dump(registration, handle)

    status = compute_status()
    assert status["ready"] is False
    row = status["models"][0]
    assert row["contract_matches"] is False
    assert row["ready"] is False


@pytest.mark.slow
def test_not_ready_if_a_seed_is_missing(trained_run, monkeypatch):
    tmp_path = _register_tiny_cnn(trained_run, monkeypatch)
    frozen_path = tmp_path / "phase2" / "artifacts" / "frozen" / "tiny_cnn.json"
    with frozen_path.open() as handle:
        registration = json.load(handle)
    registration["seeds"] = registration["seeds"][:-1]  # drop one seed
    with frozen_path.open("w") as handle:
        json.dump(registration, handle)

    status = compute_status()
    assert status["ready"] is False
    row = status["models"][0]
    assert len(row["seeds_registered"]) == 2
    assert row["ready"] is False


def test_exit_code_matches_readiness(fake_repo):
    import sys

    from btdl.cli import unblinding_status

    tmp_path, manifest, specs = fake_repo
    with pytest.raises(SystemExit) as exc_info:
        unblinding_status.main()
    assert exc_info.value.code == 1
