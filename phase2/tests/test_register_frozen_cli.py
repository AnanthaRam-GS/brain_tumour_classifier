import dataclasses
import json
import shutil
import subprocess

import pytest

from btdl import config
from btdl.cli.final_test import run_final_test
from btdl.cli.freeze import run_freeze
from btdl.cli.register_frozen import FROZEN_REGISTRATION_DIR_RELPATH, RegisterFrozenError, run_register_frozen
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


@pytest.fixture
def frozen_and_rehearsed(trained_run, monkeypatch):
    """A full set of final-seed runs (via duplication), frozen, rehearsed,
    with a matching selection file -- ready to be registered as frozen."""

    tmp_path, run_dir, model, device = trained_run
    run_freeze(run_dir)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)
    run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)
    _duplicate_to_all_final_seeds(run_dir)
    # run_dir lives under the gitignored phase2/runs/ -- nothing new to
    # commit here (the selection file was already committed above).
    return tmp_path, run_dir, device


@pytest.mark.slow
def test_register_frozen_success(frozen_and_rehearsed):
    tmp_path, run_dir, device = frozen_and_rehearsed
    registration = run_register_frozen("tiny_cnn")

    assert registration["model_name"] == "tiny_cnn"
    assert registration["selected_lr"] == config.load_contract("training")["lr_grid"][0]
    assert registration["contract_dir_sha256"] == config.contract_dir_sha256()
    assert len(registration["seeds"]) == 3
    assert {entry["seed"] for entry in registration["seeds"]} == set(
        config.load_contract("training")["final_seeds"]
    )
    for entry in registration["seeds"]:
        assert entry["best_pt_sha256"]
        assert entry["run_dir"]

    out_path = tmp_path / FROZEN_REGISTRATION_DIR_RELPATH / "tiny_cnn.json"
    assert out_path.is_file()


@pytest.mark.slow
def test_registration_file_has_no_test_information(frozen_and_rehearsed):
    tmp_path, run_dir, device = frozen_and_rehearsed
    registration = run_register_frozen("tiny_cnn")
    serialized = json.dumps(registration).lower()
    assert "eval/test" not in serialized
    assert "test_access_log" not in serialized
    assert "/test/" not in serialized


@pytest.mark.slow
def test_register_frozen_refuses_overwrite(frozen_and_rehearsed):
    tmp_path, run_dir, device = frozen_and_rehearsed
    run_register_frozen("tiny_cnn")
    with pytest.raises(RegisterFrozenError, match="already exists"):
        run_register_frozen("tiny_cnn")


def test_register_frozen_refuses_reference_only_model(fake_repo):
    # tiny_cnn is reference_only=True by default, and IS in conftest's
    # fake study.yaml -- no training needed to hit this gate.
    tmp_path, manifest, specs = fake_repo
    with pytest.raises(RegisterFrozenError, match="reference_only"):
        run_register_frozen("tiny_cnn")


def test_register_frozen_refuses_model_not_in_study(fake_repo, monkeypatch):
    tmp_path, manifest, specs = fake_repo
    _enable_real_mode(monkeypatch)
    study_path = tmp_path / "phase2" / "configs" / "study.yaml"
    study_path.write_text("study_version: '1.0.0'\nmodels: []\nbaseline_reference: x\n")
    with pytest.raises(RegisterFrozenError, match="study.yaml"):
        run_register_frozen("tiny_cnn")


def test_register_frozen_refuses_unknown_model(fake_repo):
    # Not in study.yaml's models list either -- that gate fires first.
    tmp_path, manifest, specs = fake_repo
    with pytest.raises(RegisterFrozenError, match="study.yaml"):
        run_register_frozen("definitely_not_a_real_model")


def test_register_frozen_refuses_unknown_model_even_if_in_study(fake_repo):
    tmp_path, manifest, specs = fake_repo
    study_path = tmp_path / "phase2" / "configs" / "study.yaml"
    study_path.write_text(
        "study_version: '1.0.0'\nmodels:\n  - definitely_not_a_real_model\nbaseline_reference: x\n"
    )
    with pytest.raises(ValueError, match="unknown model"):
        run_register_frozen("definitely_not_a_real_model")


@pytest.mark.slow
def test_register_frozen_refuses_missing_selection(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = trained_run
    run_freeze(run_dir)
    _enable_real_mode(monkeypatch)
    with pytest.raises(RegisterFrozenError, match="selection"):
        run_register_frozen("tiny_cnn")


@pytest.mark.slow
def test_register_frozen_refuses_when_a_seed_is_not_frozen(frozen_and_rehearsed):
    tmp_path, run_dir, device = frozen_and_rehearsed
    # Remove FROZEN.json from one of the duplicated seeds' run dirs.
    with (run_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    training_cfg = config.load_contract("training")
    other_seed = next(s for s in training_cfg["final_seeds"] if s != metadata["seed"])
    other_dir = run_dir_for("tiny_cnn", metadata["lr"], other_seed, smoke=False)
    (other_dir / "FROZEN.json").unlink()

    with pytest.raises(RegisterFrozenError, match="not frozen"):
        run_register_frozen("tiny_cnn")


@pytest.mark.slow
def test_register_frozen_refuses_sha_mismatch(frozen_and_rehearsed):
    tmp_path, run_dir, device = frozen_and_rehearsed
    frozen_path = run_dir / "FROZEN.json"
    with frozen_path.open() as handle:
        frozen = json.load(handle)
    frozen["best_pt_sha256"] = "deadbeef" * 8
    with frozen_path.open("w") as handle:
        json.dump(frozen, handle)

    with pytest.raises(RegisterFrozenError, match="sha256"):
        run_register_frozen("tiny_cnn")


@pytest.mark.slow
def test_register_frozen_refuses_non_conformant_seed(frozen_and_rehearsed):
    tmp_path, run_dir, device = frozen_and_rehearsed
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    metadata["contract_conformant"] = False
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(RegisterFrozenError, match="contract_conformant"):
        run_register_frozen("tiny_cnn")


@pytest.mark.slow
def test_register_frozen_refuses_dirty_seed(frozen_and_rehearsed):
    tmp_path, run_dir, device = frozen_and_rehearsed
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    metadata["git_dirty"] = True
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(RegisterFrozenError, match="git_dirty"):
        run_register_frozen("tiny_cnn")


@pytest.mark.slow
def test_register_frozen_refuses_missing_rehearsal(frozen_and_rehearsed):
    tmp_path, run_dir, device = frozen_and_rehearsed
    shutil.rmtree(run_dir / "eval" / "rehearsal_val")

    with pytest.raises(RegisterFrozenError, match="rehearsal_val"):
        run_register_frozen("tiny_cnn")
