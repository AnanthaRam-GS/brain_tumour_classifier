import json
import subprocess

import pytest

from btdl import config
from btdl.cli.train import TrainCliError, run_train
from btdl.training.device import select_device

DEVICE = select_device("cpu")


def test_rejects_lr_not_in_grid(fake_repo):
    # lr/seed protocol is validated before any git check, so a non-git
    # fake_repo is fine here.
    with pytest.raises(TrainCliError, match="lr_grid"):
        run_train("tiny_cnn", 0.5, 42, device=DEVICE)


def test_rejects_unknown_seed(fake_repo):
    with pytest.raises(TrainCliError, match="final_seeds"):
        run_train("tiny_cnn", 1e-4, 999, device=DEVICE)


def test_final_seed_without_selection_file_refused(fake_repo):
    training_cfg = config.load_contract("training")
    grid_seed = training_cfg["grid_seed"]
    non_grid_final_seed = next(s for s in training_cfg["final_seeds"] if s != grid_seed)
    with pytest.raises(TrainCliError, match="selection"):
        run_train("tiny_cnn", training_cfg["lr_grid"][0], non_grid_final_seed, device=DEVICE)


def test_final_seed_with_mismatched_lr_refused(fake_repo):
    tmp_path, manifest, specs = fake_repo
    training_cfg = config.load_contract("training")
    grid_seed = training_cfg["grid_seed"]
    non_grid_final_seed = next(s for s in training_cfg["final_seeds"] if s != grid_seed)
    selected_lr = training_cfg["lr_grid"][1]
    other_lr = training_cfg["lr_grid"][0]

    selection_path = tmp_path / "phase2" / "artifacts" / "selection" / "tiny_cnn.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps({"selected_lr": selected_lr}))

    with pytest.raises(TrainCliError, match="does not match"):
        run_train("tiny_cnn", other_lr, non_grid_final_seed, device=DEVICE)


@pytest.mark.slow
def test_grid_seed_allowed_for_any_grid_lr(git_fake_repo):
    tmp_path, manifest, specs = git_fake_repo
    training_cfg = config.load_contract("training")
    grid_seed = training_cfg["grid_seed"]
    for lr in training_cfg["lr_grid"]:
        result = run_train("tiny_cnn", lr, grid_seed, smoke=True, device=DEVICE)
        assert result.run_dir.exists()


@pytest.mark.slow
def test_final_seed_with_selection_file_and_matching_lr_allowed(git_fake_repo):
    tmp_path, manifest, specs = git_fake_repo
    training_cfg = config.load_contract("training")
    grid_seed = training_cfg["grid_seed"]
    non_grid_final_seed = next(s for s in training_cfg["final_seeds"] if s != grid_seed)
    selected_lr = training_cfg["lr_grid"][1]

    selection_path = tmp_path / "phase2" / "artifacts" / "selection" / "tiny_cnn.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps({"selected_lr": selected_lr}))
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add selection"], cwd=tmp_path, check=True, capture_output=True)

    result = run_train("tiny_cnn", selected_lr, non_grid_final_seed, smoke=True, device=DEVICE)
    assert result.run_dir.exists()


def test_dirty_tree_refused_without_allow_dirty(git_fake_repo):
    tmp_path, manifest, specs = git_fake_repo
    (tmp_path / "untracked.txt").write_text("dirty")  # never committed -> tree is dirty

    training_cfg = config.load_contract("training")
    with pytest.raises(TrainCliError, match="dirty"):
        run_train("tiny_cnn", training_cfg["lr_grid"][0], training_cfg["grid_seed"], device=DEVICE)


@pytest.mark.slow
def test_dirty_tree_allowed_with_allow_dirty_and_recorded(git_fake_repo):
    tmp_path, manifest, specs = git_fake_repo
    (tmp_path / "untracked.txt").write_text("dirty")

    training_cfg = config.load_contract("training")
    result = run_train(
        "tiny_cnn", training_cfg["lr_grid"][0], training_cfg["grid_seed"], smoke=True, allow_dirty=True, device=DEVICE
    )
    with (result.run_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    assert metadata["git_dirty"] is True


@pytest.mark.slow
def test_smoke_overrides_max_epochs_and_is_non_conformant(git_fake_repo):
    training_cfg = config.load_contract("training")
    result = run_train("tiny_cnn", training_cfg["lr_grid"][0], training_cfg["grid_seed"], smoke=True, device=DEVICE)
    with (result.run_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    assert metadata["contract_conformant"] is False
    deviation_keys = {d["key"] for d in metadata["contract_deviations"]}
    assert "max_epochs" in deviation_keys
    assert metadata["effective_cfg"]["max_epochs"] == 2


@pytest.mark.slow
def test_smoke_run_dir_under_smoke_prefix(git_fake_repo):
    training_cfg = config.load_contract("training")
    result = run_train("tiny_cnn", training_cfg["lr_grid"][0], training_cfg["grid_seed"], smoke=True, device=DEVICE)
    assert "_smoke" in result.run_dir.parts


@pytest.mark.slow
def test_val_evaluation_runs_automatically_after_fit(git_fake_repo):
    training_cfg = config.load_contract("training")
    result = run_train("tiny_cnn", training_cfg["lr_grid"][0], training_cfg["grid_seed"], smoke=True, device=DEVICE)
    assert (result.run_dir / "eval" / "val" / "metrics.json").is_file()
