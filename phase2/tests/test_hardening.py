"""Tests for Part A hardening: scoped dirty-tree check (A1), cache dir
override (A2), and pretrained-weights provenance (A3)."""

import dataclasses
import subprocess

import pytest

from btdl import config
from btdl.cli.train import run_train
from btdl.data.roi_cache import open_roi_cache
from btdl.models.registry import MODEL_REGISTRY, build_model
from btdl.training.device import select_device

DEVICE = select_device("cpu")


def _git(args, cwd):
    subprocess.run(["git"] + args, cwd=cwd, check=True, capture_output=True, text=True)


def _git_init_clean_repo(tmp_path):
    (tmp_path / ".gitignore").write_text("phase2/runs/\nphase2/cache/\n")
    _git(["init"], cwd=tmp_path)
    _git(["config", "user.email", "test@example.com"], cwd=tmp_path)
    _git(["config", "user.name", "Test User"], cwd=tmp_path)
    _git(["add", "-A"], cwd=tmp_path)
    _git(["commit", "-m", "init"], cwd=tmp_path)


# ---- A1: scoped dirty-tree definition --------------------------------------------------


def test_clean_tree_has_no_dirty_paths(fake_repo):
    tmp_path, manifest, specs = fake_repo
    _git_init_clean_repo(tmp_path)
    assert config.git_dirty_paths(tmp_path) == []
    assert config.git_is_dirty(tmp_path) is False


def test_untracked_file_outside_phase2_does_not_dirty(fake_repo):
    tmp_path, manifest, specs = fake_repo
    _git_init_clean_repo(tmp_path)
    (tmp_path / "some_reference.pdf").write_text("not mine to touch")
    assert config.git_dirty_paths(tmp_path) == []
    assert config.git_is_dirty(tmp_path) is False


def test_untracked_file_under_phase2_dirties(fake_repo):
    tmp_path, manifest, specs = fake_repo
    _git_init_clean_repo(tmp_path)
    (tmp_path / "phase2" / "stray.txt").write_text("stray")
    paths = config.git_dirty_paths(tmp_path)
    assert paths == ["phase2/stray.txt"]
    assert config.git_is_dirty(tmp_path) is True


def test_tracked_file_modified_outside_phase2_dirties(fake_repo):
    tmp_path, manifest, specs = fake_repo
    _git_init_clean_repo(tmp_path)
    split_csv = tmp_path / "split.csv"
    split_csv.write_text(split_csv.read_text() + "\n")
    paths = config.git_dirty_paths(tmp_path)
    assert "split.csv" in paths


def test_tracked_file_staged_or_deleted_dirties(fake_repo):
    tmp_path, manifest, specs = fake_repo
    _git_init_clean_repo(tmp_path)
    split_csv = tmp_path / "split.csv"
    split_csv.unlink()
    paths = config.git_dirty_paths(tmp_path)
    assert "split.csv" in paths


# ---- A2: cache dir override -------------------------------------------------------------


def test_cache_dir_defaults_to_phase2_cache_under_repo_root(monkeypatch, tmp_path):
    monkeypatch.delenv("BTDL_CACHE_DIR", raising=False)
    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)
    assert config.cache_dir() == tmp_path / "phase2" / "cache"


def test_cache_dir_override_must_exist(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist"
    monkeypatch.setenv("BTDL_CACHE_DIR", str(missing))
    with pytest.raises(ValueError, match="does not exist"):
        config.cache_dir()


def test_cache_dir_override_used_when_set(monkeypatch, tmp_path):
    override = tmp_path / "custom_cache"
    override.mkdir()
    monkeypatch.setenv("BTDL_CACHE_DIR", str(override))
    assert config.cache_dir() == override.resolve()


def test_build_cache_and_open_roi_cache_honour_btdl_cache_dir(monkeypatch, tmp_path_factory, fake_repo_factory):
    custom_cache = tmp_path_factory.mktemp("custom_cache")
    monkeypatch.setenv("BTDL_CACHE_DIR", str(custom_cache))

    tmp_path, manifest, specs = fake_repo_factory()

    assert (custom_cache / "roi224_v1.npy").is_file()
    assert not (tmp_path / "phase2" / "cache" / "roi224_v1.npy").exists()

    cache = open_roi_cache()
    assert len(cache) == len(manifest)


# ---- A3: pretrained-weights provenance ---------------------------------------------------


def test_tiny_cnn_metadata_records_no_pretrained_weights(git_fake_repo):
    tmp_path, manifest, specs = git_fake_repo
    training_cfg = config.load_contract("training")
    result = run_train("tiny_cnn", training_cfg["lr_grid"][0], training_cfg["grid_seed"], smoke=True, device=DEVICE)
    import json

    with (result.run_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    assert metadata["weights_id"] is None
    assert metadata["pretrained_weights_loaded"] is False


def test_train_cli_uses_pretrained_exactly_when_spec_has_weights_id(git_fake_repo, monkeypatch):
    # tiny_cnn has weights_id=None; temporarily give it a fake weights_id to
    # confirm run_train() flips pretrained=True to match, without needing a
    # real torchvision download.
    tmp_path, manifest, specs = git_fake_repo
    original_spec = MODEL_REGISTRY["tiny_cnn"]
    fake_spec = dataclasses.replace(original_spec, weights_id="FAKE_WEIGHTS_V1")
    monkeypatch.setitem(MODEL_REGISTRY, "tiny_cnn", fake_spec)

    captured = {}
    original_build_model = build_model

    def _spy_build_model(name, *, pretrained=True):
        captured["pretrained"] = pretrained
        return original_build_model(name, pretrained=pretrained)

    monkeypatch.setattr("btdl.cli.train.build_model", _spy_build_model)

    training_cfg = config.load_contract("training")
    run_train("tiny_cnn", training_cfg["lr_grid"][0], training_cfg["grid_seed"], smoke=True, device=DEVICE)

    assert captured["pretrained"] is True
