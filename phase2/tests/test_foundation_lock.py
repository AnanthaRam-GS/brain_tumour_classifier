"""python -m btdl.cli.lock --write / --check (Part C).

test_foundation_lock_matches_current_tree runs against the REAL repo (no
fake_repo) -- it fails with a readable diff whenever a locked foundation
file changes without `python -m btdl.cli.lock --write` being re-run. The
rest of this file exercises compute_lock/write_lock/check_lock's diff
behavior against a small synthetic tree, independent of the real repo's
actual file set.
"""

from pathlib import Path

import pytest

from btdl.cli.lock import FoundationLockError, check_lock, compute_lock, locked_relpaths, write_lock


def test_foundation_lock_matches_current_tree():
    ok, diff = check_lock()
    assert ok, f"foundation lock is stale -- run `python -m btdl.cli.lock --write` and review the diff: {diff}"


def _make_minimal_locked_tree(tmp_path):
    contract_dir = tmp_path / "phase2" / "configs" / "contract"
    contract_dir.mkdir(parents=True)
    (contract_dir / "data.yaml").write_text("contract_version: '1.0.0'\n")

    btdl = tmp_path / "phase2" / "src" / "btdl"
    btdl.mkdir(parents=True)
    (btdl / "config.py").write_text("X = 1\n")
    (btdl / "contracts.py").write_text("Y = 1\n")
    (btdl / "runs.py").write_text("Z = 1\n")

    for sub in ("data", "preprocessing", "training", "evaluation", "cli"):
        d = btdl / sub
        d.mkdir()
        (d / "mod.py").write_text("A = 1\n")
        pycache = d / "__pycache__"
        pycache.mkdir()
        (pycache / "mod.cpython-312.pyc").write_bytes(b"\x00\x01")  # must be excluded

    models = btdl / "models"
    models.mkdir()
    (models / "registry.py").write_text("R = 1\n")
    (models / "model_spec.py").write_text("S = 1\n")
    (models / "torchvision_common.py").write_text("T = 1\n")
    (models / "catalog.py").write_text("C = 1\n")  # NOT locked
    (models / "tiny_cnn.py").write_text("N = 1\n")  # NOT locked

    return tmp_path


@pytest.fixture
def minimal_tree(tmp_path, monkeypatch):
    root = _make_minimal_locked_tree(tmp_path)
    monkeypatch.setattr("btdl.config.repo_root", lambda: root)
    return root


def test_locked_relpaths_includes_contract_and_foundation_files_excludes_pycache(minimal_tree):
    rels = locked_relpaths(minimal_tree)
    assert "phase2/configs/contract/data.yaml" in rels
    assert "phase2/src/btdl/config.py" in rels
    assert "phase2/src/btdl/contracts.py" in rels
    assert "phase2/src/btdl/runs.py" in rels
    assert "phase2/src/btdl/data/mod.py" in rels
    assert "phase2/src/btdl/cli/mod.py" in rels
    assert "phase2/src/btdl/models/registry.py" in rels
    assert "phase2/src/btdl/models/model_spec.py" in rels
    assert "phase2/src/btdl/models/torchvision_common.py" in rels
    assert not any("__pycache__" in rel for rel in rels)


def test_locked_relpaths_excludes_catalog_and_model_files(minimal_tree):
    rels = locked_relpaths(minimal_tree)
    assert "phase2/src/btdl/models/catalog.py" not in rels
    assert "phase2/src/btdl/models/tiny_cnn.py" not in rels


def test_write_then_check_is_ok(minimal_tree):
    write_lock(minimal_tree)
    ok, diff = check_lock(minimal_tree)
    assert ok is True
    assert diff == {"added": [], "removed": [], "changed": []}


def test_check_without_write_raises(minimal_tree):
    with pytest.raises(FoundationLockError):
        check_lock(minimal_tree)


def test_modifying_a_locked_file_is_detected(minimal_tree):
    write_lock(minimal_tree)
    (minimal_tree / "phase2" / "src" / "btdl" / "config.py").write_text("X = 2\n")
    ok, diff = check_lock(minimal_tree)
    assert ok is False
    assert diff["changed"] == ["phase2/src/btdl/config.py"]
    assert diff["added"] == []
    assert diff["removed"] == []


def test_adding_a_file_under_a_locked_subdir_is_detected(minimal_tree):
    write_lock(minimal_tree)
    (minimal_tree / "phase2" / "src" / "btdl" / "data" / "new_mod.py").write_text("B = 1\n")
    ok, diff = check_lock(minimal_tree)
    assert ok is False
    assert diff["added"] == ["phase2/src/btdl/data/new_mod.py"]


def test_removing_a_locked_file_is_detected(minimal_tree):
    write_lock(minimal_tree)
    (minimal_tree / "phase2" / "src" / "btdl" / "runs.py").unlink()
    ok, diff = check_lock(minimal_tree)
    assert ok is False
    assert diff["removed"] == ["phase2/src/btdl/runs.py"]


def test_modifying_catalog_py_does_not_affect_lock(minimal_tree):
    write_lock(minimal_tree)
    (minimal_tree / "phase2" / "src" / "btdl" / "models" / "catalog.py").write_text("C = 2\n")
    ok, diff = check_lock(minimal_tree)
    assert ok is True


def test_adding_a_new_model_module_does_not_affect_lock(minimal_tree):
    write_lock(minimal_tree)
    (minimal_tree / "phase2" / "src" / "btdl" / "models" / "resnet18.py").write_text("M = 1\n")
    ok, diff = check_lock(minimal_tree)
    assert ok is True


def test_compute_lock_has_version_and_contract_hash(minimal_tree):
    lock = compute_lock(minimal_tree)
    assert lock["lock_version"]
    assert lock["contract_dir_sha256"]
    assert isinstance(lock["files"], dict)
    assert len(lock["files"]) > 0


def test_write_lock_writes_tracked_json_path(minimal_tree):
    write_lock(minimal_tree)
    lock_path = minimal_tree / "phase2" / "artifacts" / "contract" / "foundation_lock.json"
    assert lock_path.is_file()
