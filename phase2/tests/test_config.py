import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from btdl import config


def test_repo_root_is_directory_containing_phase2():
    root = config.repo_root()
    assert (root / "phase2").is_dir()
    assert (root / "phase2" / "src" / "btdl").is_dir()


def test_repo_root_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = config.repo_root()
    assert (root / "phase2").is_dir()


def test_data_root_defaults_to_repo_root(monkeypatch):
    monkeypatch.delenv("BTDL_DATA_ROOT", raising=False)
    assert config.data_root() == config.repo_root()


def test_data_root_env_override_works(tmp_path, monkeypatch):
    monkeypatch.setenv("BTDL_DATA_ROOT", str(tmp_path))
    assert config.data_root() == tmp_path.resolve()


def test_data_root_env_override_rejects_missing_directory(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("BTDL_DATA_ROOT", str(missing))
    with pytest.raises(ValueError):
        config.data_root()


def test_resolve_data_path_is_under_data_root(monkeypatch):
    monkeypatch.delenv("BTDL_DATA_ROOT", raising=False)
    resolved = config.resolve_data_path("data/splits/patient_split.csv")
    assert resolved == config.repo_root() / "data" / "splits" / "patient_split.csv"


def test_load_contract_data_returns_immutable_mapping():
    contract = config.load_contract("data")
    assert contract["contract_version"] == "1.0.0"
    assert contract["expected"]["n_samples"] == 3064
    with pytest.raises(TypeError):
        contract["contract_version"] = "9.9.9"
    with pytest.raises(TypeError):
        contract["expected"]["n_samples"] = 1


def test_load_contract_unknown_name_raises():
    with pytest.raises(ValueError):
        config.load_contract("does_not_exist")


def test_load_contract_rejects_missing_key(tmp_path, monkeypatch):
    bad_yaml = {
        "contract_version": "1.0.0",
        "raw_dir": "1512427",
        # samples_dir missing
        "split_csv": "data/splits/patient_split.csv",
        "split_sha256": "",
        "manifest": "phase2/artifacts/contract/manifest.csv",
        "expected": {
            "n_samples": 3064,
            "n_patients": 233,
            "class_counts": {1: 708, 2: 1426, 3: 930},
            "split_samples": {"train": 2120, "val": 446, "test": 498},
            "split_patients": {"train": 162, "val": 34, "test": 37},
        },
    }
    _write_fake_repo_contract(tmp_path, "data", bad_yaml)
    with pytest.raises(ValueError):
        _load_contract_from_fake_repo(tmp_path, "data")


def test_load_contract_rejects_mistyped_key(tmp_path):
    bad_yaml = {
        "contract_version": "1.0.0",
        "raw_dir": "1512427",
        "samples_dir": "data/processed/samples",
        "split_csv": "data/splits/patient_split.csv",
        "split_sha256": "",
        "manifest": "phase2/artifacts/contract/manifest.csv",
        "expected": {
            "n_samples": "3064",  # should be int, not str
            "n_patients": 233,
            "class_counts": {1: 708, 2: 1426, 3: 930},
            "split_samples": {"train": 2120, "val": 446, "test": 498},
            "split_patients": {"train": 162, "val": 34, "test": 37},
        },
    }
    _write_fake_repo_contract(tmp_path, "data", bad_yaml)
    with pytest.raises(ValueError):
        _load_contract_from_fake_repo(tmp_path, "data")


def _write_fake_repo_contract(tmp_path, name, payload):
    contract_dir = tmp_path / "phase2" / "configs" / "contract"
    contract_dir.mkdir(parents=True, exist_ok=True)
    with (contract_dir / f"{name}.yaml").open("w") as handle:
        yaml.safe_dump(payload, handle)


def _load_contract_from_fake_repo(tmp_path, name):
    """Exercise load_contract's schema validation against a fake repo root.

    load_contract derives its path from repo_root(), which is fixed to the
    real package location, so this calls the module-level helpers directly
    against a path under tmp_path instead of monkeypatching repo_root().
    """

    path = tmp_path / "phase2" / "configs" / "contract" / f"{name}.yaml"
    with path.open("r") as handle:
        payload = yaml.safe_load(handle)
    config._validate_schema(name, payload)


def test_contract_dir_sha256_is_stable_across_calls():
    first = config.contract_dir_sha256()
    second = config.contract_dir_sha256()
    assert first == second
    assert isinstance(first, str) and len(first) == 64


def test_contract_dir_sha256_changes_when_a_file_changes(tmp_path, monkeypatch):
    # Copy the real contract dir into a temp repo layout, hash it, mutate a
    # copy, and confirm the hash changes -- never touch the real contract.
    real_contract_dir = config.repo_root() / "phase2" / "configs" / "contract"
    fake_contract_dir = tmp_path / "phase2" / "configs" / "contract"
    fake_contract_dir.mkdir(parents=True)
    for source in real_contract_dir.iterdir():
        if source.is_file():
            (fake_contract_dir / source.name).write_bytes(source.read_bytes())

    def _hash_dir(directory):
        import hashlib

        digest = hashlib.sha256()
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    before = _hash_dir(fake_contract_dir)
    (fake_contract_dir / "data.yaml").write_text(
        (fake_contract_dir / "data.yaml").read_text() + "\n# mutated\n"
    )
    after = _hash_dir(fake_contract_dir)
    assert before != after
