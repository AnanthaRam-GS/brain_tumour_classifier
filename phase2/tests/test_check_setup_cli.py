"""cli/check_setup.py runs against the REAL repo (it needs the real
manifest/split/ROI cache and foundation lock, which fake_repo does not
fully replicate) -- these tests exercise it as-is plus synthetic version-
policy and failure injections."""

import shutil

import pytest

from btdl import config
from btdl.cli.check_setup import CORE_PACKAGES, CheckResult, _check_versions, run_check_setup

VERSIONS_CHECK_NAME = "versions (CORE packages vs pyproject.toml pins; others vs requirements.lock)"

# Matches phase2/pyproject.toml's pinned dependencies exactly.
_REAL_CORE_VERSIONS = {
    "torch": "2.14.1",
    "torchvision": "0.29.1",
    "numpy": "2.5.3",
    "scikit-learn": "1.9.1",
    "scipy": "1.18.1",
    "h5py": "3.16.0",
    "pyyaml": "6.0.3",
    "pandas": "3.0.6",
    "matplotlib": "3.11.2",
}


def _make_versions_tree(tmp_path, lock_body: str):
    """Copies the REAL pyproject.toml (so pyproject pins are authentic) into
    a tmp tree, with a synthetic requirements.lock."""

    real_pyproject = config.repo_root() / "phase2" / "pyproject.toml"
    phase2_dir = tmp_path / "phase2"
    phase2_dir.mkdir(parents=True)
    shutil.copy(real_pyproject, phase2_dir / "pyproject.toml")
    (phase2_dir / "requirements.lock").write_text(lock_body)
    return tmp_path


def test_run_check_setup_returns_all_named_checks():
    checks = run_check_setup()
    names = {c.name for c in checks}
    assert VERSIONS_CHECK_NAME in names
    assert "btdl import location" in names
    assert "device" in names
    assert "contracts load" in names
    assert "foundation lock" in names
    assert "manifest + split" in names
    assert "ROI cache" in names
    assert "working tree clean" in names
    assert "no Phase 1 `src` import" in names
    assert "model registry" in names
    for check in checks:
        assert check.status in ("PASS", "FAIL", "WARN")


def test_all_checks_pass_on_clean_real_repo():
    checks = run_check_setup()
    failing = [c for c in checks if c.status == "FAIL"]
    assert failing == [], f"unexpected FAILs on a clean real repo: {failing}"


def test_working_tree_warn_is_a_warn_not_a_fail(tmp_path, monkeypatch):
    # Simulate a dirty tree by pointing git_dirty_paths at a non-repo dir
    # (git status fails -> caught and reported as WARN, never FAIL).
    import subprocess

    from btdl import config

    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["git", "status"])

    monkeypatch.setattr(config, "git_dirty_paths", _raise)
    checks = run_check_setup()
    tree_check = next(c for c in checks if c.name == "working tree clean")
    assert tree_check.status == "WARN"


def test_check_result_is_a_simple_record():
    result = CheckResult("x", "PASS", "detail")
    assert result.name == "x"
    assert result.status == "PASS"
    assert result.detail == "detail"


def test_requirements_lock_is_tracked_and_present():
    lock_path = config.repo_root() / "phase2" / "requirements.lock"
    assert lock_path.is_file()


# ---- B3: version policy -------------------------------------------------------------------


def test_core_package_mismatch_vs_pyproject_is_fail(tmp_path, monkeypatch):
    tree = _make_versions_tree(tmp_path, "# Python: 3.12.13\ntorch==2.14.1\n")
    monkeypatch.setattr("btdl.cli.check_setup.config.repo_root", lambda: tree)
    monkeypatch.setattr(
        "btdl.cli.check_setup._installed_version",
        lambda name: "9.9.9" if name == "torch" else _REAL_CORE_VERSIONS.get(name, "1.0.0"),
    )
    result = _check_versions()
    assert result.status == "FAIL"
    assert "torch 9.9.9 != pyproject pin 2.14.1" in result.detail


def test_core_package_missing_is_fail(tmp_path, monkeypatch):
    tree = _make_versions_tree(tmp_path, "# Python: 3.12.13\n")
    monkeypatch.setattr("btdl.cli.check_setup.config.repo_root", lambda: tree)
    monkeypatch.setattr(
        "btdl.cli.check_setup._installed_version",
        lambda name: None if name == "numpy" else _REAL_CORE_VERSIONS.get(name, "1.0.0"),
    )
    result = _check_versions()
    assert result.status == "FAIL"
    assert "numpy not installed" in result.detail


def test_noncore_package_mismatch_vs_lock_is_warn_not_fail(tmp_path, monkeypatch):
    lock_body = "# Python: 3.12.13\n" + "\n".join(
        f"{name}=={version}" for name, version in _REAL_CORE_VERSIONS.items()
    ) + "\nsome_extra_pkg==1.0.0\n"
    tree = _make_versions_tree(tmp_path, lock_body)
    monkeypatch.setattr("btdl.cli.check_setup.config.repo_root", lambda: tree)
    monkeypatch.setattr(
        "btdl.cli.check_setup._installed_version",
        lambda name: "2.0.0" if name == "some_extra_pkg" else _REAL_CORE_VERSIONS.get(name, "1.0.0"),
    )
    result = _check_versions()
    assert result.status == "WARN"
    assert "some_extra_pkg 2.0.0 != requirements.lock 1.0.0" in result.detail


def test_python_minor_mismatch_vs_lock_header_is_warn_not_fail(tmp_path, monkeypatch):
    lock_body = "# Python: 3.10.0\n" + "\n".join(
        f"{name}=={version}" for name, version in _REAL_CORE_VERSIONS.items()
    ) + "\n"
    tree = _make_versions_tree(tmp_path, lock_body)
    monkeypatch.setattr("btdl.cli.check_setup.config.repo_root", lambda: tree)
    monkeypatch.setattr(
        "btdl.cli.check_setup._installed_version", lambda name: _REAL_CORE_VERSIONS.get(name, "1.0.0")
    )
    result = _check_versions()
    assert result.status == "WARN"
    assert "python" in result.detail.lower()


def test_all_core_and_python_matching_is_pass(tmp_path, monkeypatch):
    import sys

    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}.0"
    lock_body = f"# Python: {actual_python}\n" + "\n".join(
        f"{name}=={version}" for name, version in _REAL_CORE_VERSIONS.items()
    ) + "\n"
    tree = _make_versions_tree(tmp_path, lock_body)
    monkeypatch.setattr("btdl.cli.check_setup.config.repo_root", lambda: tree)
    monkeypatch.setattr(
        "btdl.cli.check_setup._installed_version", lambda name: _REAL_CORE_VERSIONS.get(name, "1.0.0")
    )
    result = _check_versions()
    assert result.status == "PASS"


def test_missing_requirements_lock_is_fail(tmp_path, monkeypatch):
    phase2_dir = tmp_path / "phase2"
    phase2_dir.mkdir(parents=True)
    shutil.copy(config.repo_root() / "phase2" / "pyproject.toml", phase2_dir / "pyproject.toml")
    monkeypatch.setattr("btdl.cli.check_setup.config.repo_root", lambda: tmp_path)
    result = _check_versions()
    assert result.status == "FAIL"
    assert "requirements.lock not found" in result.detail


def test_core_packages_set_matches_spec():
    assert CORE_PACKAGES == frozenset(
        {"torch", "torchvision", "numpy", "scikit-learn", "scipy", "h5py", "pyyaml", "pandas", "matplotlib"}
    )
