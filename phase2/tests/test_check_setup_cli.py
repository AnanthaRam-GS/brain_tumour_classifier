"""cli/check_setup.py runs against the REAL repo (it needs the real
manifest/split/ROI cache and foundation lock, which fake_repo does not
fully replicate) -- these tests exercise it as-is plus a couple of
synthetic failure injections."""

import pytest

from btdl.cli.check_setup import CheckResult, run_check_setup


def test_run_check_setup_returns_all_named_checks():
    checks = run_check_setup()
    names = {c.name for c in checks}
    assert "versions (vs pyproject.toml pinned deps)" in names
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
