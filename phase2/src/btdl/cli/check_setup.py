"""python -m btdl.cli.check_setup

Prints a PASS/FAIL/WARN checklist verifying a fresh clone is ready to
train, and exits non-zero if ANY check FAILs (a WARN does not fail the
exit code). Checks: python/torch/torchvision versions against
pyproject.toml's pinned dependencies (this repo has no separate
requirements.lock file -- pyproject.toml's `==`-pinned deps serve as the
lock); btdl's import location; the selected device; that every contract
file loads; the foundation lock (cli/lock.py --check); the manifest and
split, hash-verified; the ROI cache at its resolved location (honouring
BTDL_CACHE_DIR), with a FULL sha256 verify; working-tree cleanliness (a
WARN, not a FAIL -- see A1's scoped dirty-tree definition); that btdl
never imports Phase 1's `src` package; and the model registry's contents.
"""

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision

from btdl import config
from btdl.cli.lock import FoundationLockError, check_lock
from btdl.data.roi_cache import RoiCacheError, open_roi_cache
from btdl.data.split import SplitIntegrityError, load_split
from btdl.models.registry import get_spec, list_models
from btdl.training.device import describe_device, select_device


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "FAIL" | "WARN"
    detail: str = ""


def _check_versions() -> CheckResult:
    pyproject_path = config.repo_root() / "phase2" / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)

    pinned = {}
    for dep in pyproject["project"]["dependencies"]:
        if "==" in dep:
            name, version = dep.split("==", 1)
            pinned[name.strip().lower()] = version.strip()

    problems = []

    expected_python = pyproject["project"]["requires-python"].replace("==", "").replace(".*", "")
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        problems.append(f"python {actual_python} != expected {expected_python}")

    expected_torch = pinned.get("torch")
    if expected_torch and torch.__version__ != expected_torch:
        problems.append(f"torch {torch.__version__} != expected {expected_torch}")

    expected_torchvision = pinned.get("torchvision")
    if expected_torchvision and torchvision.__version__ != expected_torchvision:
        problems.append(f"torchvision {torchvision.__version__} != expected {expected_torchvision}")

    detail = f"python={actual_python} torch=={torch.__version__} torchvision=={torchvision.__version__}"
    if problems:
        return CheckResult("versions (vs pyproject.toml pinned deps)", "FAIL", detail + "; " + "; ".join(problems))
    return CheckResult("versions (vs pyproject.toml pinned deps)", "PASS", detail)


def _check_btdl_import_location() -> CheckResult:
    import btdl

    package_path = str(Path(btdl.__file__).resolve())
    expected_prefix = str((config.repo_root() / "phase2" / "src" / "btdl").resolve())
    if package_path.startswith(expected_prefix):
        return CheckResult("btdl import location", "PASS", package_path)
    return CheckResult(
        "btdl import location", "FAIL", f"btdl imported from {package_path}, expected under {expected_prefix}"
    )


def _check_device() -> CheckResult:
    device = select_device("auto")
    return CheckResult("device", "PASS", json.dumps(describe_device(device)))


def _check_contracts() -> CheckResult:
    try:
        for name in ("data", "input", "augmentation", "training", "evaluation"):
            config.load_contract(name)
        return CheckResult("contracts load", "PASS", "data, input, augmentation, training, evaluation")
    except Exception as exc:
        return CheckResult("contracts load", "FAIL", str(exc))


def _check_foundation_lock() -> CheckResult:
    try:
        ok, diff = check_lock()
    except FoundationLockError as exc:
        return CheckResult("foundation lock", "FAIL", str(exc))
    if ok:
        return CheckResult("foundation lock", "PASS", "")
    return CheckResult("foundation lock", "FAIL", json.dumps(diff))


def _check_manifest_and_split() -> CheckResult:
    try:
        data_cfg = config.load_contract("data")
        manifest_path = config.repo_root() / data_cfg["manifest"]
        if not manifest_path.is_file():
            return CheckResult("manifest + split", "FAIL", f"manifest not found: {manifest_path}")

        meta_path = config.repo_root() / "phase2" / "artifacts" / "contract" / "roi_cache_meta.json"
        if meta_path.is_file():
            with meta_path.open() as handle:
                meta = json.load(handle)
            actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if actual != meta.get("manifest_sha256"):
                return CheckResult(
                    "manifest + split", "FAIL", f"manifest sha256 {actual} != roi_cache_meta.json's recorded hash"
                )

        load_split()  # verifies split_csv's sha256 against the data contract; raises on mismatch
        return CheckResult("manifest + split", "PASS", str(manifest_path))
    except SplitIntegrityError as exc:
        return CheckResult("manifest + split", "FAIL", str(exc))
    except Exception as exc:
        return CheckResult("manifest + split", "FAIL", str(exc))


def _check_roi_cache() -> CheckResult:
    try:
        open_roi_cache(verify=True)  # verify=True -> full sha256 check of the .npy array
        return CheckResult("ROI cache", "PASS", f"resolved at {config.cache_dir()}")
    except RoiCacheError as exc:
        return CheckResult("ROI cache", "FAIL", str(exc))


def _check_working_tree() -> CheckResult:
    try:
        dirty_paths = config.git_dirty_paths()
    except Exception as exc:
        return CheckResult("working tree clean", "WARN", f"could not determine git status: {exc}")
    if not dirty_paths:
        return CheckResult("working tree clean", "PASS", "")
    return CheckResult("working tree clean", "WARN", f"dirty paths: {dirty_paths}")


def _check_no_phase1_import() -> CheckResult:
    btdl_src_dir = str(config.repo_root() / "phase2" / "src")
    result = subprocess.run(
        [sys.executable, "-c", "import btdl; import sys; assert 'src' not in sys.modules"],
        env={**os.environ, "PYTHONPATH": btdl_src_dir},
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return CheckResult("no Phase 1 `src` import", "PASS", "")
    return CheckResult("no Phase 1 `src` import", "FAIL", result.stderr.strip())


def _check_registry() -> CheckResult:
    names = list_models()
    if not names:
        return CheckResult("model registry", "WARN", "no models registered")
    detail = ", ".join(f"{name}(reference_only={get_spec(name).reference_only})" for name in names)
    return CheckResult("model registry", "PASS", detail)


def run_check_setup() -> list:
    return [
        _check_versions(),
        _check_btdl_import_location(),
        _check_device(),
        _check_contracts(),
        _check_foundation_lock(),
        _check_manifest_and_split(),
        _check_roi_cache(),
        _check_working_tree(),
        _check_no_phase1_import(),
        _check_registry(),
    ]


def main():
    checks = run_check_setup()
    any_fail = False
    for check in checks:
        line = f"[{check.status}] {check.name}"
        if check.detail:
            line += f": {check.detail}"
        print(line)
        if check.status == "FAIL":
            any_fail = True

    if any_fail:
        print("SETUP CHECK FAILED")
        sys.exit(1)
    print("SETUP CHECK PASSED")


if __name__ == "__main__":
    main()
