"""python -m btdl.cli.check_setup

Prints a PASS/FAIL/WARN checklist verifying a fresh clone is ready to
train, and exits non-zero if ANY check FAILs (a WARN does not fail the
exit code). Checks: versions (CORE packages -- torch, torchvision, numpy,
scikit-learn, scipy, h5py, pyyaml, pandas, matplotlib -- must exactly
match pyproject.toml's pins, FAIL otherwise; every other package in
requirements.lock gets a WARN on mismatch, since a different OS/Python
(e.g. Colab) may legitimately need different transitive versions; the
lock's recorded Python minor version differing from the running
interpreter is also a WARN, not a FAIL); btdl's import location; the
selected device; that every contract file loads; the foundation lock
(cli/lock.py --check); the manifest and split, hash-verified; the ROI
cache at its resolved location (honouring BTDL_CACHE_DIR), with a FULL
sha256 verify; working-tree cleanliness (a WARN, not a FAIL -- see A1's
scoped dirty-tree definition); that btdl never imports Phase 1's `src`
package; and the model registry's contents.
"""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from btdl import config
from btdl.cli.lock import FoundationLockError, check_lock
from btdl.data.roi_cache import RoiCacheError, open_roi_cache
from btdl.data.split import SplitIntegrityError, load_split
from btdl.models.registry import get_spec, list_models
from btdl.training.device import describe_device, select_device

REQUIREMENTS_LOCK_RELPATH = "phase2/requirements.lock"

# Packages whose version is load-bearing for training/evaluation
# correctness and reproducibility -- a mismatch against pyproject.toml's
# pin is a FAIL, not a WARN.
CORE_PACKAGES = frozenset(
    {"torch", "torchvision", "numpy", "scikit-learn", "scipy", "h5py", "pyyaml", "pandas", "matplotlib"}
)


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "FAIL" | "WARN"
    detail: str = ""


def _parse_requirements_lock(path: Path):
    """Returns (header: dict, packages: dict[lowercased name, version]).

    Header lines are `# Key: value` comments at the top of the file (see
    cli's own generation: `pip freeze --exclude-editable` output with a
    Python/Platform/generation-command header prepended).
    """

    header = {}
    packages = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, sep, value = line.lstrip("#").partition(":")
            if sep:
                header[key.strip().lower()] = value.strip()
            continue
        if "==" in line:
            name, version = line.split("==", 1)
            packages[name.strip().lower()] = version.strip()
    return header, packages


def _installed_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_versions() -> CheckResult:
    pyproject_path = config.repo_root() / "phase2" / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)

    pinned = {}
    for dep in pyproject["project"]["dependencies"]:
        if "==" in dep:
            name, version = dep.split("==", 1)
            pinned[name.strip().lower()] = version.strip()

    lock_path = config.repo_root() / REQUIREMENTS_LOCK_RELPATH
    if not lock_path.is_file():
        return CheckResult("versions", "FAIL", f"requirements.lock not found at {lock_path}")
    header, locked = _parse_requirements_lock(lock_path)

    failures = []
    warnings = []

    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    lock_python = header.get("python", "")
    lock_python_minor = ".".join(lock_python.split(".")[:2]) if lock_python else None
    if lock_python_minor and actual_python != lock_python_minor:
        warnings.append(f"python {actual_python} != requirements.lock header {lock_python_minor}")

    core_detail_bits = [f"python={actual_python}"]
    for name in sorted(CORE_PACKAGES):
        expected = pinned.get(name)
        if expected is None:
            continue
        installed = _installed_version(name)
        core_detail_bits.append(f"{name}=={installed}")
        if installed is None:
            failures.append(f"{name} not installed (pyproject pins {expected})")
        elif installed != expected:
            failures.append(f"{name} {installed} != pyproject pin {expected}")

    for name, locked_version in sorted(locked.items()):
        if name in CORE_PACKAGES:
            continue  # CORE packages are checked strictly (FAIL) above
        installed = _installed_version(name)
        if installed is None:
            warnings.append(f"{name} not installed (requirements.lock has {locked_version})")
        elif installed != locked_version:
            warnings.append(f"{name} {installed} != requirements.lock {locked_version}")

    detail = " ".join(core_detail_bits)
    if warnings:
        detail += "; WARN: " + "; ".join(warnings)
    if failures:
        detail += "; FAIL: " + "; ".join(failures)

    name = "versions (CORE packages vs pyproject.toml pins; others vs requirements.lock)"
    if failures:
        return CheckResult(name, "FAIL", detail)
    if warnings:
        return CheckResult(name, "WARN", detail)
    return CheckResult(name, "PASS", detail)


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
