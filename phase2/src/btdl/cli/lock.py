"""python -m btdl.cli.lock --write | --check

The foundation lock (phase2/artifacts/contract/foundation_lock.json,
tracked) records a SHA-256 per file in the locked foundation: every
configs/contract/*.yaml file, src/btdl/{config.py, contracts.py, runs.py},
every .py file under src/btdl/{data, preprocessing, training, evaluation,
cli}/, and src/btdl/models/{registry.py, model_spec.py,
torchvision_common.py} -- plus the overall contract_dir_sha256 and a lock
version.

Deliberately NOT locked: src/btdl/models/catalog.py and any
src/btdl/models/<model>.py file, so a teammate can register a new
architecture (one catalog.py entry + one new model module) without
touching anything the lock covers. tests/test_foundation_lock.py fails
with a readable diff if any locked file changes without the lock being
regenerated -- a legitimate foundation change regenerates it
(`--write`) as part of a reviewed PR.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from btdl import config

LOCK_VERSION = "1.0.0"
LOCK_RELPATH = "phase2/artifacts/contract/foundation_lock.json"

_BTDL_SRC_RELPATH = "phase2/src/btdl"
_CONTRACT_RELPATH = "phase2/configs/contract"
_TOP_LEVEL_LOCKED_FILES = ("config.py", "contracts.py", "runs.py")
_LOCKED_SUBDIRS = ("data", "preprocessing", "training", "evaluation", "cli")
_MODELS_LOCKED_FILES = ("registry.py", "model_spec.py", "torchvision_common.py")


class FoundationLockError(ValueError):
    """Raised when the lock file is missing or malformed."""


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def locked_relpaths(repo_root=None) -> list:
    """Sorted, repo-root-relative (posix) paths covered by the foundation lock."""

    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    btdl_src = repo_root / _BTDL_SRC_RELPATH
    contract_dir = repo_root / _CONTRACT_RELPATH

    rels = set()
    for path in contract_dir.rglob("*"):
        if path.is_file():
            rels.add(path.relative_to(repo_root).as_posix())

    for name in _TOP_LEVEL_LOCKED_FILES:
        if (btdl_src / name).is_file():
            rels.add(f"{_BTDL_SRC_RELPATH}/{name}")

    for subdir in _LOCKED_SUBDIRS:
        for path in (btdl_src / subdir).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rels.add(path.relative_to(repo_root).as_posix())

    for name in _MODELS_LOCKED_FILES:
        if (btdl_src / "models" / name).is_file():
            rels.add(f"{_BTDL_SRC_RELPATH}/models/{name}")

    return sorted(rels)


def compute_lock(repo_root=None) -> dict:
    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    rels = locked_relpaths(repo_root)
    files = {rel: _file_sha256(repo_root / rel) for rel in rels}
    return {
        "lock_version": LOCK_VERSION,
        "contract_dir_sha256": config.contract_dir_sha256(),
        "files": files,
    }


def write_lock(repo_root=None) -> dict:
    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    lock = compute_lock(repo_root)
    lock_path = repo_root / LOCK_RELPATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return lock


def check_lock(repo_root=None) -> tuple:
    """Returns (ok, diff). diff has keys added/removed/changed (each a list
    of relpaths) whenever ok is False."""

    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    lock_path = repo_root / LOCK_RELPATH
    if not lock_path.is_file():
        raise FoundationLockError(
            f"no foundation lock at {lock_path} -- run `python -m btdl.cli.lock --write` first"
        )
    with lock_path.open() as handle:
        recorded = json.load(handle)

    current = compute_lock(repo_root)

    recorded_files = recorded.get("files", {})
    current_files = current["files"]

    added = sorted(set(current_files) - set(recorded_files))
    removed = sorted(set(recorded_files) - set(current_files))
    changed = sorted(
        rel for rel in (set(current_files) & set(recorded_files)) if current_files[rel] != recorded_files[rel]
    )

    diff = {"added": added, "removed": removed, "changed": changed}
    ok = not (added or removed or changed) and recorded.get("contract_dir_sha256") == current["contract_dir_sha256"]
    if recorded.get("contract_dir_sha256") != current["contract_dir_sha256"] and not (added or removed or changed):
        # Shouldn't happen (contract files are themselves locked), but keep
        # the diff informative if it ever does.
        diff["contract_dir_sha256"] = {
            "recorded": recorded.get("contract_dir_sha256"),
            "current": current["contract_dir_sha256"],
        }

    return ok, diff


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.write:
        lock = write_lock()
        print(json.dumps(lock, indent=2, sort_keys=True))
        return

    ok, diff = check_lock()
    if not ok:
        print(json.dumps(diff, indent=2, sort_keys=True), file=sys.stderr)
        print("FOUNDATION LOCK CHECK FAILED", file=sys.stderr)
        sys.exit(1)
    print("FOUNDATION LOCK OK")


if __name__ == "__main__":
    main()
