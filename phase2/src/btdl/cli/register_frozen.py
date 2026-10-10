"""python -m btdl.cli.register_frozen --model NAME

The sanctioned way to record that a model's final (seed-complete) runs are
FROZEN and ready for simultaneous unblinding (D18). Writes TRACKED
phase2/artifacts/frozen/<model>.json with model/run PROVENANCE ONLY --
no test information of any kind, since this file is written entirely
upstream of final_test.

Requires: NAME in study.yaml's models list and the (non-reference_only)
catalog; a selection file (artifacts/selection/<model>.json); and, for
every seed in training.yaml's final_seeds, a run at the selected lr that
is frozen (best.pt sha256 matches FROZEN.json), contract_conformant,
clean, and has a passing eval/rehearsal_val/ (require_contract=True).
Refuses to overwrite an existing registration.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from btdl import config
from btdl.evaluation.results import validate_evaluation_dir
from btdl.models.registry import get_spec
from btdl.runs import run_dir_for

FROZEN_REGISTRATION_DIR_RELPATH = "phase2/artifacts/frozen"
SELECTION_DIR_RELPATH = "phase2/artifacts/selection"


class RegisterFrozenError(ValueError):
    """Raised when a model's final runs cannot be registered as frozen."""


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _current_git_commit(repo_root) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def _current_lock_version(repo_root) -> str:
    lock_path = repo_root / "phase2" / "artifacts" / "contract" / "foundation_lock.json"
    if not lock_path.is_file():
        return None
    with lock_path.open() as handle:
        return json.load(handle).get("lock_version")


def run_register_frozen(model_name: str) -> dict:
    repo_root = config.repo_root()

    study = config.load_study()
    if model_name not in study["models"]:
        raise RegisterFrozenError(
            f"{model_name!r} is not in study.yaml's models list {list(study['models'])}"
        )

    spec = get_spec(model_name)  # raises ValueError (unknown model) if not in the catalog
    if spec.reference_only:
        raise RegisterFrozenError(f"refusing: model {model_name!r} is reference_only")

    out_path = repo_root / FROZEN_REGISTRATION_DIR_RELPATH / f"{model_name}.json"
    if out_path.is_file():
        raise RegisterFrozenError(f"{out_path} already exists -- refusing to overwrite")

    selection_path = repo_root / SELECTION_DIR_RELPATH / f"{model_name}.json"
    if not selection_path.is_file():
        raise RegisterFrozenError(
            f"no selection file at {selection_path} -- run `python -m btdl.cli.select_lr` first"
        )
    with selection_path.open() as handle:
        selection = json.load(handle)
    selected_lr = selection["selected_lr"]

    training_cfg = config.load_contract("training")
    final_seeds = list(training_cfg["final_seeds"])

    seeds_info = []
    for seed in final_seeds:
        run_dir = run_dir_for(model_name, selected_lr, seed, smoke=False)
        frozen_path = run_dir / "FROZEN.json"
        metadata_path = run_dir / "metadata.json"
        best_path = run_dir / "best.pt"
        rehearsal_dir = run_dir / "eval" / "rehearsal_val"

        if not frozen_path.is_file():
            raise RegisterFrozenError(
                f"{run_dir} is not frozen (no FROZEN.json) -- run `python -m btdl.cli.freeze` first"
            )
        with frozen_path.open() as handle:
            frozen = json.load(handle)

        if not best_path.is_file():
            raise RegisterFrozenError(f"best.pt not found in {run_dir}")
        actual_sha256 = _file_sha256(best_path)
        if actual_sha256 != frozen.get("best_pt_sha256"):
            raise RegisterFrozenError(
                f"{run_dir}: best.pt sha256 ({actual_sha256}) does not match "
                f"FROZEN.json ({frozen.get('best_pt_sha256')})"
            )

        if not metadata_path.is_file():
            raise RegisterFrozenError(f"metadata.json not found in {run_dir}")
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        if metadata.get("contract_conformant") is not True:
            raise RegisterFrozenError(f"{run_dir}: metadata.contract_conformant is not True")
        if metadata.get("git_dirty") is not False:
            raise RegisterFrozenError(f"{run_dir}: metadata.git_dirty is not False")

        if not rehearsal_dir.is_dir():
            raise RegisterFrozenError(
                f"{run_dir} has no eval/rehearsal_val/ -- run "
                "`python -m btdl.cli.final_test --rehearsal` first"
            )
        validate_evaluation_dir(rehearsal_dir, require_contract=True)  # raises on any failure

        seeds_info.append(
            {
                "seed": seed,
                "run_dir": str(run_dir.resolve().relative_to(repo_root)),
                "best_pt_sha256": actual_sha256,
                "frozen_at_commit": frozen.get("frozen_at_commit"),
                "best_epoch": metadata.get("best_epoch"),
                "best_val_macro_f1": metadata.get("best_val_macro_f1"),
            }
        )

    registration = {
        "model_name": model_name,
        "model_version": spec.version,
        "weights_id": spec.weights_id,
        "selected_lr": selected_lr,
        "contract_dir_sha256": config.contract_dir_sha256(),
        "foundation_lock_version": _current_lock_version(repo_root),
        "git_commit": _current_git_commit(repo_root),
        "seeds": seeds_info,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(registration, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return registration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    registration = run_register_frozen(args.model)
    print(json.dumps(registration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
