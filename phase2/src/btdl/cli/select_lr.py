"""python -m btdl.cli.select_lr --model NAME

Requires all lr_grid runs (each lr in training.yaml's lr_grid, at
grid_seed) to exist, be complete, contract-conformant, and clean. Selects
by best val macro-F1, then lower best val loss, then lower lr. Writes the
TRACKED phase2/artifacts/selection/<model>.json. Refuses to overwrite an
existing selection file.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from btdl import config
from btdl.runs import run_dir_for

SELECTION_DIR_RELPATH = "phase2/artifacts/selection"


class SelectLrError(ValueError):
    """Raised when a grid run is missing, incomplete, non-conformant, dirty, or already selected."""


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _current_git_commit(repo_root) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def _sort_key(candidate: dict):
    # Best val macro-F1 (descending), then lower best val loss, then lower lr.
    return (-candidate["best_val_macro_f1"], candidate["best_val_loss"], candidate["lr"])


def run_select_lr(model_name: str) -> dict:
    training_cfg = config.load_contract("training")
    lr_grid = list(training_cfg["lr_grid"])
    grid_seed = training_cfg["grid_seed"]

    repo_root = config.repo_root()
    selection_path = repo_root / SELECTION_DIR_RELPATH / f"{model_name}.json"
    if selection_path.is_file():
        raise SelectLrError(f"selection file already exists: {selection_path}; refusing to overwrite")

    candidates = []
    for lr in lr_grid:
        run_dir = run_dir_for(model_name, lr, grid_seed, smoke=False)
        metadata_path = run_dir / "metadata.json"
        history_path = run_dir / "history.csv"
        best_path = run_dir / "best.pt"

        if not metadata_path.is_file() or not history_path.is_file() or not best_path.is_file():
            raise SelectLrError(f"grid run for lr={lr} is missing or incomplete at {run_dir}")

        with metadata_path.open() as handle:
            metadata = json.load(handle)

        if metadata.get("contract_conformant") is not True:
            raise SelectLrError(f"grid run for lr={lr} at {run_dir} is not contract_conformant")
        if metadata.get("git_dirty") is not False:
            raise SelectLrError(f"grid run for lr={lr} at {run_dir} was recorded as git_dirty")

        candidates.append(
            {
                "lr": lr,
                "best_epoch": metadata.get("best_epoch"),
                "best_val_macro_f1": metadata.get("best_val_macro_f1"),
                "best_val_loss": metadata.get("best_val_loss"),
                "run_dir": str(run_dir.resolve().relative_to(repo_root)),
                "best_pt_sha256": _file_sha256(best_path),
            }
        )

    ranked = sorted(candidates, key=_sort_key)
    selected = ranked[0]

    selection = {
        "model_name": model_name,
        "candidates": candidates,
        "selected_lr": selected["lr"],
        "selected_run_dir": selected["run_dir"],
        "contract_dir_sha256": config.contract_dir_sha256(),
        "git_commit": _current_git_commit(repo_root),
    }

    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("w") as handle:
        json.dump(selection, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    selection = run_select_lr(args.model)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
