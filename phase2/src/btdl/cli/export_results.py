"""python -m btdl.cli.export_results --run-dir R

Copies a run's human-review artifacts -- metadata.json, history.csv,
FROZEN.json, efficiency.json, eval/val/*, eval/test/* -- into the TRACKED
phase2/artifacts/results/<model>/seed<seed>/ directory. Never copies a
checkpoint (best.pt/last.pt stay in the gitignored runs/ tree). Only valid
for a run that has gone through a REAL (non-rehearsal) `cli/final_test`
(i.e. R/eval/test/ exists -- a rehearsal-only run is refused). Validates
both copied eval directories with validate_evaluation_dir(
require_contract=True) before returning, and refuses to overwrite an
existing export.
"""

import argparse
import json
import shutil
from pathlib import Path

from btdl import config
from btdl.evaluation.results import validate_evaluation_dir

RESULTS_DIR_RELPATH = "phase2/artifacts/results"
REQUIRED_RUN_FILES = ("metadata.json", "history.csv", "FROZEN.json", "efficiency.json")


class ExportResultsError(ValueError):
    """Raised when a run cannot be exported."""


def run_export_results(run_dir, *, repo_root=None) -> Path:
    run_dir = Path(run_dir)
    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()

    test_eval_dir = run_dir / "eval" / "test"
    if not test_eval_dir.is_dir():
        raise ExportResultsError(
            f"{run_dir} has no eval/test/ -- export_results requires a run that has gone through a "
            "REAL (non-rehearsal) `python -m btdl.cli.final_test` first"
        )
    val_eval_dir = run_dir / "eval" / "val"
    if not val_eval_dir.is_dir():
        raise ExportResultsError(f"{run_dir} has no eval/val/")

    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ExportResultsError(f"metadata.json not found in {run_dir}")
    with metadata_path.open() as handle:
        metadata = json.load(handle)

    model_name = metadata.get("model_name")
    seed = metadata.get("seed")
    if not model_name or seed is None:
        raise ExportResultsError(f"metadata.json in {run_dir} is missing model_name/seed")

    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise ExportResultsError(f"{run_dir} is missing required files: {missing}")

    dest_dir = repo_root / RESULTS_DIR_RELPATH / model_name / f"seed{seed}"
    if dest_dir.exists():
        raise ExportResultsError(f"{dest_dir} already exists -- refusing to overwrite")

    dest_dir.mkdir(parents=True)
    for name in REQUIRED_RUN_FILES:
        shutil.copy2(run_dir / name, dest_dir / name)
    shutil.copytree(val_eval_dir, dest_dir / "eval" / "val")
    shutil.copytree(test_eval_dir, dest_dir / "eval" / "test")

    validate_evaluation_dir(dest_dir / "eval" / "val", require_contract=True)
    validate_evaluation_dir(dest_dir / "eval" / "test", require_contract=True)

    return dest_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    dest_dir = run_export_results(args.run_dir)
    print(json.dumps({"exported_to": str(dest_dir)}, indent=2))


if __name__ == "__main__":
    main()
