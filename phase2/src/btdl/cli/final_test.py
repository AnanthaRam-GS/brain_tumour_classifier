"""python -m btdl.cli.final_test --run-dir R --confirm-final-test --model-class pkg.mod:ClassName

The ONLY sanctioned way Phase 2 touches the real test split. Every
requirement below is checked BEFORE any test data is loaded:
  - FROZEN.json exists and best.pt's sha256 matches it
  - the checkpoint's contract hash equals the CURRENT contract
  - metadata.contract_conformant is true
  - the current working tree is clean
  - this run has no prior entry in the test-access log
Then predicts on RoiDataset("test", ..., allow_test=True) and writes
R/eval/test/ (same artifacts and plots as cli/evaluate.py). Appends one row
to the tracked phase2/artifacts/test_access_log.csv.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from btdl import config
from btdl.data.dataset import RoiDataset
from btdl.data.loader import make_loader
from btdl.evaluation.plots import plot_confusion_matrix, plot_roc_curves, plot_training_curves
from btdl.evaluation.predict import predict
from btdl.evaluation.results import write_evaluation
from btdl.training.checkpointing import load_checkpoint
from btdl.training.device import select_device

TEST_ACCESS_LOG_RELPATH = "phase2/artifacts/test_access_log.csv"
TEST_ACCESS_LOG_COLUMNS = (
    "run_dir",
    "model_name",
    "lr",
    "seed",
    "best_pt_sha256",
    "git_commit",
    "utc_timestamp",
    "git_user_name",
)


class FinalTestError(ValueError):
    """Raised when any final-test gate fails."""


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _git_is_dirty(repo_root) -> bool:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True)
    return bool(status.strip())


def _git_user_name(repo_root) -> str:
    try:
        return subprocess.check_output(["git", "config", "user.name"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def _relative_run_dir(run_dir, repo_root) -> str:
    try:
        return str(Path(run_dir).resolve().relative_to(repo_root))
    except ValueError:
        return str(run_dir)


def _already_logged(log_path, run_dir_rel: str) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return any(row.get("run_dir") == run_dir_rel for row in reader)


def _append_log_row(log_path, row: dict) -> None:
    file_exists = log_path.is_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TEST_ACCESS_LOG_COLUMNS))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_final_test(
    run_dir,
    model,
    *,
    confirm_final_test: bool,
    device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict:
    if not confirm_final_test:
        raise FinalTestError("final_test requires --confirm-final-test")

    run_dir = Path(run_dir)
    repo_root = config.repo_root()

    frozen_path = run_dir / "FROZEN.json"
    best_path = run_dir / "best.pt"
    metadata_path = run_dir / "metadata.json"

    if not frozen_path.is_file():
        raise FinalTestError(f"{run_dir} is not frozen (no FROZEN.json) -- run cli/freeze.py first")
    with frozen_path.open() as handle:
        frozen = json.load(handle)

    if not best_path.is_file():
        raise FinalTestError(f"best.pt not found in {run_dir}")
    actual_best_sha256 = _file_sha256(best_path)
    if actual_best_sha256 != frozen.get("best_pt_sha256"):
        raise FinalTestError(
            "best.pt sha256 does not match FROZEN.json: "
            f"expected {frozen.get('best_pt_sha256')}, got {actual_best_sha256}"
        )

    current_contract_hash = config.contract_dir_sha256()
    if frozen.get("contract_dir_sha256") != current_contract_hash:
        raise FinalTestError(
            f"FROZEN.json contract_dir_sha256 ({frozen.get('contract_dir_sha256')}) does not match "
            f"the current contract ({current_contract_hash})"
        )

    if not metadata_path.is_file():
        raise FinalTestError(f"metadata.json not found in {run_dir}")
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    if metadata.get("contract_conformant") is not True:
        raise FinalTestError("refusing final_test: metadata.contract_conformant is not True")

    if _git_is_dirty(repo_root):
        raise FinalTestError("refusing final_test: the current working tree is not clean")

    run_dir_rel = _relative_run_dir(run_dir, repo_root)
    log_path = repo_root / TEST_ACCESS_LOG_RELPATH
    if _already_logged(log_path, run_dir_rel):
        raise FinalTestError(
            f"{run_dir_rel} already has a test-access log entry -- final_test may run only once per run"
        )

    # --- Every gate above passed. Only now does any test data get loaded. ---
    load_checkpoint(best_path, model, device=device, strict_contract=True)

    dataset = RoiDataset("test", augment=False, seed=metadata.get("seed", 0), allow_test=True)
    loader, _ = make_loader(
        dataset, batch_size=batch_size, shuffle=False, seed=0, num_workers=num_workers, device_type=device.type
    )
    predictions = predict(model, loader, device)

    out_dir = run_dir / "eval" / "test"
    metrics = write_evaluation(out_dir, predictions=predictions, run_metadata=metadata, split="test")

    plot_confusion_matrix(metrics["confusion_matrix"], out_dir)
    with (out_dir / "roc_curves.json").open() as handle:
        roc_curves = json.load(handle)
    plot_roc_curves(roc_curves, metrics["roc_auc"], out_dir)
    history_path = run_dir / "history.csv"
    if history_path.is_file():
        plot_training_curves(history_path, out_dir)

    _append_log_row(
        log_path,
        {
            "run_dir": run_dir_rel,
            "model_name": metadata.get("model_name"),
            "lr": metadata.get("lr"),
            "seed": metadata.get("seed"),
            "best_pt_sha256": actual_best_sha256,
            "git_commit": metadata.get("git_commit"),
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "git_user_name": _git_user_name(repo_root),
        },
    )

    return metrics


def _instantiate_model(model_class_spec: str):
    import importlib

    module_name, class_name = model_class_spec.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--model-class", required=True, help="module:ClassName, no-arg constructor")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model = _instantiate_model(args.model_class)
    device = select_device(args.device)
    metrics = run_final_test(
        args.run_dir,
        model,
        confirm_final_test=args.confirm_final_test,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    summary = {k: v for k, v in metrics.items() if k not in ("confusion_matrix", "per_class")}
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
