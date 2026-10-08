"""python -m btdl.cli.final_test --run-dir R --confirm-final-test [--rehearsal]

The ONLY sanctioned way Phase 2 touches the real test split. Every
requirement below is checked BEFORE any test data is loaded:
  - FROZEN.json exists and best.pt's sha256 matches it
  - the checkpoint's contract hash equals the CURRENT contract
  - metadata.contract_conformant is true
  - the current working tree is clean
Real mode (default) additionally requires:
  - this run has no prior entry in the test-access log
  - the model is NOT reference_only (tiny_cnn, for example, can never pass
    this -- only a real architecture can)
  - the run's lr equals the selected lr in
    phase2/artifacts/selection/<model>.json, and its seed is in
    training.yaml's final_seeds
  Then predicts on RoiDataset("test", ..., allow_test=True) and writes
  R/eval/test/, and appends one row to the tracked
  phase2/artifacts/test_access_log.csv.

--rehearsal (D17) runs the IDENTICAL pipeline on the VALIDATION split
instead, writing R/eval/rehearsal_val/. It still requires FROZEN and
enforces every provenance gate above, but the selection and
reference_only gates are waived (so tiny_cnn, or an unselected lr/seed,
can rehearse). It NEVER writes the access log and NEVER constructs the
test dataset -- allow_test=True appears nowhere outside the real path
(see test_no_allow_test_outside_final_test.py). Also checks reload
equivalence (C8), same as cli/evaluate.py.
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
from btdl.evaluation.reload_check import ReloadEquivalenceError, compute_reload_max_abs_diff
from btdl.evaluation.results import write_evaluation
from btdl.models.registry import build_model, get_spec
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


def _check_base_provenance(run_dir, repo_root):
    """Gates required in BOTH real and rehearsal mode. Returns (frozen, metadata, best_pt_sha256)."""

    frozen_path = run_dir / "FROZEN.json"
    best_path = run_dir / "best.pt"
    metadata_path = run_dir / "metadata.json"

    if not frozen_path.is_file():
        raise FinalTestError(f"{run_dir} is not frozen (no FROZEN.json) -- run cli/freeze.py first")
    with frozen_path.open() as handle:
        frozen = json.load(handle)

    if not best_path.is_file():
        raise FinalTestError(f"best.pt not found in {run_dir}")
    best_pt_sha256 = _file_sha256(best_path)
    if best_pt_sha256 != frozen.get("best_pt_sha256"):
        raise FinalTestError(
            f"best.pt sha256 does not match FROZEN.json: expected {frozen.get('best_pt_sha256')}, got {best_pt_sha256}"
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
        raise FinalTestError("refusing: metadata.contract_conformant is not True")

    dirty_paths = config.git_dirty_paths(repo_root)
    if dirty_paths:
        raise FinalTestError(f"refusing: the current working tree is not clean. Offending paths: {dirty_paths}")

    return frozen, metadata, best_pt_sha256


def _check_real_mode_gates(run_dir, metadata, repo_root):
    """Gates required ONLY in real mode (waived for --rehearsal)."""

    spec = get_spec(metadata["model_name"])
    if spec.reference_only:
        raise FinalTestError(
            f"refusing: model {metadata['model_name']!r} is reference_only -- only a real "
            "architecture may run a real final_test (use --rehearsal instead)"
        )

    training_cfg = config.load_contract("training")
    final_seeds = list(training_cfg["final_seeds"])
    if metadata.get("seed") not in final_seeds:
        raise FinalTestError(f"refusing: run seed {metadata.get('seed')} is not in final_seeds {final_seeds}")

    selection_path = repo_root / "phase2" / "artifacts" / "selection" / f"{metadata['model_name']}.json"
    if not selection_path.is_file():
        raise FinalTestError(f"refusing: no selection file at {selection_path} -- run cli/select_lr.py first")
    with selection_path.open() as handle:
        selection = json.load(handle)
    if metadata.get("lr") != selection["selected_lr"]:
        raise FinalTestError(
            f"refusing: run lr {metadata.get('lr')} does not match the selected lr "
            f"{selection['selected_lr']} in {selection_path}"
        )

    run_dir_rel = _relative_run_dir(run_dir, repo_root)
    log_path = repo_root / TEST_ACCESS_LOG_RELPATH
    if _already_logged(log_path, run_dir_rel):
        raise FinalTestError(
            f"{run_dir_rel} already has a test-access log entry -- final_test may run only once per run"
        )


def _write_split_evaluation(run_dir, model, device, *, split, out_dirname, metadata, batch_size, num_workers):
    seed = metadata.get("seed", 0)
    if split == "test":
        # The only allow_test=True usage outside this literal call is refused
        # by test_no_allow_test_outside_final_test.py's guard test.
        dataset = RoiDataset(split, augment=False, seed=seed, allow_test=True)
    else:
        dataset = RoiDataset(split, augment=False, seed=seed)
    loader, _ = make_loader(
        dataset, batch_size=batch_size, shuffle=False, seed=0, num_workers=num_workers, device_type=device.type
    )
    predictions = predict(model, loader, device)

    extra = {}
    if split == "val":
        val_predictions_path = run_dir / "val_predictions.csv"
        if val_predictions_path.is_file():
            reload_max_abs_diff = compute_reload_max_abs_diff(predictions, val_predictions_path)
            atol = config.load_contract("evaluation")["reload_equivalence_atol"]
            if reload_max_abs_diff > atol:
                raise ReloadEquivalenceError(
                    f"reload equivalence failed: max abs diff {reload_max_abs_diff} > atol {atol}"
                )
            extra["reload_max_abs_diff"] = reload_max_abs_diff

    out_dir = run_dir / "eval" / out_dirname
    metrics = write_evaluation(
        out_dir, predictions=predictions, run_metadata=metadata, split=split, extra=extra
    )

    plot_confusion_matrix(metrics["confusion_matrix"], out_dir)
    with (out_dir / "roc_curves.json").open() as handle:
        roc_curves = json.load(handle)
    plot_roc_curves(roc_curves, metrics["roc_auc"], out_dir)
    history_path = run_dir / "history.csv"
    if history_path.is_file():
        plot_training_curves(history_path, out_dir)

    return metrics, predictions, out_dir


def run_final_test(
    run_dir,
    *,
    confirm_final_test: bool,
    device,
    rehearsal: bool = False,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict:
    if not confirm_final_test:
        raise FinalTestError("final_test requires --confirm-final-test")

    run_dir = Path(run_dir)
    repo_root = config.repo_root()

    frozen, metadata, best_pt_sha256 = _check_base_provenance(run_dir, repo_root)

    if not rehearsal:
        _check_real_mode_gates(run_dir, metadata, repo_root)

    # --- Every required gate passed. Only now does the model get loaded
    # and (in real mode only) the test dataset get constructed. ---
    model = build_model(metadata["model_name"], pretrained=False)
    load_checkpoint(run_dir / "best.pt", model, device=device, strict_contract=True)

    if rehearsal:
        metrics, _, _ = _write_split_evaluation(
            run_dir, model, device, split="val", out_dirname="rehearsal_val",
            metadata=metadata, batch_size=batch_size, num_workers=num_workers,
        )
        return metrics

    metrics, _, _ = _write_split_evaluation(
        run_dir, model, device, split="test", out_dirname="test",
        metadata=metadata, batch_size=batch_size, num_workers=num_workers,
    )

    run_dir_rel = _relative_run_dir(run_dir, repo_root)
    log_path = repo_root / TEST_ACCESS_LOG_RELPATH
    _append_log_row(
        log_path,
        {
            "run_dir": run_dir_rel,
            "model_name": metadata.get("model_name"),
            "lr": metadata.get("lr"),
            "seed": metadata.get("seed"),
            "best_pt_sha256": best_pt_sha256,
            "git_commit": metadata.get("git_commit"),
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "git_user_name": _git_user_name(repo_root),
        },
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--rehearsal", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    metrics = run_final_test(
        args.run_dir,
        confirm_final_test=args.confirm_final_test,
        device=device,
        rehearsal=args.rehearsal,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    summary = {k: v for k, v in metrics.items() if k not in ("confusion_matrix", "per_class")}
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
