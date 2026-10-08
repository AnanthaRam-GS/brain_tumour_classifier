"""python -m btdl.cli.evaluate --run-dir R --split val

Rebuilds the model via the registry from metadata.json's model_name
(pretrained=False -- weights come from best.pt, not from torchvision),
loads best.pt (strict contract), predicts on val, and writes R/eval/val/
via write_evaluation + plots. --split test is refused -- use
cli/final_test.py, which is gated (requires a frozen checkpoint and logs
test access).

On the val split, also checks reload equivalence (C8): the freshly
reloaded best.pt's val probabilities must match the val_predictions.csv
saved at the best epoch during training, within
configs/contract/evaluation.yaml's reload_equivalence_atol. The max abs
diff is recorded in metrics.json as reload_max_abs_diff.
"""

import argparse
import json
from pathlib import Path

from btdl import config
from btdl.data.dataset import RoiDataset
from btdl.data.loader import make_loader
from btdl.evaluation.plots import plot_confusion_matrix, plot_roc_curves, plot_training_curves
from btdl.evaluation.predict import predict
from btdl.evaluation.reload_check import ReloadEquivalenceError, compute_reload_max_abs_diff
from btdl.evaluation.results import write_evaluation
from btdl.models.registry import build_model
from btdl.training.checkpointing import load_checkpoint
from btdl.training.device import select_device

ALLOWED_SPLITS = ("val",)


class EvaluateCliError(ValueError):
    """Raised on a disallowed split, reload-equivalence failure, or other misuse."""


def run_evaluate(run_dir, *, split: str, device, batch_size: int = 64, num_workers: int = 0) -> dict:
    if split == "test":
        raise EvaluateCliError(
            "split='test' is refused here -- use `python -m btdl.cli.final_test`, "
            "which is gated: it requires a frozen checkpoint and logs test access."
        )
    if split not in ALLOWED_SPLITS:
        raise EvaluateCliError(f"split must be one of {ALLOWED_SPLITS} (or 'test', which is refused), got {split!r}")

    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise EvaluateCliError(f"metadata.json not found in {run_dir}")
    with metadata_path.open() as handle:
        metadata = json.load(handle)

    model = build_model(metadata["model_name"], pretrained=False)
    load_checkpoint(run_dir / "best.pt", model, device=device, strict_contract=True)

    dataset = RoiDataset(split, augment=False, seed=metadata.get("seed", 0))
    loader, _ = make_loader(
        dataset, batch_size=batch_size, shuffle=False, seed=0, num_workers=num_workers, device_type=device.type
    )
    predictions = predict(model, loader, device)

    extra = {}
    val_predictions_path = run_dir / "val_predictions.csv"
    if split == "val" and val_predictions_path.is_file():
        reload_max_abs_diff = compute_reload_max_abs_diff(predictions, val_predictions_path)
        atol = config.load_contract("evaluation")["reload_equivalence_atol"]
        if reload_max_abs_diff > atol:
            raise ReloadEquivalenceError(
                f"reload equivalence failed: max abs diff {reload_max_abs_diff} > atol {atol}"
            )
        extra["reload_max_abs_diff"] = reload_max_abs_diff

    out_dir = run_dir / "eval" / split
    metrics = write_evaluation(out_dir, predictions=predictions, run_metadata=metadata, split=split, extra=extra)

    plot_confusion_matrix(metrics["confusion_matrix"], out_dir)
    with (out_dir / "roc_curves.json").open() as handle:
        roc_curves = json.load(handle)
    plot_roc_curves(roc_curves, metrics["roc_auc"], out_dir)
    history_path = run_dir / "history.csv"
    if history_path.is_file():
        plot_training_curves(history_path, out_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    metrics = run_evaluate(
        args.run_dir, split=args.split, device=device, batch_size=args.batch_size, num_workers=args.num_workers
    )
    summary = {k: v for k, v in metrics.items() if k not in ("confusion_matrix", "per_class")}
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
