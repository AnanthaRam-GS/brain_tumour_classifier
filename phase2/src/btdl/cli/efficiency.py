"""python -m btdl.cli.efficiency --run-dir R

Writes R/efficiency.json: parameter counts plus inference timing per the
evaluation contract (batch_sizes/warmup/iters), with the device
description. Rebuilds the model via the registry from metadata.json's
model_name and loads best.pt.
"""

import argparse
import json
from pathlib import Path

from btdl.evaluation.efficiency import count_parameters, measure_inference
from btdl.models.registry import build_model
from btdl.training.checkpointing import load_checkpoint
from btdl.training.device import select_device


class EfficiencyCliError(ValueError):
    """Raised when a run_dir is missing required files."""


def run_efficiency(run_dir, *, device=None) -> dict:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    best_path = run_dir / "best.pt"
    if not metadata_path.is_file():
        raise EfficiencyCliError(f"metadata.json not found in {run_dir}")
    if not best_path.is_file():
        raise EfficiencyCliError(f"best.pt not found in {run_dir}")

    with metadata_path.open() as handle:
        metadata = json.load(handle)

    device = device or select_device("auto")
    model = build_model(metadata["model_name"], pretrained=False)
    load_checkpoint(best_path, model, device=device, strict_contract=False)

    payload = {
        "parameters": count_parameters(model),
        "inference": {str(batch_size): stats for batch_size, stats in measure_inference(model, device).items()},
    }

    with (run_dir / "efficiency.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    payload = run_efficiency(args.run_dir, device=device)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
