"""python -m btdl.cli.train --model NAME --lr LR --seed SEED [--smoke] [--resume] [--allow-dirty]

Assembles everything from the contracts: RoiDataset train(augment)/val,
class weights, loaders, select_device, build_model(pretrained=True), then
fit(). After fit() completes, automatically runs the val evaluation
(eval/val/).

Protocol rules (refused with a clear message):
  - lr must be in training.yaml's lr_grid
  - seed == grid_seed: allowed for any grid lr
  - seed in final_seeds (and != grid_seed): requires
    phase2/artifacts/selection/<model>.json to exist, with lr == its
    selected lr
  - any other seed: refused
  - a dirty working tree is refused unless --allow-dirty (the run is then
    recorded as git_dirty, via the existing metadata mechanism, and cannot
    later be frozen)
--smoke overrides max_epochs to 2 (a recorded contract deviation, so the
run is non-conformant) and places the run under phase2/runs/_smoke/.
"""

import argparse
import json
import subprocess
from pathlib import Path

from btdl import config
from btdl.cli.evaluate import run_evaluate
from btdl.data.class_weights import compute_class_weights
from btdl.data.dataset import RoiDataset
from btdl.data.loader import make_loader
from btdl.models.registry import build_model
from btdl.runs import run_dir_for
from btdl.training.checkpointing import normalize_cfg
from btdl.training.device import select_device
from btdl.training.trainer import fit

SMOKE_MAX_EPOCHS = 2


class TrainCliError(ValueError):
    """Raised when the train CLI's lr/seed protocol is violated, or on a dirty tree."""


def _git_is_dirty(repo_root) -> bool:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True)
    return bool(status.strip())


def _validate_lr_seed(model_name: str, lr: float, seed: int, training_cfg) -> None:
    lr_grid = list(training_cfg["lr_grid"])
    grid_seed = training_cfg["grid_seed"]
    final_seeds = list(training_cfg["final_seeds"])

    if lr not in lr_grid:
        raise TrainCliError(f"lr={lr} is not in training.yaml's lr_grid {lr_grid}")

    if seed == grid_seed:
        return

    if seed in final_seeds:
        selection_path = config.repo_root() / "phase2" / "artifacts" / "selection" / f"{model_name}.json"
        if not selection_path.is_file():
            raise TrainCliError(
                f"seed={seed} is a final seed but no selection file exists at {selection_path}; "
                "run `python -m btdl.cli.select_lr --model ...` first"
            )
        with selection_path.open() as handle:
            selection = json.load(handle)
        selected_lr = selection["selected_lr"]
        if lr != selected_lr:
            raise TrainCliError(
                f"lr={lr} does not match the selected lr {selected_lr} in {selection_path}"
            )
        return

    raise TrainCliError(
        f"seed={seed} is neither grid_seed ({grid_seed}) nor one of final_seeds {final_seeds}"
    )


def run_train(
    model_name: str,
    lr: float,
    seed: int,
    *,
    smoke: bool = False,
    resume: bool = False,
    allow_dirty: bool = False,
    device=None,
    num_workers: int = None,
):
    training_cfg = config.load_contract("training")
    _validate_lr_seed(model_name, lr, seed, training_cfg)

    repo_root = config.repo_root()
    if _git_is_dirty(repo_root) and not allow_dirty:
        raise TrainCliError(
            "working tree is dirty; pass --allow-dirty to proceed "
            "(the run will be recorded as git_dirty and cannot later be frozen)"
        )

    effective_cfg = normalize_cfg(training_cfg)
    if smoke:
        effective_cfg["max_epochs"] = SMOKE_MAX_EPOCHS  # deviation -> non-conformant

    run_dir = run_dir_for(model_name, lr, seed, smoke=smoke)
    device = device or select_device("auto")
    num_workers = num_workers if num_workers is not None else training_cfg["num_workers"]

    train_ds = RoiDataset("train", augment=True, seed=seed)
    val_ds = RoiDataset("val", augment=False, seed=seed)
    train_loader, train_sampler = make_loader(
        train_ds, batch_size=effective_cfg["batch_size"], shuffle=True, seed=seed,
        num_workers=num_workers, device_type=device.type,
    )
    val_loader, _ = make_loader(
        val_ds, batch_size=training_cfg["eval_batch_size"], shuffle=False, seed=seed,
        num_workers=num_workers, device_type=device.type,
    )

    class_weights = compute_class_weights(train_ds)
    model = build_model(model_name, pretrained=True)

    result = fit(
        model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        class_weights=class_weights,
        cfg=effective_cfg,
        lr=lr,
        seed=seed,
        run_dir=run_dir,
        device=device,
        resume=resume,
        model_name=model_name,
    )

    run_evaluate(
        run_dir, split="val", device=device,
        batch_size=training_cfg["eval_batch_size"], num_workers=num_workers,
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    result = run_train(
        args.model,
        args.lr,
        args.seed,
        smoke=args.smoke,
        resume=args.resume,
        allow_dirty=args.allow_dirty,
        device=device,
        num_workers=args.num_workers,
    )
    print(json.dumps({k: v for k, v in result.__dict__.items() if not isinstance(v, Path)}, indent=2, default=str))
    print(f"run_dir: {result.run_dir}")


if __name__ == "__main__":
    main()
