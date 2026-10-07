"""Loader-agnostic training engine.

fit() never constructs a Dataset or DataLoader itself -- callers decide
what loaders to pass, so the engine has no way to reach the test split
(docs/DECISIONS.md D13).
"""

import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from btdl.contracts import CLASS_NAMES_BY_INDEX, index_to_label
from btdl.evaluation.metrics import epoch_metrics
from btdl.training.checkpointing import (
    build_run_metadata,
    load_checkpoint,
    save_best_checkpoint,
    save_last_checkpoint,
)
from btdl.training.device import describe_device
from btdl.training.early_stopping import EarlyStopping
from btdl.training.seed import seed_everything

HISTORY_COLUMNS = (
    "epoch",
    "lr",
    "train_aug_loss",
    "train_aug_macro_f1",
    "val_loss",
    "val_log_loss",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_macro_f1",
    "improved",
    "epoch_seconds",
)

PREDICTION_COLUMNS = ("sample_id", "patient_id", "true_label", "predicted_label") + tuple(
    f"prob_{name}" for name in CLASS_NAMES_BY_INDEX
)


class TrainerError(ValueError):
    """Raised on a malformed model output, a non-finite loss, or a run_dir misuse."""


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_val_macro_f1: float
    best_val_loss: float
    epochs_run: int
    stop_reason: str
    run_dir: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    history_path: Path


def _check_output_shape(output, batch_size: int):
    if not torch.is_tensor(output):
        raise TrainerError(
            "model output must be a torch.Tensor of shape [B, 3], got "
            f"{type(output).__name__} (e.g. a model returning auxiliary outputs as a tuple)"
        )
    if tuple(output.shape) != (batch_size, 3):
        raise TrainerError(f"model output must have shape [{batch_size}, 3], got {tuple(output.shape)}")


def _resolve_t_max(scheduler_cfg, max_epochs: int) -> int:
    t_max = scheduler_cfg["t_max"]
    if t_max == "max_epochs":
        return max_epochs
    return int(t_max)


def _write_history_csv(path, history):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HISTORY_COLUMNS))
        writer.writeheader()
        for row in history:
            writer.writerow({key: row[key] for key in HISTORY_COLUMNS})


def _write_val_predictions_csv(path, *, sample_ids, patient_ids, true_labels, probs):
    predicted_indices = probs.argmax(axis=1)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_COLUMNS))
        writer.writeheader()
        for i in range(len(sample_ids)):
            row = {
                "sample_id": sample_ids[i],
                "patient_id": patient_ids[i],
                "true_label": int(true_labels[i]),
                "predicted_label": index_to_label(int(predicted_indices[i])),
            }
            for class_idx, name in enumerate(CLASS_NAMES_BY_INDEX):
                row[f"prob_{name}"] = float(probs[i, class_idx])
            writer.writerow(row)


class _Log:
    def __init__(self, path):
        self._path = Path(path)

    def write(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(self._path, "a") as handle:
            handle.write(f"{timestamp} {message}\n")


def fit(
    model,
    *,
    train_loader,
    train_sampler,
    val_loader,
    class_weights,
    cfg,
    lr: float,
    seed: int,
    run_dir,
    device,
    resume: bool = False,
    stop_after_epoch=None,
) -> FitResult:
    """stop_after_epoch (optional): halt after completing that epoch, as if
    preempted -- distinct from cfg["max_epochs"], which stays fixed across a
    paused/resumed pair of calls so the cosine schedule's T_max (the true
    total training horizon) does not change between them. For external
    callers this is normally left at its default (None)."""
    run_dir = Path(run_dir)

    if run_dir.exists() and not resume:
        raise TrainerError(f"run_dir already exists and resume=False: {run_dir}")
    if resume and not (run_dir / "last.pt").is_file():
        raise TrainerError(f"resume=True requires an existing last.pt in {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(run_dir / "train.log")
    log.write(f"fit() starting: resume={resume}, lr={lr}, seed={seed}, device={device}")

    # Reset global RNG state deterministically so this fit() call is
    # reproducible by construction, independent of whatever random draws
    # happened before it was called (data order/augmentation are already
    # independently sample/epoch-keyed -- see D16's implementation note --
    # but a model's own internal randomness, e.g. dropout, is not).
    determinism_settings = seed_everything(seed)

    max_epochs = cfg["max_epochs"]
    optimizer_cfg = cfg["optimizer"]
    scheduler_cfg = cfg["scheduler"]
    early_stopping_cfg = cfg["early_stopping"]

    model = model.to(device)
    class_weights_t = class_weights.to(device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=optimizer_cfg["weight_decay"],
        betas=tuple(optimizer_cfg["betas"]),
        eps=optimizer_cfg["eps"],
    )
    t_max = _resolve_t_max(scheduler_cfg, max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=scheduler_cfg["eta_min"]
    )
    early_stopping = EarlyStopping(
        patience=early_stopping_cfg["patience"],
        min_delta=early_stopping_cfg["min_delta"],
        mode=early_stopping_cfg["mode"],
    )

    history = []
    start_epoch = 0

    if resume:
        payload = load_checkpoint(run_dir / "last.pt", model, device=device)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        early_stopping.load_state_dict(payload["early_stopping_state"])
        history = list(payload["history"])
        start_epoch = payload["epoch"] + 1
        log.write(f"resumed from epoch {payload['epoch']}; continuing at epoch {start_epoch}")

    device_description = describe_device(device)
    model_name = type(model).__name__
    model_version = getattr(model, "version", "unknown")

    stop_reason = "max_epochs"
    epoch = start_epoch - 1

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.perf_counter()
        train_sampler.set_epoch(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_targets_parts = []
        train_probs_parts = []

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            batch_size = images.shape[0]

            optimizer.zero_grad()
            output = model(images)
            _check_output_shape(output, batch_size)

            loss = criterion(output, targets)
            if not torch.isfinite(loss):
                raise TrainerError(
                    f"non-finite training loss at epoch {epoch}, batch {batch_idx}: {loss.item()}"
                )
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_size
            train_n += batch_size
            with torch.no_grad():
                train_probs_parts.append(torch.softmax(output, dim=1).detach().cpu())
            train_targets_parts.append(targets.detach().cpu())

        scheduler.step()

        train_aug_loss = train_loss_sum / train_n
        train_metrics = epoch_metrics(
            torch.cat(train_targets_parts).numpy(), torch.cat(train_probs_parts).numpy()
        )

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        val_targets_parts = []
        val_probs_parts = []
        val_sample_ids = []
        val_patient_ids = []
        val_true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["target"].to(device)
                batch_size = images.shape[0]

                output = model(images)
                _check_output_shape(output, batch_size)
                loss = criterion(output, targets)

                val_loss_sum += loss.item() * batch_size
                val_n += batch_size
                val_probs_parts.append(torch.softmax(output, dim=1).detach().cpu())
                val_targets_parts.append(targets.detach().cpu())
                val_sample_ids.extend(batch["sample_id"])
                val_patient_ids.extend(batch["patient_id"])
                labels_batch = batch["label"]
                val_true_labels.extend(labels_batch.tolist() if torch.is_tensor(labels_batch) else list(labels_batch))

        val_loss = val_loss_sum / val_n
        val_probs_cat = torch.cat(val_probs_parts).numpy()
        val_metrics = epoch_metrics(torch.cat(val_targets_parts).numpy(), val_probs_cat)

        step_result = early_stopping.step(val_metrics["macro_f1"], val_loss, epoch)
        epoch_seconds = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_aug_loss": train_aug_loss,
            "train_aug_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_log_loss": val_metrics["log_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "improved": step_result.improved,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        _write_history_csv(run_dir / "history.csv", history)

        metadata = build_run_metadata(
            model=model,
            model_name=model_name,
            model_version=model_version,
            lr=lr,
            seed=seed,
            best_epoch=early_stopping.best_epoch,
            best_val_macro_f1=early_stopping.best_f1,
            best_val_loss=early_stopping.best_loss,
            device_description=device_description,
            determinism_settings=determinism_settings,
        )

        if step_result.improved:
            save_best_checkpoint(run_dir, model=model, metadata=metadata)
            _write_val_predictions_csv(
                run_dir / "val_predictions.csv",
                sample_ids=val_sample_ids,
                patient_ids=val_patient_ids,
                true_labels=val_true_labels,
                probs=val_probs_cat,
            )

        save_last_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            epoch=epoch,
            history=history,
            metadata=metadata,
        )

        log.write(
            f"epoch {epoch}: val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_loss={val_loss:.4f} improved={step_result.improved}"
        )

        if step_result.should_stop:
            stop_reason = "early_stopping"
            log.write(f"early stopping at epoch {epoch} (patience={early_stopping.patience})")
            break

        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            stop_reason = "stop_after_epoch"
            log.write(f"stopping after epoch {epoch} (stop_after_epoch={stop_after_epoch})")
            break

    epochs_run = max(0, epoch - start_epoch + 1)
    log.write(f"fit() finished: stop_reason={stop_reason}, epochs_run={epochs_run}")

    return FitResult(
        best_epoch=early_stopping.best_epoch,
        best_val_macro_f1=early_stopping.best_f1,
        best_val_loss=early_stopping.best_loss,
        epochs_run=epochs_run,
        stop_reason=stop_reason,
        run_dir=run_dir,
        best_checkpoint_path=run_dir / "best.pt",
        last_checkpoint_path=run_dir / "last.pt",
        history_path=run_dir / "history.csv",
    )
