"""Atomic, provenance-rich checkpointing with resume support.

Two files per run directory:
  best.pt  -- model state_dict + metadata, written on every validation
              improvement.
  last.pt  -- model + optimizer + scheduler + early-stopping state + epoch
              + history + metadata, written at the end of every epoch, for
              resume. metadata is mirrored to metadata.json alongside both.
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path

import torch
import torchvision

from btdl import config

logger = logging.getLogger(__name__)

_MISSING = object()


def normalize_cfg(value):
    """Recursively convert FrozenDict/dict -> dict and tuple/list -> list.

    Used both to make an effective cfg JSON-serializable for metadata, and
    to compare two cfg-like structures (contract vs. effective) regardless
    of which concrete container types they use.
    """

    if isinstance(value, Mapping):
        return {key: normalize_cfg(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_cfg(item) for item in value]
    return value


def compute_contract_deviations(contract_cfg, effective_cfg) -> list:
    """Leaf-level differences between a committed contract and an effective cfg.

    Each entry is {"key": dotted path, "contract_value": ..., "effective_value": ...}.
    A key present in one but not the other shows up with the missing side as None.
    """

    deviations = []
    _diff_cfg(contract_cfg, effective_cfg, "", deviations)
    return deviations


def _diff_cfg(contract_value, effective_value, path, out):
    contract_value = normalize_cfg(contract_value)
    effective_value = normalize_cfg(effective_value)

    contract_is_dict = isinstance(contract_value, dict)
    effective_is_dict = isinstance(effective_value, dict)
    if contract_is_dict or effective_is_dict:
        contract_dict = contract_value if contract_is_dict else {}
        effective_dict = effective_value if effective_is_dict else {}
        for key in sorted(set(contract_dict.keys()) | set(effective_dict.keys())):
            child_path = f"{path}.{key}" if path else key
            _diff_cfg(
                contract_dict.get(key, _MISSING), effective_dict.get(key, _MISSING), child_path, out
            )
        return

    if contract_value != effective_value:
        out.append(
            {
                "key": path,
                "contract_value": None if contract_value is _MISSING else contract_value,
                "effective_value": None if effective_value is _MISSING else effective_value,
            }
        )

# torch's own optimizer state_dicts (e.g. AdamW) embed a TorchVersion marker
# for internal state-dict versioning. It's a plain str subclass shipped by
# torch itself, not arbitrary user data, so it's safe to allowlist for
# torch.load(weights_only=True).
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])


class CheckpointError(ValueError):
    """Raised on a contract mismatch or a malformed checkpoint."""


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _git_commit_and_dirty():
    repo_root = config.repo_root()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True)
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return commit, dirty


def _atomic_torch_save(obj, path) -> None:
    """Write to a temp file in the same directory, fsync, then os.replace.

    On any failure before the replace, the temp file is removed -- the
    canonical path is never touched and no partial file is left behind.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    try:
        with open(tmp_path, "wb") as handle:
            torch.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    try:
        with open(tmp_path, "w") as handle:
            json.dump(obj, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def build_run_metadata(
    *,
    model,
    model_name: str,
    model_version: str,
    effective_cfg,
    lr: float,
    seed: int,
    best_epoch,
    best_val_macro_f1,
    best_val_loss,
    device_description: dict,
    determinism_settings: dict,
) -> dict:
    """A JSON-compatible-primitives-only dict describing one run/checkpoint.

    effective_cfg is the cfg ACTUALLY used by fit() for this run -- recorded
    in full, and compared (not silently substituted) against the committed
    training.yaml to produce contract_conformant/contract_deviations.
    """

    repo_root = config.repo_root()
    training_cfg = config.load_contract("training")
    data_cfg = config.load_contract("data")

    effective_cfg_normalized = normalize_cfg(effective_cfg)
    contract_deviations = compute_contract_deviations(training_cfg, effective_cfg)
    contract_conformant = len(contract_deviations) == 0

    manifest_path = repo_root / data_cfg["manifest"]
    manifest_sha256 = _file_sha256(manifest_path) if manifest_path.is_file() else None

    roi_cache_meta_path = repo_root / "phase2" / "artifacts" / "contract" / "roi_cache_meta.json"
    roi_cache_npy_sha256 = None
    if roi_cache_meta_path.is_file():
        with roi_cache_meta_path.open() as handle:
            roi_cache_npy_sha256 = json.load(handle).get("npy_sha256")

    augmentation_yaml_sha256 = _file_sha256(
        repo_root / "phase2" / "configs" / "contract" / "augmentation.yaml"
    )
    training_yaml_sha256 = _file_sha256(repo_root / "phase2" / "configs" / "contract" / "training.yaml")

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    git_commit, git_dirty = _git_commit_and_dirty()

    return {
        "model_name": model_name,
        "model_version": model_version,
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "contract_version": training_cfg["contract_version"],
        "contract_dir_sha256": config.contract_dir_sha256(),
        "effective_cfg": effective_cfg_normalized,
        "contract_conformant": contract_conformant,
        "contract_deviations": contract_deviations,
        "manifest_sha256": manifest_sha256,
        "split_sha256": data_cfg["split_sha256"],
        "roi_cache_npy_sha256": roi_cache_npy_sha256,
        "augmentation_yaml_sha256": augmentation_yaml_sha256,
        "training_yaml_sha256": training_yaml_sha256,
        "lr": lr,
        "weight_decay": effective_cfg_normalized["optimizer"]["weight_decay"],
        "batch_size": effective_cfg_normalized["batch_size"],
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_loss": best_val_loss,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": device_description,
        "determinism": determinism_settings,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def save_best_checkpoint(run_dir, *, model, metadata: dict):
    """Writes best.pt (model only) + metadata.json. Atomic; never a partial file."""

    run_dir = Path(run_dir)
    path = run_dir / "best.pt"
    _atomic_torch_save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)
    _atomic_write_json(metadata, run_dir / "metadata.json")
    return path


def save_last_checkpoint(
    run_dir, *, model, optimizer, scheduler, early_stopping, epoch: int, history: list, metadata: dict
):
    """Writes last.pt (full resume state) + metadata.json. Atomic; never a partial file."""

    run_dir = Path(run_dir)
    path = run_dir / "last.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "early_stopping_state": early_stopping.state_dict(),
        "epoch": epoch,
        "history": history,
        "metadata": metadata,
    }
    _atomic_torch_save(payload, path)
    _atomic_write_json(metadata, run_dir / "metadata.json")
    return path


def load_checkpoint(path, model, *, device, strict_contract: bool = True) -> dict:
    """Loads a checkpoint (weights_only=True), applying its model_state_dict to `model`.

    Refuses a checkpoint whose contract_dir_sha256 differs from the current
    contract unless strict_contract=False, which is logged explicitly.
    """

    payload = torch.load(path, map_location=device, weights_only=True)
    metadata = payload.get("metadata", {})

    checkpoint_hash = metadata.get("contract_dir_sha256")
    current_hash = config.contract_dir_sha256()
    if checkpoint_hash != current_hash:
        if strict_contract:
            raise CheckpointError(
                f"checkpoint contract_dir_sha256 ({checkpoint_hash}) differs from the "
                f"current contract ({current_hash}); pass strict_contract=False to override"
            )
        logger.warning(
            "loading checkpoint %s with strict_contract=False: contract hash mismatch "
            "not enforced (checkpoint=%s, current=%s)",
            path,
            checkpoint_hash,
            current_hash,
        )

    model.load_state_dict(payload["model_state_dict"])
    return payload
