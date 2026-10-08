"""Read-only access to the cached 224x224 ROI tensor (phase2/cache/roi224_v1.npy)."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from btdl import config

META_RELPATH = "phase2/artifacts/contract/roi_cache_meta.json"
INPUT_CONTRACT_RELPATH = "phase2/configs/contract/input.yaml"


class RoiCacheError(ValueError):
    """Raised when the ROI cache is missing, stale, or built under a different input contract."""


@dataclass(frozen=True)
class RoiCache:
    array: np.ndarray  # read-only memmap, shape (N, H, W)
    sample_id_to_row: dict

    def __getitem__(self, sample_id):
        return self.array[self.sample_id_to_row[sample_id]]

    def __len__(self):
        return len(self.sample_id_to_row)


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _sample_order_sha256(sample_ids) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def open_roi_cache(verify: bool = False) -> RoiCache:
    """Open the cache read-only, refusing it if it's stale or was built under a
    different input contract. Pass verify=True for a full (slower) hash check."""

    repo_root = config.repo_root()
    meta_path = repo_root / META_RELPATH
    if not meta_path.is_file():
        raise RoiCacheError(f"cache metadata not found: {meta_path}; run `python -m btdl.cli.build_cache` first")
    with meta_path.open() as handle:
        meta = json.load(handle)

    current_input_hash = _file_sha256(repo_root / INPUT_CONTRACT_RELPATH)
    if current_input_hash != meta["input_yaml_sha256"]:
        raise RoiCacheError(
            "ROI cache was built under a different input contract -- rebuild it: "
            f"expected input.yaml sha256 {meta['input_yaml_sha256']}, got {current_input_hash}"
        )

    manifest_path = repo_root / meta["manifest_relpath"]
    if _file_sha256(manifest_path) != meta["manifest_sha256"]:
        raise RoiCacheError("ROI cache was built from a different manifest.csv -- rebuild it")

    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    sample_ids = list(manifest["sample_id"])
    if _sample_order_sha256(sample_ids) != meta["sample_order_sha256"]:
        raise RoiCacheError("ROI cache sample order does not match the current manifest order")

    cache_path = config.cache_dir() / f"{meta['cache_name']}.npy"
    if not cache_path.is_file():
        raise RoiCacheError(f"cache array not found: {cache_path}; run `python -m btdl.cli.build_cache` first")

    if verify:
        actual_npy_hash = _file_sha256(cache_path)
        if actual_npy_hash != meta["npy_sha256"]:
            raise RoiCacheError(
                f"cache array sha256 mismatch: expected {meta['npy_sha256']}, got {actual_npy_hash}"
            )

    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    array = np.load(cache_path, mmap_mode="r")
    if tuple(array.shape) != shape or array.dtype != dtype:
        raise RoiCacheError(
            f"cache array shape/dtype {array.shape}/{array.dtype} != metadata {shape}/{dtype}"
        )

    sample_id_to_row = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    return RoiCache(array=array, sample_id_to_row=sample_id_to_row)
