"""Deterministic, content-hashed manifest over the Phase 1 NPZ sample cache.

Hashing is over array/scalar CONTENT (not file bytes), so NPZs regenerated
elsewhere with identical arrays still verify against a committed manifest.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

CONTENT_HASH_FIELD_ORDER = ("image_normalized", "tumor_mask", "label", "patient_id", "sample_id")

MANIFEST_COLUMNS = (
    "sample_id",
    "patient_id",
    "label",
    "split",
    "raw_relpath",
    "npz_relpath",
    "height",
    "width",
    "mask_area",
    "bbox_r0",
    "bbox_r1",
    "bbox_c0",
    "bbox_c1",
    "bbox_h",
    "bbox_w",
    "bbox_longest",
    "content_sha256",
)


def content_sha256(npz_path) -> str:
    """SHA-256 over the canonical bytes of the five content fields, in order.

    Each field contributes: its name, its dtype string, its shape (arrays
    only), then its raw bytes/string encoding -- so changing a field's
    content, dtype, or shape changes the hash, but re-saving the identical
    arrays to a new .npz file does not.
    """

    import hashlib

    digest = hashlib.sha256()
    with np.load(npz_path) as data:
        for field in CONTENT_HASH_FIELD_ORDER:
            if field not in data:
                raise ValueError(f"{npz_path} is missing required field {field!r}")
            value = data[field]
            digest.update(field.encode("utf-8"))
            if field == "image_normalized":
                array = np.ascontiguousarray(value, dtype=np.float32)
                digest.update(str(array.dtype).encode("utf-8"))
                digest.update(str(array.shape).encode("utf-8"))
                digest.update(array.tobytes())
            elif field == "tumor_mask":
                array = np.ascontiguousarray(value, dtype=np.uint8)
                digest.update(str(array.dtype).encode("utf-8"))
                digest.update(str(array.shape).encode("utf-8"))
                digest.update(array.tobytes())
            elif field == "label":
                digest.update(str(int(value)).encode("utf-8"))
            else:  # patient_id, sample_id -- 0-d string arrays
                digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def compute_mask_bbox(mask: np.ndarray):
    """Tight half-open bounding box (r0, r1, c0, c1) of a boolean mask, or None if empty."""

    mask = np.asarray(mask).astype(bool)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0 = int(np.argmax(rows))
    r1 = int(len(rows) - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(len(cols) - np.argmax(cols[::-1]))
    return r0, r1, c0, c1


_DIGIT_STEM = re.compile(r"^\d+$")


def discover_raw_files(raw_dir) -> dict:
    """Map sample_id -> raw .mat path relative to raw_dir's parent, for every digit-named .mat file."""

    raw_dir = Path(raw_dir)
    mapping = {}
    for path in sorted(raw_dir.rglob("*.mat")):
        if not _DIGIT_STEM.match(path.stem):
            continue
        mapping[path.stem] = path
    return mapping


def load_split_map(split_csv) -> dict:
    """sample_id (str) -> split name, read directly from the split CSV."""

    frame = pd.read_csv(split_csv, dtype={"sample_id": str})
    return dict(zip(frame["sample_id"], frame["split"]))


def build_manifest(*, samples_dir, raw_dir, split_csv, data_root) -> pd.DataFrame:
    """One row per NPZ sample under samples_dir, sorted by numeric sample_id."""

    data_root = Path(data_root)
    samples_dir = Path(samples_dir)
    raw_files = discover_raw_files(raw_dir)
    split_map = load_split_map(split_csv)

    npz_paths = sorted(samples_dir.glob("*.npz"), key=lambda p: int(p.stem))

    rows = []
    for npz_path in npz_paths:
        sample_id = npz_path.stem
        with np.load(npz_path) as data:
            label = int(data["label"])
            patient_id = str(data["patient_id"])
            image = data["image_normalized"]
            mask = data["tumor_mask"]

        height, width = image.shape
        mask_area = int(np.asarray(mask).astype(bool).sum())
        bbox = compute_mask_bbox(mask)
        if bbox is None:
            bbox_r0 = bbox_r1 = bbox_c0 = bbox_c1 = bbox_h = bbox_w = bbox_longest = 0
        else:
            bbox_r0, bbox_r1, bbox_c0, bbox_c1 = bbox
            bbox_h = bbox_r1 - bbox_r0
            bbox_w = bbox_c1 - bbox_c0
            bbox_longest = max(bbox_h, bbox_w)

        raw_path = raw_files.get(sample_id)
        raw_relpath = str(raw_path.relative_to(data_root)) if raw_path is not None else ""

        rows.append(
            {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "label": label,
                "split": split_map.get(sample_id, ""),
                "raw_relpath": raw_relpath,
                "npz_relpath": str(npz_path.relative_to(data_root)),
                "height": int(height),
                "width": int(width),
                "mask_area": mask_area,
                "bbox_r0": bbox_r0,
                "bbox_r1": bbox_r1,
                "bbox_c0": bbox_c0,
                "bbox_c1": bbox_c1,
                "bbox_h": bbox_h,
                "bbox_w": bbox_w,
                "bbox_longest": bbox_longest,
                "content_sha256": content_sha256(npz_path),
            }
        )

    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    manifest = manifest.sort_values(by="sample_id", key=lambda col: col.astype(int)).reset_index(
        drop=True
    )
    return manifest


def verify_manifest(manifest: pd.DataFrame, data_root) -> pd.DataFrame:
    """Recompute content_sha256 for every row against local data; return mismatching rows."""

    data_root = Path(data_root)
    mismatches = []
    for _, row in manifest.iterrows():
        npz_path = data_root / row["npz_relpath"]
        if not npz_path.is_file():
            mismatches.append({"sample_id": row["sample_id"], "reason": "npz_missing"})
            continue
        actual_hash = content_sha256(npz_path)
        if actual_hash != row["content_sha256"]:
            mismatches.append(
                {
                    "sample_id": row["sample_id"],
                    "reason": "content_sha256_mismatch",
                    "expected": row["content_sha256"],
                    "actual": actual_hash,
                }
            )
    return pd.DataFrame(mismatches, columns=["sample_id", "reason", "expected", "actual"])
