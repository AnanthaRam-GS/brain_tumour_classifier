"""Shared Phase 1 access contract for processed MRI samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd

REQUIRED_SPLIT_COLUMNS = ("sample_id", "patient_id", "label", "split")
REQUIRED_NPZ_KEYS = (
    "image_raw",
    "image_normalized",
    "tumor_mask",
    "tumor_border",
    "label",
    "patient_id",
    "sample_id",
)
VALID_LABELS = {1, 2, 3}
VALID_SPLITS = {"train", "val", "test"}
SplitName = Literal["train", "val", "test", "all"]


class FeatureDatasetError(ValueError):
    """Raised when Phase 1 split metadata or NPZ sample data is invalid."""


@dataclass(frozen=True)
class Phase1Sample:
    """Validated processed sample plus frozen split metadata."""

    sample_id: str
    patient_id: str
    label: int
    split: str
    image_normalized: np.ndarray
    tumor_mask: np.ndarray
    image_raw: np.ndarray
    tumor_border: np.ndarray


def _normalize_sample_id(value: object) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise FeatureDatasetError(f"sample_id must be an integer-like value, got {value!r}") from exc
    if number < 1:
        raise FeatureDatasetError(f"sample_id must be positive, got {number}")
    return str(number)


def _scalar_text(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise FeatureDatasetError(f"NPZ {field} must be a scalar")
    return str(array.reshape(-1)[0])


def _scalar_int(value: np.ndarray, field: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise FeatureDatasetError(f"NPZ {field} must be a scalar")
    try:
        number = int(array.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise FeatureDatasetError(f"NPZ {field} must be an integer scalar") from exc
    return number


def _split_frame(path: Path | str) -> pd.DataFrame:
    split_path = Path(path)
    if not split_path.is_file():
        raise FeatureDatasetError(f"split file does not exist: {split_path}")
    frame = pd.read_csv(split_path, dtype={"patient_id": str})
    missing = sorted(set(REQUIRED_SPLIT_COLUMNS).difference(frame.columns))
    if missing:
        raise FeatureDatasetError(f"split file missing required column(s): {', '.join(missing)}")
    return frame.loc[:, REQUIRED_SPLIT_COLUMNS].copy()


def load_split_index(split_csv: Path | str = "data/splits/patient_split.csv") -> pd.DataFrame:
    """Load and validate the shared Phase 1 split index."""

    frame = _split_frame(split_csv)
    if frame.empty:
        raise FeatureDatasetError("split file is empty")

    frame["sample_id"] = frame["sample_id"].map(_normalize_sample_id)
    try:
        frame["label"] = frame["label"].astype(int)
    except (TypeError, ValueError) as exc:
        raise FeatureDatasetError("split labels must be integers") from exc
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    frame["split"] = frame["split"].astype(str).str.strip()

    if frame["sample_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["sample_id"].duplicated(), "sample_id"].unique(), key=int)
        raise FeatureDatasetError(f"duplicate sample IDs in split file: {duplicates[:10]}")
    if frame["patient_id"].eq("").any():
        raise FeatureDatasetError("split file contains blank patient IDs")
    invalid_labels = sorted(set(frame["label"].unique()).difference(VALID_LABELS))
    if invalid_labels:
        raise FeatureDatasetError(f"split labels outside {sorted(VALID_LABELS)}: {invalid_labels}")
    invalid_splits = sorted(set(frame["split"].unique()).difference(VALID_SPLITS))
    if invalid_splits:
        raise FeatureDatasetError(f"invalid split value(s): {invalid_splits}")

    patient_split_counts = frame.groupby("patient_id")["split"].nunique()
    leaking = patient_split_counts[patient_split_counts.ne(1)]
    if not leaking.empty:
        raise FeatureDatasetError(
            f"patients assigned to multiple splits: {sorted(leaking.index.astype(str))[:10]}"
        )
    patient_label_counts = frame.groupby("patient_id")["label"].nunique()
    conflicting = patient_label_counts[patient_label_counts.ne(1)]
    if not conflicting.empty:
        raise FeatureDatasetError(
            f"patients assigned multiple labels: {sorted(conflicting.index.astype(str))[:10]}"
        )

    return frame.sort_values("sample_id", key=lambda column: column.astype(int)).reset_index(drop=True)


def get_split_records(
    split: SplitName = "all",
    *,
    split_csv: Path | str = "data/splits/patient_split.csv",
) -> list[dict[str, object]]:
    """Return metadata records without loading MRI arrays."""

    if split != "all" and split not in VALID_SPLITS:
        raise FeatureDatasetError(f"split must be one of train, val, test, all; got {split!r}")
    frame = load_split_index(split_csv)
    if split != "all":
        frame = frame[frame["split"].eq(split)]
    return frame.to_dict(orient="records")


def _validate_arrays(
    image_raw: np.ndarray,
    image_normalized: np.ndarray,
    tumor_mask: np.ndarray,
) -> None:
    if image_raw.ndim != 2:
        raise FeatureDatasetError(f"image_raw must be 2D, got shape {image_raw.shape}")
    if image_normalized.ndim != 2:
        raise FeatureDatasetError(
            f"image_normalized must be 2D, got shape {image_normalized.shape}"
        )
    if tumor_mask.ndim != 2:
        raise FeatureDatasetError(f"tumor_mask must be 2D, got shape {tumor_mask.shape}")
    if not np.issubdtype(image_raw.dtype, np.number):
        raise FeatureDatasetError("image_raw must be numeric")
    if not np.issubdtype(image_normalized.dtype, np.number):
        raise FeatureDatasetError("image_normalized must be numeric")
    if not np.isfinite(image_raw).all():
        raise FeatureDatasetError("image_raw contains NaN or infinite values")
    if not np.isfinite(image_normalized).all():
        raise FeatureDatasetError("image_normalized contains NaN or infinite values")
    if image_raw.shape != image_normalized.shape:
        raise FeatureDatasetError(
            f"image_raw and image_normalized shapes differ: {image_raw.shape} != {image_normalized.shape}"
        )
    if image_raw.shape != tumor_mask.shape:
        raise FeatureDatasetError(
            f"image and tumor_mask shapes differ: {image_raw.shape} != {tumor_mask.shape}"
        )

    tolerance = float(np.finfo(np.float32).eps * 16)
    normalized_min = float(image_normalized.min())
    normalized_max = float(image_normalized.max())
    if normalized_min < -tolerance or normalized_max > 1.0 + tolerance:
        raise FeatureDatasetError(
            f"image_normalized values outside [0, 1]: [{normalized_min}, {normalized_max}]"
        )

    mask_values = set(np.unique(tumor_mask).tolist())
    if not mask_values.issubset({0, 1, False, True}):
        raise FeatureDatasetError(f"tumor_mask is not binary: {sorted(mask_values)}")
    if int(tumor_mask.astype(np.uint8, copy=False).sum()) == 0:
        raise FeatureDatasetError("tumor_mask is empty")


def _record_for_sample(split_index: pd.DataFrame, sample_id: str) -> pd.Series:
    matches = split_index[split_index["sample_id"].eq(sample_id)]
    if matches.empty:
        raise FeatureDatasetError(f"sample_id {sample_id} is not present in the split file")
    if len(matches) > 1:
        raise FeatureDatasetError(f"sample_id {sample_id} appears more than once in the split file")
    return matches.iloc[0]


def load_phase1_sample(
    sample_id: str | int,
    *,
    split_csv: Path | str = "data/splits/patient_split.csv",
    samples_dir: Path | str = "data/processed/samples",
    split_index: pd.DataFrame | None = None,
) -> Phase1Sample:
    """Load one validated processed sample through the shared split contract."""

    normalized_id = _normalize_sample_id(sample_id)
    index = load_split_index(split_csv) if split_index is None else split_index
    record = _record_for_sample(index, normalized_id)
    npz_path = Path(samples_dir) / f"{normalized_id}.npz"
    if not npz_path.is_file():
        raise FeatureDatasetError(f"NPZ file does not exist for sample {normalized_id}: {npz_path}")

    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = sorted(set(REQUIRED_NPZ_KEYS).difference(archive.files))
            if missing:
                raise FeatureDatasetError(f"NPZ missing required key(s): {', '.join(missing)}")
            image_raw = archive["image_raw"]
            image_normalized = archive["image_normalized"]
            tumor_mask = archive["tumor_mask"]
            tumor_border = archive["tumor_border"]
            npz_label = _scalar_int(archive["label"], "label")
            npz_patient_id = _scalar_text(archive["patient_id"], "patient_id")
            npz_sample_id = _scalar_text(archive["sample_id"], "sample_id")
    except FeatureDatasetError:
        raise
    except Exception as exc:
        raise FeatureDatasetError(f"cannot load NPZ file {npz_path}: {exc}") from exc

    expected_patient_id = str(record["patient_id"])
    expected_label = int(record["label"])
    if npz_sample_id != normalized_id:
        raise FeatureDatasetError(
            f"NPZ sample_id {npz_sample_id!r} does not match split sample_id {normalized_id!r}"
        )
    if npz_patient_id != expected_patient_id:
        raise FeatureDatasetError(
            f"NPZ patient_id {npz_patient_id!r} does not match split patient_id {expected_patient_id!r}"
        )
    if npz_label != expected_label:
        raise FeatureDatasetError(
            f"NPZ label {npz_label} does not match split label {expected_label}"
        )

    _validate_arrays(image_raw, image_normalized, tumor_mask)
    return Phase1Sample(
        sample_id=normalized_id,
        patient_id=expected_patient_id,
        label=expected_label,
        split=str(record["split"]),
        image_normalized=image_normalized,
        tumor_mask=tumor_mask,
        image_raw=image_raw,
        tumor_border=tumor_border,
    )


def iter_phase1_samples(
    split: SplitName = "all",
    *,
    split_csv: Path | str = "data/splits/patient_split.csv",
    samples_dir: Path | str = "data/processed/samples",
) -> Iterator[Phase1Sample]:
    """Iterate validated Phase 1 samples in numeric sample_id order."""

    if split != "all" and split not in VALID_SPLITS:
        raise FeatureDatasetError(f"split must be one of train, val, test, all; got {split!r}")
    index = load_split_index(split_csv)
    if split != "all":
        index = index[index["split"].eq(split)].reset_index(drop=True)
    for sample_id in index["sample_id"]:
        yield load_phase1_sample(
            sample_id,
            split_csv=split_csv,
            samples_dir=samples_dir,
            split_index=index,
        )
