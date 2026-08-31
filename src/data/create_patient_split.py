"""Create the Phase 1 leakage-safe patient-level train/val/test split."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("sample_id", "patient_id", "label")
VALID_LABELS = {1, 2, 3}
VALID_SPLITS = ("train", "val", "test")
CLASS_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}
CVIND_POLICY = (
    "Historical/reference only; not used to generate the Phase 1 "
    "train/validation/test split."
)


class PatientSplitError(ValueError):
    """Raised when a patient-level split cannot be created or validated."""


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".csv", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_manifest(path: Path | str) -> pd.DataFrame:
    """Load and validate the sample manifest columns required for splitting."""

    manifest_path = Path(path)
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str})
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise PatientSplitError(f"manifest missing required column(s): {', '.join(missing)}")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    if frame.empty:
        raise PatientSplitError("manifest is empty")

    try:
        frame["sample_id"] = frame["sample_id"].astype(int)
    except (TypeError, ValueError) as exc:
        raise PatientSplitError("sample_id values must be integers") from exc
    try:
        frame["label"] = frame["label"].astype(int)
    except (TypeError, ValueError) as exc:
        raise PatientSplitError("label values must be integers") from exc

    if frame["sample_id"].duplicated().any():
        duplicated = sorted(frame.loc[frame["sample_id"].duplicated(), "sample_id"].unique())
        raise PatientSplitError(f"duplicate sample_id values: {duplicated[:10]}")
    if frame["sample_id"].isna().any() or (frame["sample_id"] < 1).any():
        raise PatientSplitError("sample_id values must be positive integers")
    if frame["patient_id"].isna().any() or frame["patient_id"].astype(str).str.strip().eq("").any():
        raise PatientSplitError("patient_id values must be populated")
    if not set(frame["label"].unique()).issubset(VALID_LABELS):
        raise PatientSplitError(f"labels must be in {sorted(VALID_LABELS)}")

    label_counts = frame.groupby("patient_id")["label"].nunique()
    conflicting = label_counts[label_counts.ne(1)]
    if not conflicting.empty:
        examples = sorted(conflicting.index.astype(str).tolist())[:10]
        raise PatientSplitError(f"patients with multiple labels: {examples}")

    return frame.sort_values("sample_id").reset_index(drop=True)


def make_patient_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Collapse sample metadata to one row per patient."""

    patient_table = (
        manifest.groupby("patient_id", as_index=False)
        .agg(label=("label", "first"), sample_count=("sample_id", "count"))
        .sort_values(["label", "patient_id"])
        .reset_index(drop=True)
    )
    return patient_table


def _class_split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(round(total * train_ratio))
    remaining = total - train_count
    val_count = remaining // 2
    test_count = remaining - val_count
    return train_count, val_count, test_count


def assign_patient_splits(
    patient_table: pd.DataFrame,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """Assign each unique patient to exactly one split using per-class shuffling."""

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise PatientSplitError("split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    assignments: list[pd.DataFrame] = []
    for label in sorted(VALID_LABELS):
        class_patients = patient_table[patient_table["label"].eq(label)].copy()
        if class_patients.empty:
            raise PatientSplitError(f"class {label} has no patients")
        order = rng.permutation(len(class_patients))
        shuffled = class_patients.iloc[order].reset_index(drop=True)
        train_count, val_count, _ = _class_split_counts(
            len(shuffled), train_ratio, val_ratio
        )
        shuffled.loc[: train_count - 1, "split"] = "train"
        shuffled.loc[train_count : train_count + val_count - 1, "split"] = "val"
        shuffled.loc[train_count + val_count :, "split"] = "test"
        assignments.append(shuffled)

    assigned = pd.concat(assignments, ignore_index=True)
    return assigned[["patient_id", "label", "sample_count", "split"]]


def build_sample_split(manifest: pd.DataFrame, patient_split: pd.DataFrame) -> pd.DataFrame:
    """Map patient-level split assignments back to every sample."""

    sample_split = manifest.merge(
        patient_split[["patient_id", "split"]], on="patient_id", how="left", validate="many_to_one"
    )
    if sample_split["split"].isna().any():
        raise PatientSplitError("some samples did not receive a split assignment")
    return sample_split[["sample_id", "patient_id", "label", "split"]].sort_values(
        "sample_id"
    ).reset_index(drop=True)


def _nested_counts(frame: pd.DataFrame, group_column: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in VALID_SPLITS:
        split_frame = frame[frame["split"].eq(split)]
        counts[split] = {
            CLASS_NAMES[label]: int(split_frame[split_frame["label"].eq(label)][group_column].nunique())
            if group_column == "patient_id"
            else int(split_frame["label"].eq(label).sum())
            for label in sorted(VALID_LABELS)
        }
    return counts


def validate_split(
    manifest: pd.DataFrame,
    sample_split: pd.DataFrame,
    *,
    expected_samples: int | None = None,
    expected_patients: int | None = None,
) -> dict[str, Any]:
    """Validate coverage, leakage, labels, schema, and deterministic ordering."""

    expected_samples = len(manifest) if expected_samples is None else expected_samples
    expected_patients = manifest["patient_id"].nunique() if expected_patients is None else expected_patients
    expected_ids = set(manifest["sample_id"].astype(int))
    actual_ids = set(sample_split["sample_id"].astype(int))

    if list(sample_split.columns) != ["sample_id", "patient_id", "label", "split"]:
        raise PatientSplitError("sample split schema must be sample_id, patient_id, label, split")
    if len(sample_split) != expected_samples:
        raise PatientSplitError(f"expected {expected_samples} rows, found {len(sample_split)}")
    if sample_split["patient_id"].nunique() != expected_patients:
        raise PatientSplitError(
            f"expected {expected_patients} patients, found {sample_split['patient_id'].nunique()}"
        )
    if sample_split["sample_id"].duplicated().any():
        raise PatientSplitError("sample split contains duplicate sample IDs")
    if expected_ids != actual_ids:
        raise PatientSplitError("sample split IDs do not match manifest IDs")
    if not set(sample_split["split"].unique()).issubset(VALID_SPLITS):
        raise PatientSplitError("split values must be only train, val, or test")
    if sample_split["sample_id"].tolist() != sorted(sample_split["sample_id"].tolist()):
        raise PatientSplitError("sample split must be sorted by numeric sample_id")

    original_labels = manifest.set_index("sample_id")["label"].sort_index()
    split_labels = sample_split.set_index("sample_id")["label"].sort_index()
    if not original_labels.equals(split_labels):
        raise PatientSplitError("sample labels changed during split creation")

    patient_split_counts = sample_split.groupby("patient_id")["split"].nunique()
    if patient_split_counts.ne(1).any():
        raise PatientSplitError("at least one patient appears in multiple splits")
    patient_label_counts = sample_split.groupby("patient_id")["label"].nunique()
    if patient_label_counts.ne(1).any():
        raise PatientSplitError("at least one patient has multiple labels")

    patients_by_split = {
        split: set(sample_split.loc[sample_split["split"].eq(split), "patient_id"])
        for split in VALID_SPLITS
    }
    overlaps = {
        "train_val": sorted(patients_by_split["train"] & patients_by_split["val"]),
        "train_test": sorted(patients_by_split["train"] & patients_by_split["test"]),
        "val_test": sorted(patients_by_split["val"] & patients_by_split["test"]),
    }
    if any(overlaps.values()):
        raise PatientSplitError("patient leakage detected between splits")

    classes_by_split = {
        split: sorted(sample_split.loc[sample_split["split"].eq(split), "label"].unique().tolist())
        for split in VALID_SPLITS
    }
    missing_classes = {
        split: sorted(VALID_LABELS.difference(labels))
        for split, labels in classes_by_split.items()
    }
    if any(missing_classes.values()):
        raise PatientSplitError(f"missing classes in split(s): {missing_classes}")

    return {
        "valid": True,
        "row_count": int(len(sample_split)),
        "unique_patients": int(sample_split["patient_id"].nunique()),
        "duplicate_sample_ids": 0,
        "missing_sample_ids": [],
        "extra_sample_ids": [],
        "patient_overlap": {key: len(value) for key, value in overlaps.items()},
        "classes_by_split": classes_by_split,
        "all_labels_unchanged": True,
        "one_split_per_patient": True,
        "sorted_by_sample_id": True,
    }


def build_metadata(
    manifest: pd.DataFrame,
    sample_split: pd.DataFrame,
    validation: dict[str, Any],
    *,
    source_manifest: Path,
    split_version: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, Any]:
    """Build JSON metadata for the generated split."""

    patient_rows = sample_split.drop_duplicates("patient_id")
    return {
        "split_version": split_version,
        "strategy": "Deterministic class-aware split over unique patient_id values.",
        "random_seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_samples": int(len(sample_split)),
        "total_patients": int(sample_split["patient_id"].nunique()),
        "total_classes": int(sample_split["label"].nunique()),
        "sample_counts_by_split": {
            split: int(sample_split["split"].eq(split).sum()) for split in VALID_SPLITS
        },
        "patient_counts_by_split": {
            split: int(patient_rows["split"].eq(split).sum()) for split in VALID_SPLITS
        },
        "sample_class_counts_by_split": _nested_counts(sample_split, "sample_id"),
        "patient_class_counts_by_split": _nested_counts(sample_split, "patient_id"),
        "source_manifest": str(source_manifest),
        "cvind_policy": CVIND_POLICY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_summary": validation,
    }


def create_patient_split(
    manifest_path: Path | str = "data/processed/manifest.csv",
    output_csv: Path | str = "data/splits/patient_split.csv",
    metadata_json: Path | str = "data/splits/split_metadata.json",
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_version: str = "phase1_patient_split_v1",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create, validate, and write the definitive Phase 1 patient split."""

    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    patient_table = make_patient_table(manifest)
    patient_split = assign_patient_splits(
        patient_table,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    sample_split = build_sample_split(manifest, patient_split)
    validation = validate_split(manifest, sample_split)
    metadata = build_metadata(
        manifest,
        sample_split,
        validation,
        source_manifest=manifest_path,
        split_version=split_version,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    _atomic_csv(sample_split, Path(output_csv))
    _atomic_json(metadata, Path(metadata_json))
    return sample_split, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/splits/patient_split.csv"))
    parser.add_argument(
        "--metadata-json", type=Path, default=Path("data/splits/split_metadata.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, metadata = create_patient_split(
        manifest_path=args.manifest,
        output_csv=args.output_csv,
        metadata_json=args.metadata_json,
        seed=args.seed,
    )
    print(json.dumps(metadata["patient_counts_by_split"], indent=2))
    print(json.dumps(metadata["sample_counts_by_split"], indent=2))


if __name__ == "__main__":
    main()
