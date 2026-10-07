"""Frozen patient split contract (docs/DECISIONS.md D3).

No function in this module creates or reshuffles a split -- it only loads
and validates the single train/val/test split Phase 1 already produced and
froze in data/splits/patient_split.csv.
"""

import hashlib
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from btdl import config

VALID_SPLITS = ("train", "val", "test")


class SplitIntegrityError(ValueError):
    """Raised when the split file's hash, or its content, does not match the contract."""


class SplitIndex:
    """Immutable sample_id -> {split, patient_id, label} index."""

    def __init__(self, rows):
        self._rows = MappingProxyType(dict(rows))

    def __getitem__(self, sample_id):
        return self._rows[sample_id]

    def __contains__(self, sample_id):
        return sample_id in self._rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def items(self):
        return self._rows.items()

    def sample_ids(self, split_name):
        if split_name not in VALID_SPLITS:
            raise ValueError(f"split_name must be one of {VALID_SPLITS}, got {split_name!r}")
        matching = (
            sample_id for sample_id, row in self._rows.items() if row["split"] == split_name
        )
        return tuple(sorted(matching, key=int))


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def load_split() -> SplitIndex:
    """Load data/splits/patient_split.csv, verifying its SHA-256 against the contract first."""

    contract = config.load_contract("data")
    split_path = config.resolve_data_path(contract["split_csv"])
    if not split_path.is_file():
        raise SplitIntegrityError(f"split file not found: {split_path}")

    actual_hash = _file_sha256(split_path)
    expected_hash = contract["split_sha256"]
    if actual_hash != expected_hash:
        raise SplitIntegrityError(
            f"{split_path} sha256 mismatch: expected {expected_hash}, got {actual_hash}"
        )

    frame = pd.read_csv(split_path, dtype={"sample_id": str, "patient_id": str})
    rows = {}
    for row in frame.itertuples(index=False):
        rows[row.sample_id] = MappingProxyType(
            {"split": row.split, "patient_id": row.patient_id, "label": int(row.label)}
        )
    return SplitIndex(rows)


def validate_split(split_index: SplitIndex, manifest: pd.DataFrame) -> bool:
    """Cross-check split_index against a manifest DataFrame. Raises SplitIntegrityError on any disagreement."""

    manifest_ids = set(manifest["sample_id"])
    split_ids = set(iter(split_index))

    missing_in_manifest = sorted(split_ids - manifest_ids, key=int)
    missing_in_split = sorted(manifest_ids - split_ids, key=int)
    if missing_in_manifest or missing_in_split:
        raise SplitIntegrityError(
            "split/manifest sample_id mismatch: "
            f"{len(missing_in_manifest)} in split but not manifest "
            f"(e.g. {missing_in_manifest[:5]}), "
            f"{len(missing_in_split)} in manifest but not split "
            f"(e.g. {missing_in_split[:5]})"
        )

    manifest_by_id = {row.sample_id: row for row in manifest.itertuples(index=False)}
    for sample_id, split_row in split_index.items():
        manifest_row = manifest_by_id[sample_id]
        if str(manifest_row.patient_id) != split_row["patient_id"]:
            raise SplitIntegrityError(f"patient_id mismatch for sample {sample_id}")
        if int(manifest_row.label) != split_row["label"]:
            raise SplitIntegrityError(f"label mismatch for sample {sample_id}")

    observed_splits = {row["split"] for _, row in split_index.items()}
    if observed_splits != set(VALID_SPLITS):
        raise SplitIntegrityError(
            f"splits must be exactly {set(VALID_SPLITS)}, got {observed_splits}"
        )

    patients_by_split = {
        name: {row["patient_id"] for _, row in split_index.items() if row["split"] == name}
        for name in VALID_SPLITS
    }
    for i in range(len(VALID_SPLITS)):
        for j in range(i + 1, len(VALID_SPLITS)):
            a, b = VALID_SPLITS[i], VALID_SPLITS[j]
            overlap = patients_by_split[a] & patients_by_split[b]
            if overlap:
                raise SplitIntegrityError(f"patient overlap between {a} and {b}: {sorted(overlap)}")

    classes_by_split = {
        name: {row["label"] for _, row in split_index.items() if row["split"] == name}
        for name in VALID_SPLITS
    }
    for name, classes in classes_by_split.items():
        if classes != {1, 2, 3}:
            missing_classes = {1, 2, 3} - classes
            raise SplitIntegrityError(f"split {name!r} is missing classes: {missing_classes}")

    contract = config.load_contract("data")
    expected_samples = contract["expected"]["split_samples"]
    expected_patients = contract["expected"]["split_patients"]
    for name in VALID_SPLITS:
        observed_sample_count = sum(1 for _, row in split_index.items() if row["split"] == name)
        if observed_sample_count != expected_samples[name]:
            raise SplitIntegrityError(
                f"split {name!r} sample count {observed_sample_count} "
                f"!= expected {expected_samples[name]}"
            )
        observed_patient_count = len(patients_by_split[name])
        if observed_patient_count != expected_patients[name]:
            raise SplitIntegrityError(
                f"split {name!r} patient count {observed_patient_count} "
                f"!= expected {expected_patients[name]}"
            )

    return True
