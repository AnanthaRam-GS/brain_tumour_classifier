"""Shared CSV contract for Phase 1 handcrafted feature tables."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

METADATA_COLUMNS = ["sample_id", "patient_id", "label", "split"]
VALID_LABELS = {1, 2, 3}
VALID_SPLITS = {"train", "val", "test"}
FEATURE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
REQUIRED_FEATURE_METADATA_FIELDS = {
    "feature_set_name",
    "feature_version",
    "algorithm",
    "parameters",
    "input_representation",
    "roi_policy",
    "source_split_version",
    "sample_count",
    "feature_count",
    "feature_columns",
    "generated_at",
    "code_commit",
    "validation_summary",
}


class FeatureTableError(ValueError):
    """Raised when a feature table or feature metadata JSON is invalid."""


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


def _normalize_sample_id(value: object) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise FeatureTableError(f"sample_id must be integer-like, got {value!r}") from exc
    if number < 1:
        raise FeatureTableError(f"sample_id must be positive, got {number}")
    return str(number)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise FeatureTableError(f"duplicate column name(s): {duplicated}")
    missing = [column for column in METADATA_COLUMNS if column not in frame.columns]
    if missing:
        raise FeatureTableError(f"missing metadata column(s): {', '.join(missing)}")
    feature_columns = [column for column in frame.columns if column not in METADATA_COLUMNS]
    ordered = frame.loc[:, METADATA_COLUMNS + feature_columns].copy()
    if ordered.empty:
        raise FeatureTableError("feature table is empty")
    ordered["sample_id"] = ordered["sample_id"].map(_normalize_sample_id)
    ordered["patient_id"] = ordered["patient_id"].astype(str).str.strip()
    ordered["split"] = ordered["split"].astype(str).str.strip()
    try:
        ordered["label"] = ordered["label"].astype(int)
    except (TypeError, ValueError) as exc:
        raise FeatureTableError("label values must be integers") from exc
    return ordered


def load_canonical_split(split_csv: Path | str = "data/splits/patient_split.csv") -> pd.DataFrame:
    """Load canonical split metadata for feature-table validation."""

    path = Path(split_csv)
    if not path.is_file():
        raise FeatureTableError(f"canonical split file does not exist: {path}")
    split = pd.read_csv(path, dtype={"patient_id": str})
    return validate_feature_table(split, canonical_split=None, require_feature_columns=False)


def validate_feature_table(
    frame: pd.DataFrame,
    *,
    canonical_split: pd.DataFrame | Path | str | None = "data/splits/patient_split.csv",
    require_feature_columns: bool = True,
) -> pd.DataFrame:
    """Validate and return a deterministic copy of a Phase 1 feature table."""

    table = _prepare_frame(frame)
    feature_columns = [column for column in table.columns if column not in METADATA_COLUMNS]
    if require_feature_columns and not feature_columns:
        raise FeatureTableError("feature table must contain at least one feature column")

    if table["sample_id"].duplicated().any():
        duplicates = sorted(table.loc[table["sample_id"].duplicated(), "sample_id"].unique(), key=int)
        raise FeatureTableError(f"duplicate sample IDs: {duplicates[:10]}")
    if table["patient_id"].eq("").any():
        raise FeatureTableError("blank patient IDs are not allowed")
    invalid_labels = sorted(set(table["label"].unique()).difference(VALID_LABELS))
    if invalid_labels:
        raise FeatureTableError(f"labels outside {sorted(VALID_LABELS)}: {invalid_labels}")
    invalid_splits = sorted(set(table["split"].unique()).difference(VALID_SPLITS))
    if invalid_splits:
        raise FeatureTableError(f"invalid split value(s): {invalid_splits}")

    patient_splits = table.groupby("patient_id")["split"].nunique()
    leaking = patient_splits[patient_splits.ne(1)]
    if not leaking.empty:
        raise FeatureTableError(
            f"patients assigned to multiple splits: {sorted(leaking.index.astype(str))[:10]}"
        )
    patient_labels = table.groupby("patient_id")["label"].nunique()
    conflicting = patient_labels[patient_labels.ne(1)]
    if not conflicting.empty:
        raise FeatureTableError(
            f"patients assigned multiple labels: {sorted(conflicting.index.astype(str))[:10]}"
        )

    for column in feature_columns:
        if column in METADATA_COLUMNS:
            raise FeatureTableError(f"feature column reuses metadata name: {column}")
        if not FEATURE_NAME_PATTERN.fullmatch(str(column)):
            raise FeatureTableError(f"invalid feature column name: {column}")
        if not pd.api.types.is_numeric_dtype(table[column]):
            raise FeatureTableError(f"feature column is not numeric: {column}")
        values = table[column].to_numpy()
        if not np.isfinite(values).all():
            raise FeatureTableError(f"feature column contains NaN or infinite values: {column}")

    if canonical_split is not None:
        if isinstance(canonical_split, pd.DataFrame):
            canonical = validate_feature_table(
                canonical_split, canonical_split=None, require_feature_columns=False
            )
        else:
            canonical = load_canonical_split(canonical_split)
        canonical = canonical.set_index("sample_id")
        missing = sorted(set(table["sample_id"]).difference(canonical.index), key=int)
        if missing:
            raise FeatureTableError(f"sample IDs missing from canonical split: {missing[:10]}")
        for row in table.itertuples(index=False):
            reference = canonical.loc[row.sample_id]
            mismatches = [
                column
                for column in ("patient_id", "label", "split")
                if getattr(row, column) != reference[column]
            ]
            if mismatches:
                raise FeatureTableError(
                    f"metadata mismatch for sample_id {row.sample_id}: {', '.join(mismatches)}"
                )

    return table.sort_values("sample_id", key=lambda column: column.astype(int)).reset_index(drop=True)


def build_feature_dataframe(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Build a feature DataFrame from records and place metadata columns first."""

    frame = pd.DataFrame(list(records))
    return _prepare_frame(frame)


def write_feature_table(
    frame: pd.DataFrame,
    path: Path | str,
    *,
    canonical_split: pd.DataFrame | Path | str | None = "data/splits/patient_split.csv",
) -> pd.DataFrame:
    """Validate and atomically write a deterministic feature CSV."""

    validated = validate_feature_table(frame, canonical_split=canonical_split)
    _atomic_csv(validated, Path(path))
    return validated


def read_feature_table(
    path: Path | str,
    *,
    canonical_split: pd.DataFrame | Path | str | None = "data/splits/patient_split.csv",
) -> pd.DataFrame:
    """Read, validate, and return a deterministic feature CSV."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FeatureTableError(f"feature CSV does not exist: {csv_path}")
    frame = pd.read_csv(csv_path, dtype={"patient_id": str})
    return validate_feature_table(frame, canonical_split=canonical_split)


def merge_feature_tables(
    tables: Iterable[pd.DataFrame],
    *,
    require_identical_samples: bool = True,
    canonical_split: pd.DataFrame | Path | str | None = "data/splits/patient_split.csv",
) -> pd.DataFrame:
    """Merge compatible feature tables by sample_id with strict metadata agreement."""

    validated_tables = [
        validate_feature_table(table, canonical_split=canonical_split) for table in tables
    ]
    if not validated_tables:
        raise FeatureTableError("at least one feature table is required")

    base = validated_tables[0].copy()
    base_ids = set(base["sample_id"])
    feature_columns_seen = [column for column in base.columns if column not in METADATA_COLUMNS]
    if len(feature_columns_seen) != len(set(feature_columns_seen)):
        raise FeatureTableError("duplicate feature names in first table")

    for table in validated_tables[1:]:
        table_ids = set(table["sample_id"])
        if require_identical_samples and table_ids != base_ids:
            raise FeatureTableError("feature tables have mismatched sample coverage")
        overlapping_features = sorted(
            set(feature_columns_seen).intersection(
                column for column in table.columns if column not in METADATA_COLUMNS
            )
        )
        if overlapping_features:
            raise FeatureTableError(
                f"duplicate feature names across feature tables: {overlapping_features[:10]}"
            )

        metadata = table.loc[:, METADATA_COLUMNS]
        comparison = base.loc[:, METADATA_COLUMNS].merge(
            metadata,
            on="sample_id",
            how="inner",
            suffixes=("_base", "_next"),
            validate="one_to_one",
        )
        for column in ("patient_id", "label", "split"):
            mismatch = comparison[f"{column}_base"] != comparison[f"{column}_next"]
            if mismatch.any():
                sample_id = comparison.loc[mismatch, "sample_id"].iloc[0]
                raise FeatureTableError(f"metadata mismatch for sample_id {sample_id}: {column}")

        new_feature_columns = [column for column in table.columns if column not in METADATA_COLUMNS]
        base = base.merge(
            table.loc[:, ["sample_id"] + new_feature_columns],
            on="sample_id",
            how="inner" if not require_identical_samples else "left",
            validate="one_to_one",
        )
        feature_columns_seen.extend(new_feature_columns)

    return validate_feature_table(
        base.loc[:, METADATA_COLUMNS + feature_columns_seen],
        canonical_split=canonical_split,
    )


def build_feature_metadata(
    *,
    feature_set_name: str,
    feature_version: str,
    algorithm: str,
    parameters: dict[str, Any],
    input_representation: str,
    roi_policy: str,
    source_split_version: str,
    sample_count: int,
    feature_columns: list[str],
    code_commit: str,
    validation_summary: dict[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build metadata JSON payload for one feature extractor."""

    return {
        "feature_set_name": feature_set_name,
        "feature_version": feature_version,
        "algorithm": algorithm,
        "parameters": parameters,
        "input_representation": input_representation,
        "roi_policy": roi_policy,
        "source_split_version": source_split_version,
        "sample_count": int(sample_count),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "validation_summary": validation_summary,
    }


def validate_feature_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate lightweight feature metadata JSON."""

    missing = sorted(REQUIRED_FEATURE_METADATA_FIELDS.difference(metadata))
    if missing:
        raise FeatureTableError(f"feature metadata missing field(s): {', '.join(missing)}")
    if not isinstance(metadata["parameters"], dict):
        raise FeatureTableError("feature metadata parameters must be a JSON object")
    if not isinstance(metadata["validation_summary"], dict):
        raise FeatureTableError("feature metadata validation_summary must be a JSON object")
    if not isinstance(metadata["feature_columns"], list) or not all(
        isinstance(column, str) for column in metadata["feature_columns"]
    ):
        raise FeatureTableError("feature metadata feature_columns must be a list of strings")
    if int(metadata["feature_count"]) != len(metadata["feature_columns"]):
        raise FeatureTableError("feature_count does not match feature_columns length")
    if int(metadata["sample_count"]) < 0:
        raise FeatureTableError("sample_count must be non-negative")
    for field in (
        "feature_set_name",
        "feature_version",
        "algorithm",
        "input_representation",
        "roi_policy",
        "source_split_version",
        "generated_at",
        "code_commit",
    ):
        if not str(metadata[field]).strip():
            raise FeatureTableError(f"feature metadata {field} must be populated")
    return metadata


def write_feature_metadata(metadata: dict[str, Any], path: Path | str) -> dict[str, Any]:
    """Validate and atomically write feature metadata JSON."""

    validated = validate_feature_metadata(metadata)
    _atomic_json(validated, Path(path))
    return validated


def read_feature_metadata(path: Path | str) -> dict[str, Any]:
    """Read and validate feature metadata JSON."""

    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FeatureTableError(f"feature metadata JSON does not exist: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return validate_feature_metadata(metadata)
