from pathlib import Path

import pandas as pd
import pytest

from src.data.create_patient_split import (
    PatientSplitError,
    assign_patient_splits,
    build_sample_split,
    create_patient_split,
    load_manifest,
    make_patient_table,
    validate_split,
)


def _synthetic_manifest() -> pd.DataFrame:
    rows = []
    sample_id = 1
    for label in (1, 2, 3):
        for patient_index in range(12):
            patient_id = f"C{label}_P{patient_index:02d}"
            for _ in range((patient_index % 3) + 1):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "patient_id": patient_id,
                        "label": label,
                    }
                )
                sample_id += 1
    return pd.DataFrame(rows)


def _write_manifest(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    return path


def test_patient_split_has_schema_coverage_and_no_leakage(tmp_path: Path) -> None:
    manifest = _synthetic_manifest()
    path = _write_manifest(tmp_path, manifest)

    sample_split, metadata = create_patient_split(
        path,
        tmp_path / "patient_split.csv",
        tmp_path / "split_metadata.json",
        seed=42,
    )

    assert list(sample_split.columns) == ["sample_id", "patient_id", "label", "split"]
    assert len(sample_split) == len(manifest)
    assert set(sample_split["sample_id"]) == set(manifest["sample_id"])
    assert sample_split["sample_id"].is_unique
    assert set(sample_split["split"]) == {"train", "val", "test"}
    assert sample_split.groupby("patient_id")["split"].nunique().max() == 1
    assert metadata["cvind_policy"].startswith("Historical/reference only")
    assert (tmp_path / "patient_split.csv").is_file()
    assert (tmp_path / "split_metadata.json").is_file()


def test_split_is_deterministic_with_seed_42(tmp_path: Path) -> None:
    manifest = _synthetic_manifest()
    path = _write_manifest(tmp_path, manifest)

    first, _ = create_patient_split(
        path,
        tmp_path / "patient_split_1.csv",
        tmp_path / "metadata_1.json",
        seed=42,
    )
    second, _ = create_patient_split(
        path,
        tmp_path / "patient_split_2.csv",
        tmp_path / "metadata_2.json",
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_all_three_classes_are_present_in_each_split(tmp_path: Path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path, _synthetic_manifest()))
    patients = make_patient_table(manifest)
    patient_split = assign_patient_splits(patients, seed=42)
    sample_split = build_sample_split(manifest, patient_split)

    for split in ("train", "val", "test"):
        assert set(sample_split.loc[sample_split["split"].eq(split), "label"]) == {1, 2, 3}


def test_labels_are_unchanged_after_mapping_to_samples(tmp_path: Path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path, _synthetic_manifest()))
    patient_split = assign_patient_splits(make_patient_table(manifest), seed=42)
    sample_split = build_sample_split(manifest, patient_split)

    original = manifest.set_index("sample_id")["label"].sort_index()
    split_labels = sample_split.set_index("sample_id")["label"].sort_index()
    pd.testing.assert_series_equal(original, split_labels)
    validate_split(manifest, sample_split)


def test_invalid_patient_with_multiple_labels_raises_error(tmp_path: Path) -> None:
    manifest = _synthetic_manifest()
    patient_id = "C1_P01"
    rows = manifest["patient_id"].eq(patient_id)
    manifest.loc[rows, "label"] = [1, 2]

    with pytest.raises(PatientSplitError, match="multiple labels"):
        load_manifest(_write_manifest(tmp_path, manifest))


def test_missing_required_manifest_column_raises_error(tmp_path: Path) -> None:
    manifest = _synthetic_manifest().drop(columns=["patient_id"])

    with pytest.raises(PatientSplitError, match="missing required column"):
        load_manifest(_write_manifest(tmp_path, manifest))


def test_duplicate_sample_ids_are_rejected(tmp_path: Path) -> None:
    manifest = _synthetic_manifest()
    manifest.loc[1, "sample_id"] = manifest.loc[0, "sample_id"]

    with pytest.raises(PatientSplitError, match="duplicate sample_id"):
        load_manifest(_write_manifest(tmp_path, manifest))
