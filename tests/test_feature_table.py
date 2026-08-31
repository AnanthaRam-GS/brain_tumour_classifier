from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.feature_table import (
    FeatureTableError,
    build_feature_dataframe,
    build_feature_metadata,
    merge_feature_tables,
    read_feature_metadata,
    read_feature_table,
    validate_feature_table,
    write_feature_metadata,
    write_feature_table,
)


def _canonical() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": 1, "patient_id": "P1", "label": 1, "split": "train"},
            {"sample_id": 2, "patient_id": "P2", "label": 2, "split": "val"},
            {"sample_id": 10, "patient_id": "P3", "label": 3, "split": "test"},
        ]
    )


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": 10,
                "patient_id": "P3",
                "label": 3,
                "split": "test",
                "glcm_contrast_mean": 0.3,
                "glcm_energy_mean": 0.7,
            },
            {
                "sample_id": 1,
                "patient_id": "P1",
                "label": 1,
                "split": "train",
                "glcm_contrast_mean": 0.1,
                "glcm_energy_mean": 0.9,
            },
            {
                "sample_id": 2,
                "patient_id": "P2",
                "label": 2,
                "split": "val",
                "glcm_contrast_mean": 0.2,
                "glcm_energy_mean": 0.8,
            },
        ]
    )


def test_valid_feature_table_accepted() -> None:
    validated = validate_feature_table(_features(), canonical_split=_canonical())

    assert len(validated) == 3
    assert validated["sample_id"].tolist() == ["1", "2", "10"]


def test_metadata_columns_required() -> None:
    frame = _features().drop(columns=["patient_id"])

    with pytest.raises(FeatureTableError, match="missing metadata"):
        validate_feature_table(frame, canonical_split=_canonical())


def test_metadata_columns_appear_first_after_build_and_write(tmp_path: Path) -> None:
    records = _features().to_dict(orient="records")
    for record in records:
        record["extra_feature"] = record.pop("glcm_energy_mean")
    frame = build_feature_dataframe(records)
    path = tmp_path / "features.csv"

    written = write_feature_table(frame, path, canonical_split=_canonical())

    assert list(written.columns[:4]) == ["sample_id", "patient_id", "label", "split"]
    assert list(pd.read_csv(path).columns[:4]) == ["sample_id", "patient_id", "label", "split"]


def test_numeric_sample_ordering_is_deterministic() -> None:
    validated = validate_feature_table(_features(), canonical_split=_canonical())

    assert validated["sample_id"].tolist() == ["1", "2", "10"]


def test_duplicate_sample_ids_rejected() -> None:
    frame = pd.concat([_features(), _features().iloc[[0]]], ignore_index=True)

    with pytest.raises(FeatureTableError, match="duplicate sample IDs"):
        validate_feature_table(frame, canonical_split=None)


def test_invalid_label_rejected() -> None:
    frame = _features()
    frame.loc[0, "label"] = 4

    with pytest.raises(FeatureTableError, match="labels outside"):
        validate_feature_table(frame, canonical_split=None)


def test_invalid_split_rejected() -> None:
    frame = _features()
    frame.loc[0, "split"] = "dev"

    with pytest.raises(FeatureTableError, match="invalid split"):
        validate_feature_table(frame, canonical_split=None)


def test_blank_patient_id_rejected() -> None:
    frame = _features()
    frame.loc[0, "patient_id"] = " "

    with pytest.raises(FeatureTableError, match="blank patient"):
        validate_feature_table(frame, canonical_split=None)


def test_nonnumeric_feature_column_rejected() -> None:
    frame = _features()
    frame["glcm_bad"] = ["a", "b", "c"]

    with pytest.raises(FeatureTableError, match="not numeric"):
        validate_feature_table(frame, canonical_split=None)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_feature_values_rejected(value: float) -> None:
    frame = _features()
    frame.loc[0, "glcm_contrast_mean"] = value

    with pytest.raises(FeatureTableError, match="NaN or infinite"):
        validate_feature_table(frame, canonical_split=None)


def test_duplicate_feature_names_rejected_if_constructible() -> None:
    frame = pd.DataFrame(
        [
            [1, "P1", 1, "train", 0.1, 0.2],
            [2, "P2", 2, "val", 0.3, 0.4],
        ],
        columns=["sample_id", "patient_id", "label", "split", "glcm_a", "glcm_a"],
    )

    with pytest.raises(FeatureTableError, match="duplicate column"):
        validate_feature_table(frame, canonical_split=None)


def test_metadata_mismatch_against_canonical_split_rejected() -> None:
    frame = _features()
    frame.loc[0, "patient_id"] = "PX"

    with pytest.raises(FeatureTableError, match="metadata mismatch"):
        validate_feature_table(frame, canonical_split=_canonical())


def test_valid_canonical_subset_accepted() -> None:
    subset = _features().iloc[[0, 1]]

    validated = validate_feature_table(subset, canonical_split=_canonical())

    assert validated["sample_id"].tolist() == ["1", "10"]


def test_patient_in_multiple_splits_rejected() -> None:
    frame = _features()
    frame.loc[1, "patient_id"] = "P3"

    with pytest.raises(FeatureTableError, match="multiple splits"):
        validate_feature_table(frame, canonical_split=None)


def test_patient_with_multiple_labels_rejected() -> None:
    frame = _features()
    frame.loc[1, "patient_id"] = "P3"
    frame.loc[1, "split"] = "test"

    with pytest.raises(FeatureTableError, match="multiple labels"):
        validate_feature_table(frame, canonical_split=None)


def test_writing_and_reading_round_trip_preserves_values_schema(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"

    written = write_feature_table(_features(), path, canonical_split=_canonical())
    read = read_feature_table(path, canonical_split=_canonical())

    pd.testing.assert_frame_equal(written, read)


def test_merging_two_compatible_tables_succeeds() -> None:
    left = _features().drop(columns=["glcm_energy_mean"])
    right = _features().drop(columns=["glcm_contrast_mean"]).rename(
        columns={"glcm_energy_mean": "lbp_bin_00"}
    )

    merged = merge_feature_tables([left, right], canonical_split=_canonical())

    assert list(merged.columns) == [
        "sample_id",
        "patient_id",
        "label",
        "split",
        "glcm_contrast_mean",
        "lbp_bin_00",
    ]
    assert len(merged) == 3


def test_merged_metadata_appears_only_once() -> None:
    left = _features().drop(columns=["glcm_energy_mean"])
    right = _features().drop(columns=["glcm_contrast_mean"]).rename(
        columns={"glcm_energy_mean": "lbp_bin_00"}
    )

    merged = merge_feature_tables([left, right], canonical_split=_canonical())

    assert [column for column in merged.columns if column == "patient_id"] == ["patient_id"]
    assert [column for column in merged.columns if column == "label"] == ["label"]
    assert [column for column in merged.columns if column == "split"] == ["split"]


def test_merge_preserves_deterministic_feature_order() -> None:
    left = _features().drop(columns=["glcm_energy_mean"])
    right = _features().drop(columns=["glcm_contrast_mean"]).rename(
        columns={"glcm_energy_mean": "lbp_bin_00"}
    )

    merged = merge_feature_tables([right, left], canonical_split=_canonical())

    assert list(merged.columns[4:]) == ["lbp_bin_00", "glcm_contrast_mean"]


@pytest.mark.parametrize(
    ("column", "value"),
    [("patient_id", "PX"), ("label", 1), ("split", "train")],
)
def test_merge_rejects_metadata_mismatch(column: str, value: object) -> None:
    left = _features().drop(columns=["glcm_energy_mean"])
    right = _features().drop(columns=["glcm_contrast_mean"]).rename(
        columns={"glcm_energy_mean": "lbp_bin_00"}
    )
    right.loc[0, column] = value

    with pytest.raises(FeatureTableError, match="metadata mismatch"):
        merge_feature_tables([left, right], canonical_split=None)


def test_merge_rejects_mismatched_sample_coverage() -> None:
    left = _features().drop(columns=["glcm_energy_mean"])
    right = _features().drop(columns=["glcm_contrast_mean"]).iloc[:2].rename(
        columns={"glcm_energy_mean": "lbp_bin_00"}
    )

    with pytest.raises(FeatureTableError, match="mismatched sample coverage"):
        merge_feature_tables([left, right], canonical_split=None)


def test_merge_rejects_duplicate_feature_names_across_sets() -> None:
    left = _features().drop(columns=["glcm_energy_mean"])
    right = _features().drop(columns=["glcm_energy_mean"])

    with pytest.raises(FeatureTableError, match="duplicate feature names"):
        merge_feature_tables([left, right], canonical_split=None)


def test_feature_metadata_json_write_read_validation_works(tmp_path: Path) -> None:
    metadata = build_feature_metadata(
        feature_set_name="glcm",
        feature_version="v1",
        algorithm="GLCM",
        parameters={"levels": 32},
        input_representation="native_roi",
        roi_policy="phase1_roi_contract",
        source_split_version="phase1_patient_split_v1",
        sample_count=3,
        feature_columns=["glcm_contrast_mean", "glcm_energy_mean"],
        code_commit="abc123",
        validation_summary={"valid": True},
        generated_at="2026-08-31T00:00:00+00:00",
    )
    path = tmp_path / "glcm_metadata.json"

    written = write_feature_metadata(metadata, path)
    read = read_feature_metadata(path)

    assert written == read
    assert read["feature_count"] == 2
