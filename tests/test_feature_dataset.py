from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.feature_dataset import (
    FeatureDatasetError,
    get_split_records,
    iter_phase1_samples,
    load_phase1_sample,
    load_split_index,
)


def _write_split(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _base_rows() -> list[dict]:
    return [
        {"sample_id": 1, "patient_id": "P1", "label": 1, "split": "train"},
        {"sample_id": 2, "patient_id": "P2", "label": 2, "split": "val"},
        {"sample_id": 10, "patient_id": "P3", "label": 3, "split": "test"},
    ]


def _arrays(shape: tuple[int, int] = (4, 5)) -> dict[str, np.ndarray]:
    image_raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    image_normalized = image_raw / max(float(image_raw.max()), 1.0)
    tumor_mask = np.zeros(shape, dtype=np.uint8)
    tumor_mask[1:3, 2:4] = 1
    return {
        "image_raw": image_raw,
        "image_normalized": image_normalized.astype(np.float32),
        "tumor_mask": tumor_mask,
        "tumor_border": np.array([2.0, 1.0, 4.0, 3.0]),
    }


def _write_npz(
    samples_dir: Path,
    sample_id: int,
    patient_id: str,
    label: int,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    omit_key: str | None = None,
    npz_sample_id: str | None = None,
    npz_patient_id: str | None = None,
    npz_label: int | None = None,
) -> Path:
    samples_dir.mkdir(parents=True, exist_ok=True)
    payload = _arrays() if arrays is None else arrays
    payload = {
        **payload,
        "label": np.asarray(label if npz_label is None else npz_label, dtype=np.int64),
        "patient_id": np.asarray(patient_id if npz_patient_id is None else npz_patient_id),
        "sample_id": np.asarray(str(sample_id) if npz_sample_id is None else npz_sample_id),
    }
    if omit_key is not None:
        del payload[omit_key]
    path = samples_dir / f"{sample_id}.npz"
    np.savez_compressed(path, **payload)
    return path


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    split_csv = _write_split(tmp_path / "patient_split.csv", _base_rows())
    samples_dir = tmp_path / "samples"
    _write_npz(samples_dir, 1, "P1", 1)
    _write_npz(samples_dir, 2, "P2", 2)
    _write_npz(samples_dir, 10, "P3", 3)
    return split_csv, samples_dir


def test_valid_sample_loads_with_metadata_and_arrays(tmp_path: Path) -> None:
    split_csv, samples_dir = _dataset(tmp_path)

    sample = load_phase1_sample(1, split_csv=split_csv, samples_dir=samples_dir)

    assert sample.sample_id == "1"
    assert sample.patient_id == "P1"
    assert sample.label == 1
    assert sample.split == "train"
    assert sample.image_normalized.shape == (4, 5)
    assert sample.image_normalized.dtype == np.float32
    np.testing.assert_array_equal(sample.tumor_mask, _arrays()["tumor_mask"])
    np.testing.assert_array_equal(sample.image_raw, _arrays()["image_raw"])


def test_metadata_records_do_not_load_arrays(tmp_path: Path) -> None:
    split_csv, _ = _dataset(tmp_path)

    records = get_split_records("train", split_csv=split_csv)

    assert records == [{"sample_id": "1", "patient_id": "P1", "label": 1, "split": "train"}]


def test_iteration_uses_numeric_order_and_split_filters(tmp_path: Path) -> None:
    split_csv, samples_dir = _dataset(tmp_path)

    assert [s.sample_id for s in iter_phase1_samples("all", split_csv=split_csv, samples_dir=samples_dir)] == [
        "1",
        "2",
        "10",
    ]
    assert [s.sample_id for s in iter_phase1_samples("train", split_csv=split_csv, samples_dir=samples_dir)] == ["1"]
    assert [s.sample_id for s in iter_phase1_samples("val", split_csv=split_csv, samples_dir=samples_dir)] == ["2"]
    assert [s.sample_id for s in iter_phase1_samples("test", split_csv=split_csv, samples_dir=samples_dir)] == ["10"]


def test_duplicate_sample_id_in_split_is_rejected(tmp_path: Path) -> None:
    rows = _base_rows()
    rows[1]["sample_id"] = 1

    with pytest.raises(FeatureDatasetError, match="duplicate sample IDs"):
        load_split_index(_write_split(tmp_path / "patient_split.csv", rows))


def test_invalid_split_name_is_rejected(tmp_path: Path) -> None:
    split_csv, samples_dir = _dataset(tmp_path)

    with pytest.raises(FeatureDatasetError, match="split must be one of"):
        list(iter_phase1_samples("dev", split_csv=split_csv, samples_dir=samples_dir))


def test_patient_in_multiple_splits_is_rejected(tmp_path: Path) -> None:
    rows = _base_rows()
    rows[1]["patient_id"] = "P1"

    with pytest.raises(FeatureDatasetError, match="multiple splits"):
        load_split_index(_write_split(tmp_path / "patient_split.csv", rows))


def test_patient_with_multiple_labels_is_rejected(tmp_path: Path) -> None:
    rows = _base_rows()
    rows[1]["patient_id"] = "P1"
    rows[1]["split"] = "train"

    with pytest.raises(FeatureDatasetError, match="multiple labels"):
        load_split_index(_write_split(tmp_path / "patient_split.csv", rows))


def test_npz_sample_id_mismatch_is_rejected(tmp_path: Path) -> None:
    split_csv = _write_split(tmp_path / "patient_split.csv", _base_rows())
    samples_dir = tmp_path / "samples"
    _write_npz(samples_dir, 1, "P1", 1, npz_sample_id="99")

    with pytest.raises(FeatureDatasetError, match="NPZ sample_id"):
        load_phase1_sample(1, split_csv=split_csv, samples_dir=samples_dir)


def test_npz_patient_id_mismatch_is_rejected(tmp_path: Path) -> None:
    split_csv = _write_split(tmp_path / "patient_split.csv", _base_rows())
    samples_dir = tmp_path / "samples"
    _write_npz(samples_dir, 1, "P1", 1, npz_patient_id="PX")

    with pytest.raises(FeatureDatasetError, match="NPZ patient_id"):
        load_phase1_sample(1, split_csv=split_csv, samples_dir=samples_dir)


def test_npz_label_mismatch_is_rejected(tmp_path: Path) -> None:
    split_csv = _write_split(tmp_path / "patient_split.csv", _base_rows())
    samples_dir = tmp_path / "samples"
    _write_npz(samples_dir, 1, "P1", 1, npz_label=2)

    with pytest.raises(FeatureDatasetError, match="NPZ label"):
        load_phase1_sample(1, split_csv=split_csv, samples_dir=samples_dir)


def test_missing_npz_required_key_is_rejected(tmp_path: Path) -> None:
    split_csv = _write_split(tmp_path / "patient_split.csv", _base_rows())
    samples_dir = tmp_path / "samples"
    _write_npz(samples_dir, 1, "P1", 1, omit_key="image_normalized")

    with pytest.raises(FeatureDatasetError, match="missing required key"):
        load_phase1_sample(1, split_csv=split_csv, samples_dir=samples_dir)


def test_non_binary_mask_is_rejected(tmp_path: Path) -> None:
    split_csv = _write_split(tmp_path / "patient_split.csv", _base_rows())
    samples_dir = tmp_path / "samples"
    arrays = _arrays()
    arrays["tumor_mask"][0, 0] = 2
    _write_npz(samples_dir, 1, "P1", 1, arrays=arrays)

    with pytest.raises(FeatureDatasetError, match="not binary"):
        load_phase1_sample(1, split_csv=split_csv, samples_dir=samples_dir)


def test_empty_mask_is_rejected(tmp_path: Path) -> None:
    split_csv = _write_split(tmp_path / "patient_split.csv", _base_rows())
    samples_dir = tmp_path / "samples"
    arrays = _arrays()
    arrays["tumor_mask"] = np.zeros((4, 5), dtype=np.uint8)
    _write_npz(samples_dir, 1, "P1", 1, arrays=arrays)

    with pytest.raises(FeatureDatasetError, match="empty"):
        load_phase1_sample(1, split_csv=split_csv, samples_dir=samples_dir)


def test_image_mask_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    split_csv = _write_split(tmp_path / "patient_split.csv", _base_rows())
    samples_dir = tmp_path / "samples"
    arrays = _arrays()
    arrays["tumor_mask"] = np.ones((3, 5), dtype=np.uint8)
    _write_npz(samples_dir, 1, "P1", 1, arrays=arrays)

    with pytest.raises(FeatureDatasetError, match="tumor_mask shapes differ"):
        load_phase1_sample(1, split_csv=split_csv, samples_dir=samples_dir)


def test_image_normalized_outside_unit_range_is_rejected(tmp_path: Path) -> None:
    split_csv = _write_split(tmp_path / "patient_split.csv", _base_rows())
    samples_dir = tmp_path / "samples"
    arrays = _arrays()
    arrays["image_normalized"][0, 0] = 1.5
    _write_npz(samples_dir, 1, "P1", 1, arrays=arrays)

    with pytest.raises(FeatureDatasetError, match="outside"):
        load_phase1_sample(1, split_csv=split_csv, samples_dir=samples_dir)


@pytest.mark.parametrize("field", ["image_raw", "image_normalized"])
def test_nan_or_inf_image_values_are_rejected(tmp_path: Path, field: str) -> None:
    split_csv = _write_split(tmp_path / "patient_split.csv", _base_rows())
    samples_dir = tmp_path / "samples"
    arrays = _arrays()
    arrays[field][0, 0] = np.nan
    _write_npz(samples_dir, 1, "P1", 1, arrays=arrays)

    with pytest.raises(FeatureDatasetError, match="NaN or infinite"):
        load_phase1_sample(1, split_csv=split_csv, samples_dir=samples_dir)


def test_missing_npz_file_raises_descriptive_error(tmp_path: Path) -> None:
    split_csv = _write_split(tmp_path / "patient_split.csv", _base_rows())

    with pytest.raises(FeatureDatasetError, match="NPZ file does not exist"):
        load_phase1_sample(1, split_csv=split_csv, samples_dir=tmp_path / "samples")
