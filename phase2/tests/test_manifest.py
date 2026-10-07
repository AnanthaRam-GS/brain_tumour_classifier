import numpy as np
import pandas as pd
import pytest

from btdl.data.manifest import (
    build_manifest,
    compute_mask_bbox,
    content_sha256,
    discover_raw_files,
    verify_manifest,
)


def _write_npz(path, *, sample_id="1", patient_id="100360", label=1, image=None, mask=None):
    if image is None:
        image = np.zeros((8, 8), dtype=np.float32)
    if mask is None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 3:6] = 1
    np.savez(
        path,
        image_raw=image.astype(np.float32),
        image_normalized=image.astype(np.float32),
        tumor_mask=mask.astype(np.uint8),
        tumor_border=np.array([1.0, 2.0]),
        label=np.array(label),
        patient_id=np.array(patient_id),
        sample_id=np.array(sample_id),
    )


def test_content_sha256_is_stable(tmp_path):
    path = tmp_path / "1.npz"
    _write_npz(path)
    assert content_sha256(path) == content_sha256(path)


def test_content_sha256_changes_when_image_changes(tmp_path):
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    image_a = np.zeros((8, 8), dtype=np.float32)
    image_b = image_a.copy()
    image_b[0, 0] = 1.0
    _write_npz(path_a, image=image_a)
    _write_npz(path_b, image=image_b)
    assert content_sha256(path_a) != content_sha256(path_b)


def test_content_sha256_changes_when_label_changes(tmp_path):
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    _write_npz(path_a, label=1)
    _write_npz(path_b, label=2)
    assert content_sha256(path_a) != content_sha256(path_b)


def test_content_sha256_unaffected_by_npz_resave(tmp_path):
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "subdir" / "a_resaved.npz"
    path_b.parent.mkdir()
    _write_npz(path_a)

    # Re-save the identical arrays under a different filename/location.
    with np.load(path_a) as data:
        np.savez(path_b, **{key: data[key] for key in data.files})

    assert content_sha256(path_a) == content_sha256(path_b)


def test_compute_mask_bbox_tight_box():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:6] = 1  # rows 2,3,4 ; cols 3,4,5
    bbox = compute_mask_bbox(mask)
    assert bbox == (2, 5, 3, 6)


def test_compute_mask_bbox_empty_mask_returns_none():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert compute_mask_bbox(mask) is None


def test_discover_raw_files_excludes_cvind_and_non_digit_stems(tmp_path):
    raw_dir = tmp_path / "raw"
    sub = raw_dir / "sub"
    sub.mkdir(parents=True)
    (raw_dir / "cvind.mat").write_bytes(b"x")
    (sub / "1.mat").write_bytes(b"x")
    (sub / "2.mat").write_bytes(b"x")
    (sub / "notanumber.mat").write_bytes(b"x")

    mapping = discover_raw_files(raw_dir)
    assert set(mapping.keys()) == {"1", "2"}


def _build_tiny_dataset(tmp_path):
    data_root = tmp_path
    samples_dir = data_root / "data" / "processed" / "samples"
    samples_dir.mkdir(parents=True)
    raw_dir = data_root / "raw"
    raw_dir.mkdir()

    for sample_id, patient_id, label in [("1", "100360", 1), ("2", "100360", 1), ("3", "200000", 2)]:
        _write_npz(samples_dir / f"{sample_id}.npz", sample_id=sample_id, patient_id=patient_id, label=label)
        (raw_dir / f"{sample_id}.mat").write_bytes(b"fake-raw-bytes")

    split_csv = data_root / "split.csv"
    pd.DataFrame(
        {
            "sample_id": ["1", "2", "3"],
            "patient_id": ["100360", "100360", "200000"],
            "label": [1, 1, 2],
            "split": ["train", "train", "val"],
        }
    ).to_csv(split_csv, index=False)

    return data_root, samples_dir, raw_dir, split_csv


def test_build_manifest_is_deterministic_across_runs(tmp_path):
    data_root, samples_dir, raw_dir, split_csv = _build_tiny_dataset(tmp_path)

    first = build_manifest(samples_dir=samples_dir, raw_dir=raw_dir, split_csv=split_csv, data_root=data_root)
    second = build_manifest(samples_dir=samples_dir, raw_dir=raw_dir, split_csv=split_csv, data_root=data_root)

    pd.testing.assert_frame_equal(first, second)


def test_build_manifest_sorted_by_numeric_sample_id(tmp_path):
    data_root, samples_dir, raw_dir, split_csv = _build_tiny_dataset(tmp_path)
    manifest = build_manifest(samples_dir=samples_dir, raw_dir=raw_dir, split_csv=split_csv, data_root=data_root)
    assert list(manifest["sample_id"]) == ["1", "2", "3"]


def test_build_manifest_columns_and_split_lookup(tmp_path):
    data_root, samples_dir, raw_dir, split_csv = _build_tiny_dataset(tmp_path)
    manifest = build_manifest(samples_dir=samples_dir, raw_dir=raw_dir, split_csv=split_csv, data_root=data_root)
    row = manifest[manifest["sample_id"] == "3"].iloc[0]
    assert row["split"] == "val"
    assert row["patient_id"] == "200000"
    assert row["label"] == 2
    assert row["mask_area"] > 0
    assert row["bbox_longest"] >= row["bbox_h"]
    assert row["bbox_longest"] >= row["bbox_w"]


def test_verify_manifest_detects_mismatch(tmp_path):
    data_root, samples_dir, raw_dir, split_csv = _build_tiny_dataset(tmp_path)
    manifest = build_manifest(samples_dir=samples_dir, raw_dir=raw_dir, split_csv=split_csv, data_root=data_root)

    # Corrupt sample 2's NPZ in place.
    _write_npz(samples_dir / "2.npz", sample_id="2", patient_id="100360", label=2)

    mismatches = verify_manifest(manifest, data_root)
    assert set(mismatches["sample_id"]) == {"2"}


def test_verify_manifest_detects_missing_file(tmp_path):
    data_root, samples_dir, raw_dir, split_csv = _build_tiny_dataset(tmp_path)
    manifest = build_manifest(samples_dir=samples_dir, raw_dir=raw_dir, split_csv=split_csv, data_root=data_root)

    (samples_dir / "3.npz").unlink()

    mismatches = verify_manifest(manifest, data_root)
    assert "3" in set(mismatches["sample_id"])
