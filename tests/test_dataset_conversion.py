from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.io import savemat

from src.data.convert_dataset import NPZ_KEYS, convert_dataset, validate_npz


def _write_mat(
    path: Path,
    *,
    shape: tuple[int, int] = (12, 10),
    label: int = 1,
    patient_id: str = "MR017260F",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1:4, 2:5] = 1
    savemat(
        path,
        {
            "cjdata": {
                "label": label,
                "PID": patient_id,
                "image": image,
                "tumorBorder": np.array([2.0, 1.0, 4.0, 3.0]),
                "tumorMask": mask,
            }
        },
    )


def _raw_manifest(path: Path, records: list[dict]) -> Path:
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _run(
    tmp_path: Path,
    records: list[dict],
    **kwargs,
) -> tuple[dict, Path, Path]:
    output = tmp_path / "processed"
    report = tmp_path / "conversion.json"
    raw_manifest = _raw_manifest(tmp_path / "raw_manifest.csv", records)
    result = convert_dataset(
        tmp_path / "raw", raw_manifest, output, report, **kwargs
    )
    return result, output, report


def test_valid_sample_round_trip_preserves_values_and_identifiers(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_mat(raw / "1.mat", label=3, patient_id="MR017260F")
    records = [{"sample_id": 1, "patient_id": "MR017260F", "label": 3}]

    result, output, _ = _run(tmp_path, records)

    assert result["successfully_converted_samples"] == 1
    assert result["failed_samples"] == 0
    npz_path = output / "samples" / "1.npz"
    stats = validate_npz(npz_path)
    assert stats["patient_id"] == "MR017260F"
    assert stats["label"] == 3
    with np.load(npz_path, allow_pickle=False) as archive:
        assert set(archive.files) == NPZ_KEYS
        assert archive["image_raw"].dtype == np.float32
        assert archive["image_normalized"].dtype == np.float32
        assert archive["tumor_mask"].dtype == np.uint8
        assert str(archive["patient_id"].item()) == "MR017260F"
        assert str(archive["sample_id"].item()) == "1"
        assert int(archive["label"].item()) == 3
        assert set(np.unique(archive["tumor_mask"]).tolist()) == {0, 1}
    with Image.open(output / "images" / "1.png") as image_png:
        assert image_png.size == (10, 12)
    with Image.open(output / "masks" / "1.png") as mask_png:
        assert set(np.unique(np.asarray(mask_png)).tolist()) == {0, 255}


@pytest.mark.parametrize("shape", [(512, 512), (256, 256)])
def test_native_dimensions_are_preserved(tmp_path: Path, shape: tuple[int, int]) -> None:
    raw = tmp_path / "raw"
    _write_mat(raw / "1.mat", shape=shape)
    records = [{"sample_id": 1, "patient_id": "MR017260F", "label": 1}]

    _, output, _ = _run(tmp_path, records)

    with np.load(output / "samples" / "1.npz", allow_pickle=False) as archive:
        assert archive["image_raw"].shape == shape
        assert archive["image_normalized"].shape == shape
        assert archive["tumor_mask"].shape == shape


def test_existing_valid_output_is_skipped(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_mat(raw / "1.mat")
    records = [{"sample_id": 1, "patient_id": "MR017260F", "label": 1}]
    _run(tmp_path, records)

    result, _, _ = _run(tmp_path, records)

    assert result["skipped_existing_valid_samples"] == 1
    assert result["newly_converted_samples"] == 0


def test_overwrite_forces_regeneration(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_mat(raw / "1.mat")
    records = [{"sample_id": 1, "patient_id": "MR017260F", "label": 1}]
    _run(tmp_path, records)

    result, _, _ = _run(tmp_path, records, overwrite=True)

    assert result["overwritten_samples"] == 1
    assert result["skipped_existing_valid_samples"] == 0


def test_incomplete_output_is_regenerated(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_mat(raw / "1.mat")
    records = [{"sample_id": 1, "patient_id": "MR017260F", "label": 1}]
    _, output, _ = _run(tmp_path, records)
    (output / "masks" / "1.png").unlink()

    result, output, _ = _run(tmp_path, records)

    assert result["regenerated_invalid_or_incomplete_samples"] == 1
    assert (output / "masks" / "1.png").is_file()
    validate_npz(output / "samples" / "1.npz")


def test_malformed_sample_failure_does_not_stop_conversion(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_mat(raw / "1.mat")
    savemat(raw / "2.mat", {"not_cjdata": np.array([1])})
    records = [
        {"sample_id": 1, "patient_id": "MR017260F", "label": 1},
        {"sample_id": 2, "patient_id": "unknown", "label": 2},
    ]

    result, output, _ = _run(tmp_path, records)

    assert result["successfully_converted_samples"] == 1
    assert result["failed_samples"] == 1
    assert result["failures"][0]["sample_id"] == "2"
    assert (output / "samples" / "1.npz").is_file()
    manifest = pd.read_csv(output / "manifest.csv")
    assert manifest["conversion_status"].tolist() == ["converted", "failed"]


def test_sample_ids_and_limit_select_deterministically(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    records = []
    for sample_id in (1, 2, 3):
        _write_mat(raw / f"{sample_id}.mat", patient_id=f"P{sample_id}")
        records.append({"sample_id": sample_id, "patient_id": f"P{sample_id}", "label": 1})

    result, output, _ = _run(tmp_path, records, sample_ids=[3, 1], limit=1)

    assert result["selected_samples"] == 1
    assert (output / "samples" / "3.npz").is_file()
    assert not (output / "samples" / "1.npz").exists()
