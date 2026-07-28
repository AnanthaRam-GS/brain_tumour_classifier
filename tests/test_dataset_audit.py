import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from src.data.audit_dataset import audit_dataset


def _write_sample(path: Path, label: int, patient_id: str, empty_mask: bool = False) -> None:
    image = np.arange(30, dtype=np.int16).reshape(5, 6)
    mask = np.zeros_like(image, dtype=np.uint8)
    if not empty_mask:
        mask[1:4, 2:5] = 1
    savemat(
        path,
        {
            "cjdata": {
                "label": label,
                "PID": patient_id,
                "image": image,
                "tumorBorder": np.array([2, 1, 4, 3], dtype=float),
                "tumorMask": mask,
            }
        },
    )


def test_audit_continues_after_invalid_file_and_writes_outputs(tmp_path: Path) -> None:
    dataset = tmp_path / "raw"
    reports = tmp_path / "reports"
    dataset.mkdir()
    _write_sample(dataset / "1.mat", 1, "P1")
    _write_sample(dataset / "2.mat", 2, "P1", empty_mask=True)
    _write_sample(dataset / "3.mat", 3, "P3")
    savemat(dataset / "4.mat", {"not_cjdata": np.array([1])})
    savemat(dataset / "cvind.mat", {"cvind": np.arange(4)})

    result = audit_dataset(dataset, reports, seed=17)

    assert result["total_mat_sample_files"] == 4
    assert result["valid_files"] == 3
    assert result["invalid_files"] == 1
    assert result["unique_patients"] == 2
    assert result["duplicate_sample_ids"] == {}
    assert len(result["empty_masks"]) == 1
    assert len(result["malformed_files"]) == 1
    assert len(result["patients_with_multiple_class_labels"]) == 1

    expected = {
        "dataset_audit.json",
        "sample_manifest_raw.csv",
        "class_distribution.csv",
        "patient_distribution.csv",
        "image_shape_distribution.csv",
    }
    assert expected.issubset(path.name for path in reports.iterdir())
    manifest = pd.read_csv(reports / "sample_manifest_raw.csv")
    assert list(manifest.columns) == [
        "sample_id", "source_path", "patient_id", "label", "class_name",
        "image_height", "image_width", "image_dtype_original", "image_min",
        "image_max", "image_mean", "image_std", "mask_height", "mask_width",
        "tumour_pixels", "tumour_fraction", "has_empty_mask",
        "validation_status", "validation_message",
    ]
    assert manifest["validation_status"].tolist() == ["valid", "valid", "valid", "invalid"]
    with (reports / "dataset_audit.json").open(encoding="utf-8") as handle:
        assert json.load(handle)["random_seed"] == 17


def test_audit_reports_duplicate_numeric_sample_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "raw"
    reports = tmp_path / "reports"
    (dataset / "a").mkdir(parents=True)
    (dataset / "b").mkdir()
    _write_sample(dataset / "a" / "1.mat", 1, "P1")
    _write_sample(dataset / "b" / "1.mat", 1, "P2")

    result = audit_dataset(dataset, reports)

    assert result["duplicate_sample_ids"] == {"1": 2}
