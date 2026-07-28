from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from src.data.mat_loader import MatSampleError, find_mat_sample_files, load_mat_sample


def _valid_cjdata(
    *,
    label: int = 1,
    patient_id: str = "PID001",
    image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> dict:
    image = np.arange(20, dtype=np.int16).reshape(4, 5) if image is None else image
    mask = np.zeros(image.shape, dtype=np.uint8) if mask is None else mask
    mask[1:3, 2:4] = 1
    return {
        "label": np.array([[label]], dtype=np.float64),
        "PID": np.array(list(patient_id), dtype="U1"),
        "image": image,
        "tumorBorder": np.array([[2.0, 1.0, 3.0, 2.0]]),
        "tumorMask": mask,
    }


def _save_sample(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"cjdata": _valid_cjdata(**kwargs)})


def test_find_files_excludes_cvind_and_sorts_numerically(tmp_path: Path) -> None:
    for name in ("10.mat", "2.mat", "1.mat", "cvind.mat", "notes.mat"):
        _save_sample(tmp_path / "nested" / name)

    files = find_mat_sample_files(tmp_path)

    assert [path.name for path in files] == ["1.mat", "2.mat", "10.mat"]


def test_load_ordinary_mat_converts_types_and_values(tmp_path: Path) -> None:
    path = tmp_path / "7.mat"
    _save_sample(path, label=2, patient_id="P007")

    sample = load_mat_sample(path)

    assert sample.sample_id == "7"
    assert sample.patient_id == "P007"
    assert sample.label == 2
    assert sample.image.dtype == np.float32
    assert sample.tumour_mask.dtype == np.bool_
    assert sample.image.shape == sample.tumour_mask.shape == (4, 5)
    assert sample.image_dtype_original == "int16"


def test_load_hdf5_v73_fallback(tmp_path: Path) -> None:
    path = tmp_path / "8.mat"
    image = np.arange(20, dtype=np.int16).reshape(5, 4)  # Stored reversed as MATLAB does.
    mask = np.zeros((5, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    with h5py.File(path, "w") as handle:
        group = handle.create_group("cjdata")
        group.create_dataset("label", data=np.array([[3.0]]))
        pid = group.create_dataset(
            "PID", data=np.array([[ord(char)] for char in "H5P008"], dtype=np.uint16)
        )
        pid.attrs["MATLAB_class"] = np.bytes_("char")
        group.create_dataset("image", data=image)
        group.create_dataset("tumorMask", data=mask)
        group.create_dataset("tumorBorder", data=np.array([[1.0, 1.0, 2.0, 2.0]]))

    sample = load_mat_sample(path)

    assert sample.label == 3
    assert sample.patient_id == "H5P008"
    assert sample.image.shape == sample.tumour_mask.shape == (4, 5)
    np.testing.assert_array_equal(sample.image, image.T.astype(np.float32))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"label": 4}, "label must be one of"),
        ({"image": np.ones((2, 2, 2))}, "image must be two-dimensional"),
        (
            {
                "image": np.ones((3, 4)),
                "mask": np.zeros((4, 3), dtype=np.uint8),
            },
            "shapes differ",
        ),
        ({"image": np.array([[np.nan, 0.0], [1.0, 2.0]])}, "NaN or infinite"),
    ],
)
def test_validation_errors_are_descriptive(
    tmp_path: Path, change: dict, message: str
) -> None:
    path = tmp_path / "9.mat"
    _save_sample(path, **change)

    with pytest.raises(MatSampleError, match=message):
        load_mat_sample(path)


def test_missing_cjdata_and_required_field_are_reported(tmp_path: Path) -> None:
    missing_structure = tmp_path / "11.mat"
    savemat(missing_structure, {"other": np.array([1])})
    with pytest.raises(MatSampleError, match="missing required MATLAB structure"):
        load_mat_sample(missing_structure)

    missing_field = tmp_path / "12.mat"
    data = _valid_cjdata()
    del data["tumorMask"]
    savemat(missing_field, {"cjdata": data})
    with pytest.raises(MatSampleError, match="tumorMask"):
        load_mat_sample(missing_field)
