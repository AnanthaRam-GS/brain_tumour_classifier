import h5py
import numpy as np
import pytest

from btdl.data.raw_reader import RawSampleError, read_raw_sample


def _write_synthetic_cjdata(
    path,
    *,
    label=2.0,
    pid_string="100360",
    image=None,
    mask=None,
    border=None,
    omit_fields=(),
):
    """Write a synthetic MATLAB-v7.3-style HDF5 file mimicking cjdata.

    Numeric 2D fields are written TRANSPOSED (column-major on-disk, as real
    MATLAB v7.3 files are) so read_raw_sample's transpose-back logic is
    exercised the same way it is against real data.
    """

    if image is None:
        image = np.arange(12, dtype=np.int16).reshape(3, 4)
    if mask is None:
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[1, 2] = 1
    if border is None:
        border = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    pid_codes = np.array([[ord(ch)] for ch in pid_string], dtype=np.uint16)

    with h5py.File(path, "w") as handle:
        group = handle.create_group("cjdata")
        fields = {
            "label": np.array([[label]], dtype=np.float64),
            "PID": pid_codes,
            "image": image.T,  # on-disk transposed
            "tumorBorder": border.reshape(1, -1),
            "tumorMask": mask.T,  # on-disk transposed
        }
        for name, value in fields.items():
            if name in omit_fields:
                continue
            group.create_dataset(name, data=value)

    return image, mask, border


def test_read_raw_sample_decodes_fields_correctly(tmp_path):
    path = tmp_path / "1.mat"
    image, mask, border = _write_synthetic_cjdata(path, label=2.0, pid_string="100360")

    sample = read_raw_sample(path)

    assert sample.label == 2
    assert sample.patient_id == "100360"
    assert np.array_equal(sample.image, image)
    assert np.array_equal(sample.tumor_mask.astype(np.uint8), mask)
    assert sample.tumor_mask.dtype == bool
    assert np.allclose(sample.tumor_border, border)


def test_read_raw_sample_strips_null_and_whitespace_from_pid(tmp_path):
    path = tmp_path / "2.mat"
    pid_codes = np.array(
        [[ord(ch)] for ch in "42"] + [[0], [0]], dtype=np.uint16
    )
    with h5py.File(path, "w") as handle:
        group = handle.create_group("cjdata")
        group.create_dataset("label", data=np.array([[1.0]]))
        group.create_dataset("PID", data=pid_codes)
        group.create_dataset("image", data=np.zeros((2, 2), dtype=np.int16))
        group.create_dataset("tumorMask", data=np.zeros((2, 2), dtype=np.uint8))
        group.create_dataset("tumorBorder", data=np.zeros((1, 2)))

    sample = read_raw_sample(path)
    assert sample.patient_id == "42"


@pytest.mark.parametrize("label_value", [1.0, 2.0, 3.0])
def test_read_raw_sample_accepts_all_valid_labels(tmp_path, label_value):
    path = tmp_path / "label.mat"
    _write_synthetic_cjdata(path, label=label_value)
    sample = read_raw_sample(path)
    assert sample.label == int(label_value)


def test_read_raw_sample_rejects_non_integral_label(tmp_path):
    path = tmp_path / "bad_label.mat"
    _write_synthetic_cjdata(path, label=1.5)
    with pytest.raises(RawSampleError):
        read_raw_sample(path)


def test_read_raw_sample_rejects_missing_field(tmp_path):
    path = tmp_path / "missing.mat"
    _write_synthetic_cjdata(path, omit_fields=("tumorMask",))
    with pytest.raises(RawSampleError):
        read_raw_sample(path)


def test_read_raw_sample_rejects_mask_shape_mismatch(tmp_path):
    path = tmp_path / "shape_mismatch.mat"
    image = np.zeros((4, 4), dtype=np.int16)
    mask = np.zeros((3, 3), dtype=np.uint8)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("cjdata")
        group.create_dataset("label", data=np.array([[1.0]]))
        group.create_dataset("PID", data=np.array([[ord("1")]], dtype=np.uint16))
        group.create_dataset("image", data=image.T)
        group.create_dataset("tumorMask", data=mask.T)
        group.create_dataset("tumorBorder", data=np.zeros((1, 2)))

    with pytest.raises(RawSampleError):
        read_raw_sample(path)


def test_read_raw_sample_rejects_missing_cjdata_group(tmp_path):
    path = tmp_path / "no_group.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("not_cjdata", data=np.zeros((2, 2)))

    with pytest.raises(RawSampleError):
        read_raw_sample(path)


def test_read_raw_sample_rejects_non_hdf5_file(tmp_path):
    path = tmp_path / "not_hdf5.mat"
    path.write_bytes(b"this is not an hdf5 file")

    with pytest.raises(RawSampleError):
        read_raw_sample(path)


def test_read_raw_sample_rejects_empty_pid(tmp_path):
    path = tmp_path / "empty_pid.mat"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("cjdata")
        group.create_dataset("label", data=np.array([[1.0]]))
        group.create_dataset("PID", data=np.array([[0], [0]], dtype=np.uint16))
        group.create_dataset("image", data=np.zeros((2, 2), dtype=np.int16))
        group.create_dataset("tumorMask", data=np.zeros((2, 2), dtype=np.uint8))
        group.create_dataset("tumorBorder", data=np.zeros((1, 2)))

    with pytest.raises(RawSampleError):
        read_raw_sample(path)
