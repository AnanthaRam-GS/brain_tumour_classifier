"""Independent h5py-only reader for the Figshare brain-tumor MATLAB v7.3 files.

Written from scratch against the raw HDF5 layout (verified directly with
h5py, not by reading Phase 1's src/data/mat_loader.py) so that Phase 2's
dataset audit (see cli/audit.py) is a genuine independent check, not the
same code path re-run.

Observed layout of cjdata/* for every sampled file (checked interactively):
    cjdata/label       shape (1, 1),  float64   -- numeric class label
    cjdata/PID         shape (N, 1),  uint16    -- ASCII char codes, column vector
    cjdata/image       shape (H, W),  int16     -- raw intensity, MATLAB column-major
    cjdata/tumorBorder shape (1, M),  float64   -- flattened (x, y) border points
    cjdata/tumorMask   shape (H, W),  uint8     -- binary mask, MATLAB column-major

MATLAB v7.3 (HDF5) stores >=2-D numeric arrays in reversed dimension order
relative to how MATLAB indexes them (column-major on-disk). Any field read
here with ndim >= 2 is transposed back to the row-major (row, col)
orientation MATLAB code (and Phase 1's output) would show.
"""

from dataclasses import dataclass

import h5py
import numpy as np

REQUIRED_FIELDS = ("label", "PID", "image", "tumorBorder", "tumorMask")


class RawSampleError(ValueError):
    """Raised when a raw .mat file cannot be read as a valid cjdata sample."""


@dataclass(frozen=True)
class RawSample:
    label: int
    patient_id: str
    image: np.ndarray
    tumor_mask: np.ndarray
    tumor_border: np.ndarray


def _transpose_if_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array.T
    return array


def _decode_pid(raw: np.ndarray) -> str:
    codes = raw.ravel()
    if not np.issubdtype(codes.dtype, np.integer):
        raise RawSampleError(f"PID must be an integer code array, got dtype {codes.dtype}")
    chars = [chr(int(code)) for code in codes if int(code) != 0]
    pid = "".join(chars).strip()
    if not pid:
        raise RawSampleError("PID decoded to an empty string")
    return pid


def _decode_label(raw: np.ndarray) -> int:
    if raw.size != 1:
        raise RawSampleError(f"label must be a scalar, got shape {raw.shape}")
    value = float(raw.reshape(-1)[0])
    if not value.is_integer():
        raise RawSampleError(f"label must be an integral value, got {value}")
    return int(value)


def read_raw_sample(path) -> RawSample:
    """Read one Figshare cjdata sample from a MATLAB v7.3 (HDF5) .mat file."""

    if not h5py.is_hdf5(str(path)):
        raise RawSampleError(
            f"{path} is not a MATLAB v7.3 / HDF5 file; "
            "no scipy fallback is used here, report this file instead."
        )

    with h5py.File(str(path), "r") as handle:
        if "cjdata" not in handle:
            raise RawSampleError(f"{path} has no top-level 'cjdata' group")
        group = handle["cjdata"]

        missing = [field for field in REQUIRED_FIELDS if field not in group]
        if missing:
            raise RawSampleError(f"{path} is missing cjdata field(s): {missing}")

        label_raw = group["label"][()]
        pid_raw = group["PID"][()]
        image_raw = group["image"][()]
        border_raw = group["tumorBorder"][()]
        mask_raw = group["tumorMask"][()]

        label = _decode_label(np.asarray(label_raw))
        patient_id = _decode_pid(np.asarray(pid_raw))

        image = _transpose_if_2d(np.asarray(image_raw))
        mask = _transpose_if_2d(np.asarray(mask_raw))
        border = _transpose_if_2d(np.asarray(border_raw)).ravel()

        if image.ndim != 2:
            raise RawSampleError(f"{path} image must be 2D, got shape {image.shape}")
        if mask.shape != image.shape:
            raise RawSampleError(
                f"{path} tumorMask shape {mask.shape} does not match image shape {image.shape}"
            )

        return RawSample(
            label=label,
            patient_id=patient_id,
            image=image,
            tumor_mask=mask.astype(bool),
            tumor_border=border,
        )
