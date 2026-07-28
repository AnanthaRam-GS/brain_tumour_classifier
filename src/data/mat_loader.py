"""Reliable loading and validation for Figshare brain tumour MATLAB samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat

REQUIRED_FIELDS = ("label", "PID", "image", "tumorBorder", "tumorMask")
VALID_LABELS = {1, 2, 3}


class MatSampleError(ValueError):
    """Raised when a MATLAB sample cannot be loaded or validated."""

    def __init__(self, path: Path | str, message: str):
        self.path = Path(path)
        super().__init__(f"{self.path}: {message}")


@dataclass(frozen=True)
class BrainTumourSample:
    """Validated data extracted from one raw MATLAB sample."""

    sample_id: str
    source_path: str
    label: int
    patient_id: str
    image: np.ndarray
    tumour_mask: np.ndarray
    tumour_border: np.ndarray
    image_dtype_original: str


def find_mat_sample_files(dataset_root: Path | str) -> list[Path]:
    """Recursively find numbered sample files and sort them by numeric stem."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")

    files = [
        path
        for path in root.rglob("*.mat")
        if path.name.lower() != "cvind.mat" and path.stem.isdigit()
    ]
    return sorted(files, key=lambda path: (int(path.stem), str(path)))


def _unwrap_scalar(value: Any) -> Any:
    """Remove MATLAB singleton containers without damaging non-scalars."""

    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_text(value: Any) -> str:
    """Convert nested MATLAB strings, character arrays, or scalars to text."""

    value = _unwrap_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()

    array = np.asarray(value)
    if array.dtype.kind in {"U", "S"}:
        chars = []
        for item in array.ravel(order="F"):
            if isinstance(item, bytes):
                chars.append(item.decode("utf-8", errors="replace"))
            else:
                chars.append(str(item))
        return "".join(chars).strip()
    if array.dtype.kind in {"u", "i"} and array.ndim >= 1:
        try:
            return "".join(chr(int(code)) for code in array.ravel(order="F") if int(code)).strip()
        except (ValueError, OverflowError):
            pass
    if array.size == 1:
        return str(_unwrap_scalar(array)).strip()
    return str(value).strip()


def _field_from_scipy(cjdata: Any, field: str) -> Any:
    if isinstance(cjdata, dict):
        if field not in cjdata:
            raise KeyError(field)
        return cjdata[field]
    if isinstance(cjdata, np.ndarray) and cjdata.dtype.names:
        if field not in cjdata.dtype.names:
            raise KeyError(field)
        return cjdata[field]
    if hasattr(cjdata, field):
        return getattr(cjdata, field)
    raise KeyError(field)


def _load_with_scipy(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"squeeze_me": True, "struct_as_record": False}
    try:
        payload = loadmat(path, simplify_cells=True)
    except TypeError:
        payload = loadmat(path, **kwargs)

    if "cjdata" not in payload:
        raise MatSampleError(path, "missing required MATLAB structure 'cjdata'")
    cjdata = payload["cjdata"]
    fields: dict[str, Any] = {}
    missing = []
    for field in REQUIRED_FIELDS:
        try:
            fields[field] = _field_from_scipy(cjdata, field)
        except KeyError:
            missing.append(field)
    if missing:
        raise MatSampleError(path, f"cjdata missing required field(s): {', '.join(missing)}")
    return fields


def _hdf5_value(node: h5py.Dataset | h5py.Group, h5file: h5py.File) -> Any:
    if isinstance(node, h5py.Group):
        return {name: _hdf5_value(child, h5file) for name, child in node.items()}

    value = node[()]
    if h5py.check_dtype(ref=node.dtype) is not None:
        refs = np.asarray(value)
        decoded = np.empty(refs.shape, dtype=object)
        for index, reference in np.ndenumerate(refs):
            decoded[index] = _hdf5_value(h5file[reference], h5file)
        return _unwrap_scalar(decoded)

    matlab_class = node.attrs.get("MATLAB_class", b"")
    if isinstance(matlab_class, np.ndarray):
        matlab_class = matlab_class.tobytes()
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode("ascii", errors="ignore")
    if matlab_class == "char":
        return _to_text(value)

    array = np.asarray(value)
    # MATLAB v7.3 numeric arrays are stored in reversed dimension order.
    if array.ndim > 1:
        array = array.transpose()
    return array


def _load_with_h5py(path: Path) -> dict[str, Any]:
    try:
        with h5py.File(path, "r") as h5file:
            if "cjdata" not in h5file:
                raise MatSampleError(path, "missing required MATLAB structure 'cjdata'")
            cjdata = h5file["cjdata"]
            if not isinstance(cjdata, h5py.Group):
                raise MatSampleError(path, "'cjdata' is not an HDF5 structure group")
            missing = [field for field in REQUIRED_FIELDS if field not in cjdata]
            if missing:
                raise MatSampleError(
                    path, f"cjdata missing required field(s): {', '.join(missing)}"
                )
            return {
                field: _hdf5_value(cjdata[field], h5file) for field in REQUIRED_FIELDS
            }
    except MatSampleError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise MatSampleError(path, f"unable to read HDF5 MATLAB file: {exc}") from exc


def _python_int(value: Any, path: Path, field: str) -> int:
    value = _unwrap_scalar(value)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MatSampleError(path, f"{field} is not a numeric scalar") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise MatSampleError(path, f"{field} is not an integer scalar: {number!r}")
    return int(number)


def _validate_and_build(path: Path, fields: dict[str, Any]) -> BrainTumourSample:
    original_image = np.asarray(fields["image"])
    image_dtype_original = str(original_image.dtype)
    image = original_image.astype(np.float32, copy=False)
    mask = np.asarray(fields["tumorMask"]).astype(bool, copy=False)
    border = np.asarray(fields["tumorBorder"]).squeeze()
    label = _python_int(fields["label"], path, "label")
    patient_id = _to_text(fields["PID"])

    if image.ndim != 2:
        raise MatSampleError(path, f"image must be two-dimensional, got shape {image.shape}")
    if mask.ndim != 2:
        raise MatSampleError(path, f"tumour mask must be two-dimensional, got shape {mask.shape}")
    if image.shape != mask.shape:
        raise MatSampleError(
            path, f"image and tumour mask shapes differ: {image.shape} != {mask.shape}"
        )
    if image.size == 0:
        raise MatSampleError(path, "image is empty")
    if mask.size == 0:
        raise MatSampleError(path, "tumour mask is empty")
    if label not in VALID_LABELS:
        raise MatSampleError(path, f"label must be one of {sorted(VALID_LABELS)}, got {label}")
    if not np.isfinite(image).all():
        raise MatSampleError(path, "image contains NaN or infinite values")
    if not patient_id:
        raise MatSampleError(path, "patient ID is empty")

    return BrainTumourSample(
        sample_id=str(path.stem),
        source_path=str(path),
        label=label,
        patient_id=patient_id,
        image=image,
        tumour_mask=mask,
        tumour_border=border,
        image_dtype_original=image_dtype_original,
    )


def load_mat_sample(path: Path | str) -> BrainTumourSample:
    """Load and validate one MATLAB sample, with MATLAB v7.3 fallback."""

    sample_path = Path(path)
    try:
        fields = _load_with_scipy(sample_path)
    except MatSampleError:
        raise
    except (NotImplementedError, OSError, TypeError, ValueError) as scipy_error:
        try:
            fields = _load_with_h5py(sample_path)
        except MatSampleError as hdf5_error:
            raise MatSampleError(
                sample_path,
                f"unable to load with scipy ({scipy_error}) or h5py ({hdf5_error})",
            ) from hdf5_error
    return _validate_and_build(sample_path, fields)
