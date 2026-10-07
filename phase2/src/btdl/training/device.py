"""Device selection: CUDA -> MPS -> CPU, overridable via BTDL_DEVICE."""

import os
import platform

import torch

_VALID_PREFERENCES = ("auto", "cuda", "mps", "cpu")


class DeviceError(ValueError):
    """Raised when an explicitly requested device is unavailable."""


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _mps_available() -> bool:
    return torch.backends.mps.is_available()


def select_device(preference: str = "auto") -> torch.device:
    """CUDA -> MPS -> CPU, in order, unless BTDL_DEVICE or `preference` names one explicitly.

    BTDL_DEVICE (if set) takes priority over the `preference` argument.
    An explicitly requested but unavailable device (cuda/mps) raises
    DeviceError rather than silently falling back.
    """

    requested = os.environ.get("BTDL_DEVICE", preference)
    if requested not in _VALID_PREFERENCES:
        raise ValueError(f"device preference must be one of {_VALID_PREFERENCES}, got {requested!r}")

    if requested == "cuda":
        if not _cuda_available():
            raise DeviceError("BTDL_DEVICE/preference requested 'cuda' but CUDA is not available")
        return torch.device("cuda")

    if requested == "mps":
        if not _mps_available():
            raise DeviceError("BTDL_DEVICE/preference requested 'mps' but MPS is not available")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    # auto: CUDA -> MPS -> CPU
    if _cuda_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> dict:
    """A JSON-serializable description of `device`, for run metadata."""

    description = {
        "type": device.type,
        "name": None,
        "torch_version": torch.__version__,
        "cuda_available": _cuda_available(),
        "mps_available": _mps_available(),
        "platform": platform.platform(),
    }
    if device.type == "cuda" and _cuda_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        description["name"] = torch.cuda.get_device_name(index)
    elif device.type == "mps":
        description["name"] = "Apple MPS"
    elif device.type == "cpu":
        description["name"] = platform.processor() or "cpu"
    return description
