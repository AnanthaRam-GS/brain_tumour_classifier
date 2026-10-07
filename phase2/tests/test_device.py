import pytest
import torch

from btdl.training import device as device_module
from btdl.training.device import DeviceError, describe_device, select_device


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("BTDL_DEVICE", raising=False)


def test_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(device_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(device_module, "_mps_available", lambda: True)
    assert select_device("auto").type == "cuda"


def test_auto_prefers_mps_over_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(device_module, "_cuda_available", lambda: False)
    monkeypatch.setattr(device_module, "_mps_available", lambda: True)
    assert select_device("auto").type == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(device_module, "_cuda_available", lambda: False)
    monkeypatch.setattr(device_module, "_mps_available", lambda: False)
    assert select_device("auto").type == "cpu"


def test_explicit_cpu_always_allowed(monkeypatch):
    monkeypatch.setattr(device_module, "_cuda_available", lambda: False)
    monkeypatch.setattr(device_module, "_mps_available", lambda: False)
    assert select_device("cpu").type == "cpu"


def test_explicit_unavailable_cuda_raises(monkeypatch):
    monkeypatch.setattr(device_module, "_cuda_available", lambda: False)
    with pytest.raises(DeviceError):
        select_device("cuda")


def test_explicit_unavailable_mps_raises(monkeypatch):
    monkeypatch.setattr(device_module, "_mps_available", lambda: False)
    with pytest.raises(DeviceError):
        select_device("mps")


def test_explicit_available_cuda_succeeds(monkeypatch):
    monkeypatch.setattr(device_module, "_cuda_available", lambda: True)
    assert select_device("cuda").type == "cuda"


def test_env_var_overrides_preference_argument(monkeypatch):
    monkeypatch.setattr(device_module, "_cuda_available", lambda: False)
    monkeypatch.setattr(device_module, "_mps_available", lambda: False)
    monkeypatch.setenv("BTDL_DEVICE", "cpu")
    assert select_device("auto").type == "cpu"


def test_env_var_unavailable_explicit_device_raises(monkeypatch):
    monkeypatch.setattr(device_module, "_mps_available", lambda: False)
    monkeypatch.setenv("BTDL_DEVICE", "mps")
    with pytest.raises(DeviceError):
        select_device("auto")


def test_invalid_preference_raises():
    with pytest.raises(ValueError):
        select_device("tpu")


def test_describe_device_cpu():
    description = describe_device(torch.device("cpu"))
    assert description["type"] == "cpu"
    assert description["torch_version"] == torch.__version__
    assert isinstance(description["cuda_available"], bool)
    assert isinstance(description["mps_available"], bool)
