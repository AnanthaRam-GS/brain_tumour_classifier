import sys


def test_core_imports_succeed():
    import h5py  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import yaml  # noqa: F401


def test_report_versions_and_device_availability(capsys):
    import torch
    import torchvision

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    print(f"torch=={torch.__version__}")
    print(f"torchvision=={torchvision.__version__}")
    print(f"cuda_available={cuda_available}")
    print(f"mps_available={mps_available}")

    assert isinstance(torch.__version__, str) and torch.__version__
    assert isinstance(torchvision.__version__, str) and torchvision.__version__
    assert isinstance(cuda_available, bool)
    assert isinstance(mps_available, bool)


def _best_available_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_tiny_matmul_and_backward_on_best_device():
    import torch

    device = _best_available_device()
    x = torch.randn(8, 4, device=device, requires_grad=True)
    w = torch.randn(4, 2, device=device, requires_grad=True)
    y = (x @ w).sum()
    y.backward()

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(w.grad).all()


def test_torchvision_imagenet_weight_enums_exist_without_downloading():
    from torchvision.models import (
        AlexNet_Weights,
        EfficientNet_B0_Weights,
        GoogLeNet_Weights,
        ResNet18_Weights,
        VGG16_Weights,
    )

    assert AlexNet_Weights.IMAGENET1K_V1 is not None
    assert VGG16_Weights.IMAGENET1K_V1 is not None
    assert GoogLeNet_Weights.IMAGENET1K_V1 is not None
    assert ResNet18_Weights.IMAGENET1K_V1 is not None
    assert EfficientNet_B0_Weights.IMAGENET1K_V1 is not None


def test_btdl_resolves_under_phase2_and_never_imports_src():
    import btdl

    package_path = str(btdl.__file__).replace("\\", "/")
    assert "phase2/src/btdl" in package_path
    assert "src" not in sys.modules
