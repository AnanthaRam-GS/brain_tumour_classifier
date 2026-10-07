import random

import numpy as np
import torch

from btdl.training.seed import seed_everything


def test_same_seed_gives_same_torch_draws():
    seed_everything(42)
    a = torch.rand(10)
    seed_everything(42)
    b = torch.rand(10)
    assert torch.equal(a, b)


def test_same_seed_gives_same_numpy_draws():
    seed_everything(42)
    a = np.random.rand(10)
    seed_everything(42)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_same_seed_gives_same_python_draws():
    seed_everything(42)
    a = [random.random() for _ in range(10)]
    seed_everything(42)
    b = [random.random() for _ in range(10)]
    assert a == b


def test_different_seed_gives_different_draws():
    seed_everything(42)
    a = torch.rand(10)
    seed_everything(43)
    b = torch.rand(10)
    assert not torch.equal(a, b)


def test_returned_settings_dict():
    settings = seed_everything(42)
    assert settings["seed"] == 42
    assert settings["cudnn_benchmark"] is False
    assert settings["cudnn_deterministic"] is True
    assert settings["use_deterministic_algorithms"] is True
    assert settings["deterministic_algorithms_warn_only"] is True
    assert settings["cuda_available"] == torch.cuda.is_available()
    if not torch.cuda.is_available():
        assert settings["cublas_workspace_config"] is None
