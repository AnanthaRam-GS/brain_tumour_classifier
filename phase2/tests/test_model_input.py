import numpy as np
import pytest
import torch

from btdl import config
from btdl.preprocessing.model_input import to_model_input


@pytest.fixture(scope="module")
def cfg():
    return config.load_contract("input")


def test_output_shape(cfg):
    roi = torch.rand(224, 224)
    out = to_model_input(roi, cfg)
    assert tuple(out.shape) == (3, 224, 224)


def test_three_channels_identical_before_normalization(cfg):
    roi = torch.rand(224, 224)
    mean = torch.tensor(cfg["model_input"]["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["model_input"]["std"]).view(3, 1, 1)
    out = to_model_input(roi, cfg)
    # Undo normalization; all three channels should be identical to the source ROI.
    recovered = out * std + mean
    assert torch.allclose(recovered[0], recovered[1], atol=1e-5)
    assert torch.allclose(recovered[1], recovered[2], atol=1e-5)
    assert torch.allclose(recovered[0], roi, atol=1e-5)


def test_zero_input_maps_to_minus_mean_over_std(cfg):
    roi = torch.zeros(224, 224)
    out = to_model_input(roi, cfg)
    mean = cfg["model_input"]["mean"]
    std = cfg["model_input"]["std"]
    for channel in range(3):
        expected = -mean[channel] / std[channel]
        assert torch.allclose(out[channel], torch.full((224, 224), expected), atol=1e-6)


def test_one_input_maps_to_one_minus_mean_over_std(cfg):
    roi = torch.ones(224, 224)
    out = to_model_input(roi, cfg)
    mean = cfg["model_input"]["mean"]
    std = cfg["model_input"]["std"]
    for channel in range(3):
        expected = (1.0 - mean[channel]) / std[channel]
        assert torch.allclose(out[channel], torch.full((224, 224), expected), atol=1e-6)


def test_accepts_hw_and_1hw(cfg):
    roi_hw = torch.rand(224, 224)
    roi_1hw = roi_hw.unsqueeze(0)
    out_hw = to_model_input(roi_hw, cfg)
    out_1hw = to_model_input(roi_1hw, cfg)
    assert torch.equal(out_hw, out_1hw)


@pytest.mark.parametrize(
    "bad_shape",
    [(3, 224, 224), (2, 224, 224), (224,), (1, 1, 224, 224)],
)
def test_rejects_other_shapes(cfg, bad_shape):
    roi = torch.rand(*bad_shape)
    with pytest.raises(ValueError):
        to_model_input(roi, cfg)


def test_accepts_numpy_array(cfg):
    roi = np.random.RandomState(0).rand(224, 224).astype(np.float32)
    out = to_model_input(roi, cfg)
    assert tuple(out.shape) == (3, 224, 224)
