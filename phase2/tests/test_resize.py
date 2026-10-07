import numpy as np
import pytest
import torch
import torch.nn.functional as F

from btdl import config
from btdl.preprocessing.resize import resize_roi


@pytest.fixture(scope="module")
def cfg():
    return config.load_contract("input")


def test_resize_output_shape_and_dtype(cfg):
    roi = np.random.RandomState(0).rand(60, 60).astype(np.float32)
    out = resize_roi(roi, cfg)
    assert out.shape == (224, 224)
    assert out.dtype == np.float32


def test_resize_output_range(cfg):
    roi = (np.random.RandomState(1).rand(60, 60) * 2 - 0.5).astype(np.float32)
    out = resize_roi(roi, cfg)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_resize_constant_image_stays_constant(cfg):
    roi = np.full((40, 40), 0.42, dtype=np.float32)
    out = resize_roi(roi, cfg)
    assert np.allclose(out, 0.42, atol=1e-5)


def test_resize_is_deterministic(cfg):
    roi = np.random.RandomState(2).rand(77, 77).astype(np.float32)
    first = resize_roi(roi, cfg)
    second = resize_roi(roi, cfg)
    assert np.array_equal(first, second)


def test_resize_matches_direct_interpolate_call(cfg):
    roi = np.random.RandomState(3).rand(53, 53).astype(np.float32)
    out = resize_roi(roi, cfg)

    tensor = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0)
    expected = F.interpolate(
        tensor,
        size=tuple(cfg["resize"]["size"]),
        mode=cfg["resize"]["mode"],
        align_corners=cfg["resize"]["align_corners"],
        antialias=cfg["resize"]["antialias"],
    )
    low, high = cfg["resize"]["clamp"]
    expected = torch.clamp(expected, float(low), float(high)).squeeze(0).squeeze(0).numpy()
    assert np.array_equal(out, expected.astype(np.float32))


def test_resize_rejects_non_2d_input(cfg):
    with pytest.raises(ValueError):
        resize_roi(np.zeros((3, 10, 10), dtype=np.float32), cfg)
