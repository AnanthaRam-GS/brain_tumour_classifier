import random

import numpy as np
import pytest
import torch

from btdl import config
from btdl.preprocessing.augmentation import AugmentationParams, apply_augmentation, sample_params


@pytest.fixture(scope="module")
def cfg():
    return config.load_contract("augmentation")


def test_sample_params_deterministic_for_same_key(cfg):
    a = sample_params(42, 0, "1", cfg)
    b = sample_params(42, 0, "1", cfg)
    assert a == b


def test_sample_params_differs_by_epoch(cfg):
    a = sample_params(42, 0, "1", cfg)
    b = sample_params(42, 1, "1", cfg)
    assert a != b


def test_sample_params_differs_by_seed(cfg):
    a = sample_params(42, 0, "1", cfg)
    b = sample_params(43, 0, "1", cfg)
    assert a != b


def test_sample_params_differs_by_sample_id(cfg):
    a = sample_params(42, 0, "1", cfg)
    b = sample_params(42, 0, "2", cfg)
    assert a != b


def test_sample_params_does_not_consume_global_rng(cfg):
    torch_state_before = torch.get_rng_state()
    np_state_before = np.random.get_state()
    py_state_before = random.getstate()

    sample_params(42, 0, "1", cfg)

    assert torch.equal(torch_state_before, torch.get_rng_state())
    np_state_after = np.random.get_state()
    assert np_state_before[0] == np_state_after[0]
    assert np.array_equal(np_state_before[1], np_state_after[1])
    assert random.getstate() == py_state_before


def test_apply_augmentation_does_not_consume_global_rng(cfg):
    params = sample_params(42, 0, "1", cfg)
    roi = torch.rand(1, 224, 224)

    torch_state_before = torch.get_rng_state()
    apply_augmentation(roi, params, cfg)
    assert torch.equal(torch_state_before, torch.get_rng_state())


def test_2000_draws_parameters_in_range_and_hflip_frequency(cfg):
    n = 2000
    hflip_count = 0
    for i in range(n):
        params = sample_params(42, 0, str(i), cfg)
        if params.hflip:
            hflip_count += 1
        assert -cfg["affine"]["degrees"] <= params.degrees <= cfg["affine"]["degrees"]
        max_translate_px = round(cfg["affine"]["translate"] * 224)
        assert abs(params.translate_px[0]) <= max_translate_px
        assert abs(params.translate_px[1]) <= max_translate_px
        scale_low, scale_high = cfg["affine"]["scale"]
        assert scale_low <= params.scale <= scale_high
        b_low, b_high = cfg["brightness"]["factor_range"]
        assert b_low <= params.brightness_factor <= b_high
        c_low, c_high = cfg["contrast"]["factor_range"]
        assert c_low <= params.contrast_factor <= c_high

    frequency = hflip_count / n
    assert 0.45 <= frequency <= 0.55


def test_hflip_only_matches_torch_flip(cfg):
    roi = torch.rand(1, 224, 224)
    params = AugmentationParams(
        hflip=True, degrees=0.0, translate_px=(0, 0), scale=1.0, brightness_factor=1.0, contrast_factor=1.0
    )
    out = apply_augmentation(roi, params, cfg)
    expected = torch.flip(roi, dims=[-1])
    # Going through affine's grid_sample even with angle=0/scale=1/translate=0
    # introduces negligible float32 interpolation noise relative to a pure index flip.
    assert torch.allclose(out, expected, atol=1e-3)


def test_affine_fill_regions_are_exactly_zero(cfg):
    roi = torch.ones(1, 224, 224)
    params = AugmentationParams(
        hflip=False, degrees=0.0, translate_px=(50, 0), scale=1.0, brightness_factor=1.0, contrast_factor=1.0
    )
    out = apply_augmentation(roi, params, cfg)
    # Shifted right by 50px: the leftmost columns are entirely outside the
    # source image and must be exactly the fill value (0), well clear of any
    # bilinear blend near the boundary.
    assert torch.all(out[:, :, :30] == 0.0)


def test_output_in_unit_range(cfg):
    roi = torch.rand(1, 224, 224)
    params = sample_params(42, 0, "1", cfg)
    out = apply_augmentation(roi, params, cfg)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_output_shape_and_dtype_preserved(cfg):
    roi = torch.rand(1, 224, 224)
    params = sample_params(42, 0, "1", cfg)
    out = apply_augmentation(roi, params, cfg)
    assert out.shape == roi.shape
    assert out.dtype == torch.float32
