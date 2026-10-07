"""Train-only augmentation (docs/DECISIONS.md D7).

Randomness is keyed by (seed, epoch, sample_id) via a stable sha256-based
hash seeding a per-call torch.Generator -- never Python's hash() (not
stable across processes/runs) and never torch's/numpy's/python's global
RNG (so augmentation is independent of num_workers, batch order, and
device; see the D16 implementation note in DECISIONS.md). Operates on the
single-channel [0,1] float32 ROI, BEFORE to_model_input.
"""

import hashlib
from dataclasses import dataclass

import torch
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as TF


@dataclass(frozen=True)
class AugmentationParams:
    hflip: bool
    degrees: float
    translate_px: tuple
    scale: float
    brightness_factor: float
    contrast_factor: float


def _seed_from(seed: int, epoch: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sample_params(seed: int, epoch: int, sample_id: str, cfg, image_size: int = 224) -> AugmentationParams:
    """Draw all random augmentation parameters for one (seed, epoch, sample_id).

    Uses a local torch.Generator seeded from a stable hash -- never consumes
    or alters the global torch/numpy/python RNG state.
    """

    generator = torch.Generator()
    generator.manual_seed(_seed_from(seed, epoch, sample_id))

    hflip = bool(torch.rand(1, generator=generator).item() < cfg["hflip"]["p"])

    affine_cfg = cfg["affine"]
    degrees_max = float(affine_cfg["degrees"])
    degrees = float(
        torch.empty(1).uniform_(-degrees_max, degrees_max, generator=generator).item()
    )

    translate_frac = float(affine_cfg["translate"])
    tx_frac = float(torch.empty(1).uniform_(-translate_frac, translate_frac, generator=generator).item())
    ty_frac = float(torch.empty(1).uniform_(-translate_frac, translate_frac, generator=generator).item())
    translate_px = (round(tx_frac * image_size), round(ty_frac * image_size))

    scale_low, scale_high = affine_cfg["scale"]
    scale = float(torch.empty(1).uniform_(float(scale_low), float(scale_high), generator=generator).item())

    brightness_low, brightness_high = cfg["brightness"]["factor_range"]
    brightness_factor = float(
        torch.empty(1).uniform_(float(brightness_low), float(brightness_high), generator=generator).item()
    )

    contrast_low, contrast_high = cfg["contrast"]["factor_range"]
    contrast_factor = float(
        torch.empty(1).uniform_(float(contrast_low), float(contrast_high), generator=generator).item()
    )

    return AugmentationParams(
        hflip=hflip,
        degrees=degrees,
        translate_px=translate_px,
        scale=scale,
        brightness_factor=brightness_factor,
        contrast_factor=contrast_factor,
    )


def apply_augmentation(roi: torch.Tensor, params: AugmentationParams, cfg) -> torch.Tensor:
    """Apply `params` to a [1, H, W] float32 ROI, in the contract's fixed order."""

    x = roi
    if params.hflip:
        x = TF.horizontal_flip(x)

    affine_cfg = cfg["affine"]
    x = TF.affine(
        x,
        angle=params.degrees,
        translate=list(params.translate_px),
        scale=params.scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=float(affine_cfg["fill"]),
    )

    x = TF.adjust_brightness(x, params.brightness_factor)
    x = TF.adjust_contrast(x, params.contrast_factor)

    low, high = cfg["clamp"]
    x = torch.clamp(x, float(low), float(high))
    return x
