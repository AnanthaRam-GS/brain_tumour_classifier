"""Intensity normalization for brain MRI images."""

from __future__ import annotations

from typing import Any

import numpy as np


def robust_foreground_percentile_normalize(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize positive foreground intensities to ``[0, 1]`` robustly.

    Percentiles are estimated only from pixels greater than zero. The whole
    image is clipped and scaled, after which original zero-valued background
    pixels are explicitly restored to zero.
    """

    image_float = np.asarray(image, dtype=np.float32)
    if image_float.ndim != 2:
        raise ValueError(f"image must be two-dimensional, got shape {image_float.shape}")
    if not np.isfinite(image_float).all():
        raise ValueError("image contains NaN or infinite values")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")

    foreground_mask = image_float > 0
    foreground = image_float[foreground_mask]
    foreground_count = int(foreground.size)
    metadata: dict[str, Any] = {
        "normalization_low": 0.0,
        "normalization_high": 0.0,
        "foreground_pixel_count": foreground_count,
        "empty_foreground": foreground_count == 0,
        "constant_foreground": False,
        "normalization_warning": "",
    }

    if foreground_count == 0:
        metadata["normalization_warning"] = "empty foreground: no positive-intensity pixels"
        return np.zeros_like(image_float, dtype=np.float32), metadata

    low, high = np.percentile(
        foreground, [lower_percentile, upper_percentile]
    ).astype(np.float64)
    metadata["normalization_low"] = float(low)
    metadata["normalization_high"] = float(high)

    scale = float(high - low)
    tolerance = float(np.finfo(np.float32).eps * max(1.0, abs(float(low)), abs(float(high))))
    if not np.isfinite(scale) or scale <= tolerance:
        metadata["constant_foreground"] = True
        metadata["normalization_warning"] = (
            "constant foreground: percentile range is zero or numerically negligible"
        )
        return np.zeros_like(image_float, dtype=np.float32), metadata

    clipped = np.clip(image_float, low, high)
    normalized = ((clipped - low) / scale).astype(np.float32, copy=False)
    normalized[~foreground_mask] = 0.0
    np.clip(normalized, 0.0, 1.0, out=normalized)
    if not np.isfinite(normalized).all():
        metadata["normalization_warning"] = "non-finite normalized values replaced with zero"
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(normalized, 0.0, 1.0, out=normalized)
        normalized[~foreground_mask] = 0.0
    return normalized.astype(np.float32, copy=False), metadata
