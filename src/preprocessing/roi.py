"""Shared in-memory tumor ROI preprocessing for Phase 1 handcrafted features.

The square ROI metadata uses Python half-open coordinates in the original
image space: ``(row_start, row_end, col_start, col_end)``. The square box may
extend outside the source image, in which case the missing area is zero-padded
after cropping the valid image region.

Native and standardized ROI views serve different later feature families:
native masked logic for texture/intensity/geometry, and fixed 128x128 masked
views for algorithms that require a stable spatial input size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.transform import resize

from src.data.feature_dataset import Phase1Sample


class ROIError(ValueError):
    """Raised when tumor ROI preprocessing input or output is invalid."""


@dataclass(frozen=True)
class TumorROI:
    """All agreed Phase 1 ROI representations for one processed sample."""

    sample_id: str
    tight_bbox: tuple[int, int, int, int]
    square_bbox: tuple[int, int, int, int]
    padding_fraction: float
    roi_image: np.ndarray
    roi_mask: np.ndarray
    roi_image_masked: np.ndarray
    roi_image_resized: np.ndarray
    roi_mask_resized: np.ndarray
    roi_image_masked_resized: np.ndarray
    original_shape: tuple[int, int]
    roi_shape: tuple[int, int]
    standard_size: tuple[int, int]
    required_padding: bool


def _validate_inputs(
    image: np.ndarray,
    mask: np.ndarray,
    padding_fraction: float,
    standard_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image_array = np.asarray(image, dtype=np.float32)
    mask_array = np.asarray(mask)
    if image_array.ndim != 2:
        raise ROIError(f"image must be 2D, got shape {image_array.shape}")
    if mask_array.ndim != 2:
        raise ROIError(f"mask must be 2D, got shape {mask_array.shape}")
    if image_array.shape != mask_array.shape:
        raise ROIError(f"image and mask shapes differ: {image_array.shape} != {mask_array.shape}")
    if not np.isfinite(image_array).all():
        raise ROIError("image contains NaN or infinite values")
    tolerance = float(np.finfo(np.float32).eps * 16)
    if float(image_array.min()) < -tolerance or float(image_array.max()) > 1.0 + tolerance:
        raise ROIError("image values must be within [0, 1]")
    mask_values = set(np.unique(mask_array).tolist())
    if not mask_values.issubset({0, 1, False, True}):
        raise ROIError(f"mask is not binary: {sorted(mask_values)}")
    mask_binary = mask_array.astype(np.uint8, copy=False)
    if int(mask_binary.sum()) == 0:
        raise ROIError("mask is empty")
    if padding_fraction < 0:
        raise ROIError("padding_fraction must be non-negative")
    if len(standard_size) != 2 or any(int(size) <= 0 for size in standard_size):
        raise ROIError("standard_size must contain two positive integers")
    return image_array, mask_binary


def find_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the tight half-open tumor bbox as row_start, row_end, col_start, col_end."""

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ROIError(f"mask must be 2D, got shape {mask_array.shape}")
    rows, cols = np.nonzero(mask_array)
    if rows.size == 0:
        raise ROIError("mask is empty")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def make_square_bbox(
    tight_bbox: tuple[int, int, int, int],
    *,
    padding_fraction: float = 0.10,
) -> tuple[int, int, int, int]:
    """Expand a tight bbox to a padded square around the tight bbox center.

    Rounding rule: padding is ``ceil(base_side * padding_fraction)`` pixels on
    each side, and the final integer side length is ``base_side + 2 * padding``.
    """

    if padding_fraction < 0:
        raise ROIError("padding_fraction must be non-negative")
    row_start, row_end, col_start, col_end = tight_bbox
    height = row_end - row_start
    width = col_end - col_start
    if height <= 0 or width <= 0:
        raise ROIError(f"tight_bbox must have positive area, got {tight_bbox}")
    base_side = max(height, width)
    pad_pixels = int(np.ceil(base_side * padding_fraction))
    side = base_side + 2 * pad_pixels
    row_center = (row_start + row_end) / 2.0
    col_center = (col_start + col_end) / 2.0
    square_row_start = int(np.floor(row_center - side / 2.0))
    square_col_start = int(np.floor(col_center - side / 2.0))
    return (
        square_row_start,
        square_row_start + side,
        square_col_start,
        square_col_start + side,
    )


def crop_and_pad(
    array: np.ndarray,
    square_bbox: tuple[int, int, int, int],
    *,
    pad_value: float | int,
) -> np.ndarray:
    """Crop a desired square bbox and zero-pad areas outside image boundaries."""

    source = np.asarray(array)
    if source.ndim != 2:
        raise ROIError(f"array must be 2D, got shape {source.shape}")
    row_start, row_end, col_start, col_end = square_bbox
    side_h = row_end - row_start
    side_w = col_end - col_start
    if side_h <= 0 or side_w <= 0 or side_h != side_w:
        raise ROIError(f"square_bbox must describe a positive square, got {square_bbox}")

    output = np.full((side_h, side_w), pad_value, dtype=source.dtype)
    src_row_start = max(row_start, 0)
    src_row_end = min(row_end, source.shape[0])
    src_col_start = max(col_start, 0)
    src_col_end = min(col_end, source.shape[1])
    if src_row_start >= src_row_end or src_col_start >= src_col_end:
        return output

    dst_row_start = src_row_start - row_start
    dst_col_start = src_col_start - col_start
    output[
        dst_row_start : dst_row_start + (src_row_end - src_row_start),
        dst_col_start : dst_col_start + (src_col_end - src_col_start),
    ] = source[src_row_start:src_row_end, src_col_start:src_col_end]
    return output


def resize_roi(
    roi_image: np.ndarray,
    roi_mask: np.ndarray,
    *,
    standard_size: tuple[int, int] = (128, 128),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize ROI image with bilinear interpolation and mask with nearest neighbor."""

    if len(standard_size) != 2 or any(int(size) <= 0 for size in standard_size):
        raise ROIError("standard_size must contain two positive integers")
    output_shape = tuple(int(size) for size in standard_size)
    image_resized = resize(
        roi_image,
        output_shape,
        order=1,
        mode="constant",
        cval=0.0,
        clip=True,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)
    mask_resized_float = resize(
        roi_mask.astype(np.uint8, copy=False),
        output_shape,
        order=0,
        mode="constant",
        cval=0,
        clip=True,
        preserve_range=True,
        anti_aliasing=False,
    )
    mask_resized = (mask_resized_float >= 0.5).astype(np.uint8)
    image_resized = np.clip(image_resized, 0.0, 1.0).astype(np.float32)
    return image_resized, mask_resized, image_resized * mask_resized


def prepare_tumor_roi(
    sample: Phase1Sample,
    *,
    padding_fraction: float = 0.10,
    standard_size: tuple[int, int] = (128, 128),
) -> TumorROI:
    """Prepare deterministic native and standardized tumor ROI views in memory."""

    image, mask = _validate_inputs(
        sample.image_normalized,
        sample.tumor_mask,
        padding_fraction,
        standard_size,
    )
    normalized_standard_size = tuple(int(size) for size in standard_size)
    tight_bbox = find_mask_bbox(mask)
    square_bbox = make_square_bbox(tight_bbox, padding_fraction=padding_fraction)
    roi_image = crop_and_pad(image, square_bbox, pad_value=0.0).astype(np.float32)
    roi_mask = crop_and_pad(mask, square_bbox, pad_value=0).astype(np.uint8)
    roi_mask = (roi_mask > 0).astype(np.uint8)
    roi_image_masked = roi_image * roi_mask
    roi_image_resized, roi_mask_resized, roi_image_masked_resized = resize_roi(
        roi_image,
        roi_mask,
        standard_size=normalized_standard_size,
    )
    required_padding = (
        square_bbox[0] < 0
        or square_bbox[2] < 0
        or square_bbox[1] > image.shape[0]
        or square_bbox[3] > image.shape[1]
    )
    _validate_outputs(
        roi_image,
        roi_mask,
        roi_image_masked,
        roi_image_resized,
        roi_mask_resized,
        roi_image_masked_resized,
        standard_size=normalized_standard_size,
    )
    return TumorROI(
        sample_id=sample.sample_id,
        tight_bbox=tight_bbox,
        square_bbox=square_bbox,
        padding_fraction=padding_fraction,
        roi_image=roi_image,
        roi_mask=roi_mask,
        roi_image_masked=roi_image_masked,
        roi_image_resized=roi_image_resized,
        roi_mask_resized=roi_mask_resized,
        roi_image_masked_resized=roi_image_masked_resized,
        original_shape=tuple(int(size) for size in image.shape),
        roi_shape=tuple(int(size) for size in roi_image.shape),
        standard_size=normalized_standard_size,
        required_padding=required_padding,
    )


def _validate_outputs(
    roi_image: np.ndarray,
    roi_mask: np.ndarray,
    roi_image_masked: np.ndarray,
    roi_image_resized: np.ndarray,
    roi_mask_resized: np.ndarray,
    roi_image_masked_resized: np.ndarray,
    *,
    standard_size: tuple[int, int],
) -> None:
    for name, array in {
        "roi_image": roi_image,
        "roi_image_masked": roi_image_masked,
        "roi_image_resized": roi_image_resized,
        "roi_image_masked_resized": roi_image_masked_resized,
    }.items():
        if not np.isfinite(array).all():
            raise ROIError(f"{name} contains NaN or infinite values")
        tolerance = float(np.finfo(np.float32).eps * 16)
        if float(array.min()) < -tolerance or float(array.max()) > 1.0 + tolerance:
            raise ROIError(f"{name} values are outside [0, 1]")
    for name, array in {"roi_mask": roi_mask, "roi_mask_resized": roi_mask_resized}.items():
        if not set(np.unique(array).tolist()).issubset({0, 1}):
            raise ROIError(f"{name} is not binary")
    if roi_image.shape[0] != roi_image.shape[1]:
        raise ROIError("native ROI image must be square")
    if roi_image.shape != roi_mask.shape or roi_image.shape != roi_image_masked.shape:
        raise ROIError("native ROI image, mask, and masked image shapes must match")
    if roi_image_resized.shape != standard_size:
        raise ROIError("resized ROI image has the wrong shape")
    if roi_mask_resized.shape != standard_size:
        raise ROIError("resized ROI mask has the wrong shape")
    if not np.array_equal(roi_image_masked[roi_mask == 0], np.zeros_like(roi_image_masked[roi_mask == 0])):
        raise ROIError("native masked ROI is nonzero outside mask")
    if not np.array_equal(
        roi_image_masked_resized[roi_mask_resized == 0],
        np.zeros_like(roi_image_masked_resized[roi_mask_resized == 0]),
    ):
        raise ROIError("resized masked ROI is nonzero outside resized mask")
