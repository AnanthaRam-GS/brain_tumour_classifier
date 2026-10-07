"""Phase 2 ROI geometry, ported to the exact Phase 1 specification.

Written fresh (numpy-only, pure functions) against the geometry documented
in src/preprocessing/roi.py's find_mask_bbox, make_square_bbox, and
crop_and_pad (as of the commit this module was added in), and verified
byte-for-byte against Phase 1's native (unmasked) roi_image for all 3,064
real samples in tests/test_phase1_parity.py. That test is the parity
contract this module must keep satisfying; this module is not a copy of
Phase 1's code.

Geometry, matching Phase 1 exactly (docs/DECISIONS.md D4):
  - tight bbox: half-open (row_start, row_end, col_start, col_end) from
    mask.nonzero(), i.e. [rows.min(), rows.max()+1), [cols.min(), cols.max()+1).
  - square padding: pad_pixels = ceil(base_side * padding_fraction), where
    base_side = max(tight_height, tight_width); final side =
    base_side + 2 * pad_pixels (so pad_pixels is added on both sides).
  - centering: the square is centered on the tight bbox's (float) center;
    the square's start coordinate is floor(center - side / 2.0).
  - out-of-bounds: the square may extend beyond the image; crop_with_zero_pad
    copies the in-bounds region unmodified and zero-fills the rest.
  - the UNMASKED image is cropped (Phase 1's native roi_image, not
    roi_image_masked) -- the mask only localizes the crop.
"""

from dataclasses import dataclass

import numpy as np


class ROIError(ValueError):
    """Raised when ROI geometry input or output is invalid."""


@dataclass(frozen=True)
class ROIGeometry:
    tight_bbox: tuple
    crop_box: tuple
    crop_side: int
    oob_top: int
    oob_bottom: int
    oob_left: int
    oob_right: int
    upscale_factor: float


def tight_bbox(mask) -> tuple:
    """Half-open (row_start, row_end, col_start, col_end) tight bbox of a binary mask."""

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ROIError(f"mask must be 2D, got shape {mask_array.shape}")
    mask_values = set(np.unique(mask_array).tolist())
    if not mask_values.issubset({0, 1, True, False}):
        raise ROIError(f"mask is not binary: {sorted(mask_values)}")

    rows, cols = np.nonzero(mask_array)
    if rows.size == 0:
        raise ROIError("mask is empty")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def square_crop_box(bbox, padding_fraction: float) -> tuple:
    """Square crop box (may extend beyond the image) centered on a tight bbox, with padding."""

    if padding_fraction < 0:
        raise ROIError("padding_fraction must be non-negative")
    row_start, row_end, col_start, col_end = bbox
    height = row_end - row_start
    width = col_end - col_start
    if height <= 0 or width <= 0:
        raise ROIError(f"bbox must have positive area, got {bbox}")

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


def crop_with_zero_pad(image, box) -> np.ndarray:
    """Crop `box` from `image`, zero-filling any part of `box` outside the image."""

    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2:
        raise ROIError(f"image must be 2D, got shape {source.shape}")
    if not np.isfinite(source).all():
        raise ROIError("image contains NaN or infinite values")

    row_start, row_end, col_start, col_end = box
    side_h = row_end - row_start
    side_w = col_end - col_start
    if side_h <= 0 or side_w <= 0 or side_h != side_w:
        raise ROIError(f"box must describe a positive square, got {box}")

    output = np.zeros((side_h, side_w), dtype=np.float32)
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


def roi_geometry(mask, cfg) -> ROIGeometry:
    """Full ROI geometry for `mask` under the input contract `cfg` (a loaded config.load_contract("input"))."""

    mask_array = np.asarray(mask)
    bbox = tight_bbox(mask_array)
    padding_fraction = cfg["roi"]["padding_fraction"]
    box = square_crop_box(bbox, padding_fraction)
    row_start, row_end, col_start, col_end = box
    side = row_end - row_start

    oob_top = max(0, -row_start)
    oob_left = max(0, -col_start)
    oob_bottom = max(0, row_end - mask_array.shape[0])
    oob_right = max(0, col_end - mask_array.shape[1])

    target_size = cfg["resize"]["size"][0]
    upscale_factor = target_size / side

    return ROIGeometry(
        tight_bbox=bbox,
        crop_box=box,
        crop_side=side,
        oob_top=oob_top,
        oob_bottom=oob_bottom,
        oob_left=oob_left,
        oob_right=oob_right,
        upscale_factor=upscale_factor,
    )


def compute_roi(image, mask, cfg):
    """Validate image/mask together, then return (ROIGeometry, unmasked float32 crop)."""

    image_array = np.asarray(image)
    mask_array = np.asarray(mask)
    if image_array.ndim != 2:
        raise ROIError(f"image must be 2D, got shape {image_array.shape}")
    if mask_array.ndim != 2:
        raise ROIError(f"mask must be 2D, got shape {mask_array.shape}")
    if image_array.shape != mask_array.shape:
        raise ROIError(f"image and mask shapes differ: {image_array.shape} != {mask_array.shape}")
    if not np.isfinite(image_array.astype(np.float64, copy=False)).all():
        raise ROIError("image contains NaN or infinite values")

    geometry = roi_geometry(mask_array, cfg)
    crop = crop_with_zero_pad(image_array, geometry.crop_box)
    return geometry, crop
