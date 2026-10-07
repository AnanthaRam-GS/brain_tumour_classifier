import numpy as np
import pytest

from btdl.preprocessing.roi import (
    ROIError,
    compute_roi,
    crop_with_zero_pad,
    roi_geometry,
    square_crop_box,
    tight_bbox,
)

FAKE_CFG = {
    "roi": {"padding_fraction": 0.10},
    "resize": {"size": [224, 224]},
}


def _mask(shape, rows, cols):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rows, cols] = 1
    return mask


# ---- tight_bbox ---------------------------------------------------------------------


def test_tight_bbox_single_pixel():
    mask = _mask((20, 20), 5, 7)
    assert tight_bbox(mask) == (5, 6, 7, 8)


def test_tight_bbox_non_square_region():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[3:6, 10:25] = 1  # rows 3,4,5 ; cols 10..24
    assert tight_bbox(mask) == (3, 6, 10, 25)


def test_tight_bbox_empty_mask_raises():
    mask = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ROIError):
        tight_bbox(mask)


def test_tight_bbox_non_binary_raises():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2, 2] = 5
    with pytest.raises(ROIError):
        tight_bbox(mask)


def test_tight_bbox_wrong_ndim_raises():
    with pytest.raises(ROIError):
        tight_bbox(np.zeros((4, 4, 4), dtype=np.uint8))


# ---- square_crop_box ------------------------------------------------------------------


def test_square_crop_box_matches_padding_formula():
    bbox = (10, 20, 15, 25)  # height=10, width=10
    box = square_crop_box(bbox, 0.10)
    base_side = 10
    pad = int(np.ceil(base_side * 0.10))  # 1
    side = base_side + 2 * pad  # 12
    row_center = (10 + 20) / 2.0
    col_center = (15 + 25) / 2.0
    expected_row_start = int(np.floor(row_center - side / 2.0))
    expected_col_start = int(np.floor(col_center - side / 2.0))
    assert box == (
        expected_row_start,
        expected_row_start + side,
        expected_col_start,
        expected_col_start + side,
    )


def test_square_crop_box_odd_tight_side():
    bbox = (0, 7, 0, 7)  # height=width=7 (odd)
    box = square_crop_box(bbox, 0.10)
    row_start, row_end, col_start, col_end = box
    assert (row_end - row_start) == (col_end - col_start)


def test_square_crop_box_even_tight_side():
    bbox = (0, 8, 0, 8)  # height=width=8 (even)
    box = square_crop_box(bbox, 0.10)
    row_start, row_end, col_start, col_end = box
    assert (row_end - row_start) == (col_end - col_start)


def test_square_crop_box_negative_padding_raises():
    with pytest.raises(ROIError):
        square_crop_box((0, 5, 0, 5), -0.1)


def test_square_crop_box_empty_bbox_raises():
    with pytest.raises(ROIError):
        square_crop_box((5, 5, 0, 5), 0.1)


# ---- crop_with_zero_pad: edges and corners --------------------------------------------


def test_crop_fully_inside_preserves_pixels_outside_mask():
    image = np.arange(400, dtype=np.float32).reshape(20, 20)
    box = (5, 15, 5, 15)
    crop = crop_with_zero_pad(image, box)
    assert np.array_equal(crop, image[5:15, 5:15])


def test_crop_touching_top_left_corner_zero_pads_top_and_left():
    image = np.ones((20, 20), dtype=np.float32)
    box = (-3, 7, -2, 8)
    crop = crop_with_zero_pad(image, box)
    assert np.array_equal(crop[:3, :], np.zeros((3, 10), dtype=np.float32))
    assert np.array_equal(crop[:, :2], np.zeros((10, 2), dtype=np.float32))
    assert np.array_equal(crop[3:, 2:], np.ones((7, 8), dtype=np.float32))


def test_crop_touching_bottom_right_corner_zero_pads_bottom_and_right():
    image = np.ones((20, 20), dtype=np.float32)
    box = (15, 25, 16, 26)
    crop = crop_with_zero_pad(image, box)
    assert np.array_equal(crop[-5:, :], np.zeros((5, 10), dtype=np.float32))
    assert np.array_equal(crop[:, -6:], np.zeros((10, 6), dtype=np.float32))


def test_crop_touching_top_edge_only():
    image = np.ones((20, 20), dtype=np.float32)
    box = (-4, 6, 5, 15)
    crop = crop_with_zero_pad(image, box)
    assert np.array_equal(crop[:4, :], np.zeros((4, 10), dtype=np.float32))
    assert np.array_equal(crop[4:, :], np.ones((6, 10), dtype=np.float32))


def test_crop_touching_bottom_edge_only():
    image = np.ones((20, 20), dtype=np.float32)
    box = (14, 24, 5, 15)
    crop = crop_with_zero_pad(image, box)
    assert np.array_equal(crop[:6, :], np.ones((6, 10), dtype=np.float32))
    assert np.array_equal(crop[6:, :], np.zeros((4, 10), dtype=np.float32))


def test_crop_touching_left_edge_only():
    image = np.ones((20, 20), dtype=np.float32)
    box = (5, 15, -3, 7)
    crop = crop_with_zero_pad(image, box)
    assert np.array_equal(crop[:, :3], np.zeros((10, 3), dtype=np.float32))
    assert np.array_equal(crop[:, 3:], np.ones((10, 7), dtype=np.float32))


def test_crop_touching_right_edge_only():
    image = np.ones((20, 20), dtype=np.float32)
    box = (5, 15, 13, 23)
    crop = crop_with_zero_pad(image, box)
    assert np.array_equal(crop[:, :7], np.ones((10, 7), dtype=np.float32))
    assert np.array_equal(crop[:, 7:], np.zeros((10, 3), dtype=np.float32))


def test_crop_entirely_out_of_bounds_is_all_zero():
    image = np.ones((20, 20), dtype=np.float32)
    box = (-30, -20, -30, -20)
    crop = crop_with_zero_pad(image, box)
    assert np.array_equal(crop, np.zeros((10, 10), dtype=np.float32))


def test_crop_non_finite_image_raises():
    image = np.ones((10, 10), dtype=np.float32)
    image[0, 0] = np.nan
    with pytest.raises(ROIError):
        crop_with_zero_pad(image, (0, 5, 0, 5))


def test_crop_non_square_box_raises():
    with pytest.raises(ROIError):
        crop_with_zero_pad(np.ones((10, 10), dtype=np.float32), (0, 5, 0, 6))


# ---- roi_geometry / compute_roi ---------------------------------------------------------


def test_roi_geometry_upscale_factor():
    mask = _mask((100, 100), slice(40, 50), slice(40, 50))  # 10x10 tight box
    geometry = roi_geometry(mask, FAKE_CFG)
    assert geometry.upscale_factor == pytest.approx(224 / geometry.crop_side)


def test_roi_geometry_no_oob_when_fully_inside():
    mask = _mask((100, 100), slice(40, 50), slice(40, 50))
    geometry = roi_geometry(mask, FAKE_CFG)
    assert geometry.oob_top == geometry.oob_bottom == geometry.oob_left == geometry.oob_right == 0


def test_roi_geometry_oob_when_mask_touches_corner():
    mask = _mask((50, 50), slice(0, 5), slice(0, 5))  # touches top-left
    geometry = roi_geometry(mask, FAKE_CFG)
    assert geometry.oob_top > 0
    assert geometry.oob_left > 0


def test_compute_roi_apply_mask_false_preserves_pixels_outside_mask():
    image = np.full((50, 50), 0.7, dtype=np.float32)
    mask = _mask((50, 50), slice(20, 25), slice(20, 25))
    geometry, crop = compute_roi(image, mask, FAKE_CFG)
    # crop is unmasked: every in-bounds pixel equals the source image value,
    # not just the pixels under the mask.
    assert np.all(crop[crop != 0] == pytest.approx(0.7))


def test_compute_roi_rejects_shape_mismatch():
    image = np.zeros((10, 10), dtype=np.float32)
    mask = np.zeros((11, 11), dtype=np.uint8)
    mask[5, 5] = 1
    with pytest.raises(ROIError):
        compute_roi(image, mask, FAKE_CFG)


def test_compute_roi_rejects_non_finite_image():
    image = np.zeros((10, 10), dtype=np.float32)
    image[0, 0] = np.inf
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 1
    with pytest.raises(ROIError):
        compute_roi(image, mask, FAKE_CFG)


def test_compute_roi_rejects_non_binary_mask():
    image = np.zeros((10, 10), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 7
    with pytest.raises(ROIError):
        compute_roi(image, mask, FAKE_CFG)
