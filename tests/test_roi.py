import numpy as np
import pytest

from src.data.feature_dataset import Phase1Sample
from src.preprocessing.roi import (
    ROIError,
    crop_and_pad,
    find_mask_bbox,
    make_square_bbox,
    prepare_tumor_roi,
)


def _sample(image: np.ndarray, mask: np.ndarray, sample_id: str = "1") -> Phase1Sample:
    return Phase1Sample(
        sample_id=sample_id,
        patient_id="P1",
        label=1,
        split="train",
        image_normalized=image,
        tumor_mask=mask,
        image_raw=image.copy(),
        tumor_border=np.array([0.0, 0.0]),
    )


def _image(shape: tuple[int, int]) -> np.ndarray:
    return np.linspace(0, 1, num=shape[0] * shape[1], dtype=np.float32).reshape(shape)


def _mask(shape: tuple[int, int], rows: slice, cols: slice) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rows, cols] = 1
    return mask


def test_tight_bounding_box_is_correct() -> None:
    mask = _mask((10, 12), slice(2, 5), slice(7, 10))

    assert find_mask_bbox(mask) == (2, 5, 7, 10)


def test_rectangular_tumor_bbox_becomes_square() -> None:
    bbox = make_square_bbox((2, 6, 4, 14), padding_fraction=0.0)

    assert bbox[1] - bbox[0] == bbox[3] - bbox[2] == 10


def test_ten_percent_padding_is_applied_deterministically() -> None:
    bbox = make_square_bbox((10, 20, 30, 40), padding_fraction=0.10)

    assert bbox == (9, 21, 29, 41)


def test_native_output_roi_is_square_and_contains_tumor() -> None:
    mask = _mask((50, 60), slice(20, 25), slice(30, 37))
    roi = prepare_tumor_roi(_sample(_image((50, 60)), mask), padding_fraction=0.10)

    assert roi.roi_shape[0] == roi.roi_shape[1]
    assert roi.roi_mask.sum() == mask.sum()


def test_centered_tumor_requires_no_zero_padding() -> None:
    image = np.ones((40, 40), dtype=np.float32)
    mask = _mask((40, 40), slice(15, 25), slice(15, 25))
    roi = prepare_tumor_roi(_sample(image, mask), padding_fraction=0.10)

    assert roi.required_padding is False
    assert np.all(roi.roi_image > 0)


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        (slice(0, 4), slice(8, 12)),
        (slice(16, 20), slice(8, 12)),
        (slice(8, 12), slice(0, 4)),
        (slice(8, 12), slice(16, 20)),
        (slice(0, 4), slice(0, 4)),
    ],
)
def test_boundary_and_corner_tumors_are_padded(rows: slice, cols: slice) -> None:
    image = np.ones((20, 20), dtype=np.float32)
    mask = _mask((20, 20), rows, cols)
    roi = prepare_tumor_roi(_sample(image, mask), padding_fraction=0.25)

    assert roi.required_padding is True
    assert roi.roi_shape == (6, 6)
    assert roi.roi_mask.sum() == mask.sum()
    assert 0.0 in np.unique(roi.roi_image)


def test_zero_padding_preserves_requested_square_shape() -> None:
    array = np.ones((5, 5), dtype=np.float32)

    cropped = crop_and_pad(array, (-2, 4, -1, 5), pad_value=0.0)

    assert cropped.shape == (6, 6)
    assert np.count_nonzero(cropped == 0.0) > 0


def test_image_and_mask_remain_aligned_after_crop_padding() -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    image[0:3, 0:3] = 0.75
    mask = _mask((8, 8), slice(0, 3), slice(0, 3))
    roi = prepare_tumor_roi(_sample(image, mask), padding_fraction=0.34)

    assert np.all(roi.roi_image[roi.roi_mask == 1] == 0.75)


def test_native_mask_is_binary_and_masked_image_zero_outside_mask() -> None:
    mask = _mask((30, 30), slice(10, 15), slice(8, 17))
    roi = prepare_tumor_roi(_sample(_image((30, 30)), mask))

    assert set(np.unique(roi.roi_mask).tolist()).issubset({0, 1})
    assert np.all(roi.roi_image_masked[roi.roi_mask == 0] == 0)


def test_standardized_outputs_are_128_square_binary_and_masked() -> None:
    mask = _mask((30, 30), slice(10, 15), slice(8, 17))
    roi = prepare_tumor_roi(_sample(_image((30, 30)), mask))

    assert roi.roi_image_resized.shape == (128, 128)
    assert roi.roi_mask_resized.shape == (128, 128)
    assert roi.roi_image_masked_resized.shape == (128, 128)
    assert set(np.unique(roi.roi_mask_resized).tolist()).issubset({0, 1})
    assert np.all(roi.roi_image_masked_resized[roi.roi_mask_resized == 0] == 0)


def test_values_remain_finite_and_in_unit_range() -> None:
    mask = _mask((30, 30), slice(10, 15), slice(8, 17))
    roi = prepare_tumor_roi(_sample(_image((30, 30)), mask))

    for array in (
        roi.roi_image,
        roi.roi_image_masked,
        roi.roi_image_resized,
        roi.roi_image_masked_resized,
    ):
        assert np.isfinite(array).all()
        assert 0.0 <= float(array.min()) <= float(array.max()) <= 1.0


def test_repeated_calls_are_deterministic() -> None:
    sample = _sample(_image((30, 30)), _mask((30, 30), slice(10, 15), slice(8, 17)))

    first = prepare_tumor_roi(sample)
    second = prepare_tumor_roi(sample)

    assert first.tight_bbox == second.tight_bbox
    assert first.square_bbox == second.square_bbox
    np.testing.assert_array_equal(first.roi_image, second.roi_image)
    np.testing.assert_array_equal(first.roi_mask, second.roi_mask)
    np.testing.assert_array_equal(first.roi_image_resized, second.roi_image_resized)


@pytest.mark.parametrize("shape", [(256, 256), (512, 512)])
def test_common_source_image_sizes_are_supported(shape: tuple[int, int]) -> None:
    mask = _mask(shape, slice(20, 40), slice(25, 55))
    roi = prepare_tumor_roi(_sample(_image(shape), mask))

    assert roi.original_shape == shape
    assert roi.roi_mask.sum() == mask.sum()
    assert roi.standard_size == (128, 128)


def test_invalid_negative_padding_fraction_is_rejected() -> None:
    sample = _sample(_image((10, 10)), _mask((10, 10), slice(2, 4), slice(2, 4)))

    with pytest.raises(ROIError, match="padding_fraction"):
        prepare_tumor_roi(sample, padding_fraction=-0.1)


@pytest.mark.parametrize("standard_size", [(0, 128), (128, 0), (128,), (128, -1)])
def test_zero_or_invalid_standard_size_is_rejected(standard_size: tuple[int, ...]) -> None:
    sample = _sample(_image((10, 10)), _mask((10, 10), slice(2, 4), slice(2, 4)))

    with pytest.raises(ROIError, match="standard_size"):
        prepare_tumor_roi(sample, standard_size=standard_size)  # type: ignore[arg-type]


def test_empty_mask_is_rejected_defensively() -> None:
    sample = _sample(_image((10, 10)), np.zeros((10, 10), dtype=np.uint8))

    with pytest.raises(ROIError, match="empty"):
        prepare_tumor_roi(sample)


def test_image_mask_shape_mismatch_is_rejected_defensively() -> None:
    sample = _sample(_image((10, 10)), np.ones((9, 10), dtype=np.uint8))

    with pytest.raises(ROIError, match="shapes differ"):
        prepare_tumor_roi(sample)


@pytest.mark.parametrize(
    ("image", "mask"),
    [
        (np.zeros((2, 2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.uint8)),
        (np.zeros((2, 2), dtype=np.float32), np.ones((2, 2, 2), dtype=np.uint8)),
    ],
)
def test_non_2d_arrays_are_rejected(image: np.ndarray, mask: np.ndarray) -> None:
    sample = _sample(image, mask)

    with pytest.raises(ROIError, match="2D"):
        prepare_tumor_roi(sample)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nan_or_inf_values_are_rejected(bad_value: float) -> None:
    image = _image((10, 10))
    image[0, 0] = bad_value
    sample = _sample(image, _mask((10, 10), slice(2, 4), slice(2, 4)))

    with pytest.raises(ROIError, match="NaN or infinite"):
        prepare_tumor_roi(sample)
