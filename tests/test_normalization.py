import numpy as np
import pytest

from src.preprocessing.normalization import robust_foreground_percentile_normalize


def test_standard_positive_image_is_float32_and_bounded() -> None:
    image = np.arange(1, 101, dtype=np.int16).reshape(10, 10)

    normalized, metadata = robust_foreground_percentile_normalize(image)

    assert normalized.dtype == np.float32
    assert normalized.shape == image.shape
    assert float(normalized.min()) == pytest.approx(0.0)
    assert float(normalized.max()) == pytest.approx(1.0)
    assert metadata["foreground_pixel_count"] == 100
    assert not metadata["empty_foreground"]
    assert not metadata["constant_foreground"]
    assert np.isfinite(normalized).all()


def test_zero_background_remains_zero() -> None:
    image = np.array([[0, 0, 10], [0, 20, 30]], dtype=np.float32)

    normalized, _ = robust_foreground_percentile_normalize(image)

    np.testing.assert_array_equal(normalized[image == 0], 0.0)
    assert np.isfinite(normalized).all()


def test_all_zero_image_returns_safe_empty_foreground() -> None:
    normalized, metadata = robust_foreground_percentile_normalize(
        np.zeros((4, 5), dtype=np.int16)
    )

    np.testing.assert_array_equal(normalized, 0.0)
    assert normalized.dtype == np.float32
    assert metadata["empty_foreground"] is True
    assert metadata["constant_foreground"] is False
    assert "empty foreground" in metadata["normalization_warning"]


def test_constant_nonzero_foreground_returns_safe_finite_result() -> None:
    image = np.array([[0, 7, 7], [0, 7, 7]], dtype=np.float32)

    normalized, metadata = robust_foreground_percentile_normalize(image)

    np.testing.assert_array_equal(normalized, 0.0)
    assert metadata["constant_foreground"] is True
    assert metadata["empty_foreground"] is False
    assert "constant foreground" in metadata["normalization_warning"]
    assert np.isfinite(normalized).all()


def test_extreme_outlier_is_clipped_without_nonfinite_values() -> None:
    foreground = np.arange(1, 101, dtype=np.float32)
    foreground[-1] = 1_000_000
    image = foreground.reshape(10, 10)

    normalized, metadata = robust_foreground_percentile_normalize(image)

    assert metadata["normalization_high"] < 1_000_000
    assert normalized[-1, -1] == pytest.approx(1.0)
    assert 0.0 <= float(normalized.min()) <= float(normalized.max()) <= 1.0
    assert np.isfinite(normalized).all()


@pytest.mark.parametrize(
    "image",
    [
        np.array([[np.nan, 1.0]], dtype=np.float32),
        np.array([[np.inf, 1.0]], dtype=np.float32),
    ],
)
def test_nonfinite_input_is_rejected(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        robust_foreground_percentile_normalize(image)
