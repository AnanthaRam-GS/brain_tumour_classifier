from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_dataset import Phase1Sample
from src.features.config import load_feature_config
from src.features.feature_table import read_feature_table, write_feature_table
from src.features.glcm import (
    GLCM_FEATURE_COLUMNS,
    GLCMExtractionError,
    build_masked_glcm,
    compute_glcm_properties,
    extract_glcm_features,
    extract_glcm_features_from_roi,
    offset_for_angle,
    quantize_glcm_image,
)
from src.preprocessing.roi import prepare_tumor_roi


def _sample(image: np.ndarray, mask: np.ndarray, sample_id: str = "1") -> Phase1Sample:
    return Phase1Sample(
        sample_id=sample_id,
        patient_id=f"p{sample_id}",
        label=1,
        split="train",
        image_normalized=image.astype(np.float32),
        tumor_mask=mask.astype(np.uint8),
        image_raw=(image * 100).astype(np.float32),
        tumor_border=np.array([0.0, 1.0]),
    )


def test_config_loads_correctly() -> None:
    config = load_feature_config("glcm")
    assert config["feature_set_name"] == "glcm"
    assert config["parameters"]["gray_levels"] == 32
    assert config["expected_feature_count"] == 12


def test_quantization_bounds_and_range() -> None:
    image = np.array([[0.0, 0.1, 0.999], [1.0, 0.5, 0.03125]], dtype=np.float32)
    quantized = quantize_glcm_image(image, gray_levels=32)
    assert quantized[0, 0] == 0
    assert quantized[1, 0] == 31
    assert quantized.min() >= 0
    assert quantized.max() <= 31
    assert np.issubdtype(quantized.dtype, np.integer)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0, (0, 1)), (45, (-1, 1)), (90, (-1, 0)), (135, (-1, -1))],
)
def test_angle_offsets_are_explicit(angle: int, expected: tuple[int, int]) -> None:
    assert offset_for_angle(1, angle) == expected


@pytest.mark.parametrize("distance", [1, 2, 4])
def test_masked_glcm_distances_work(distance: int) -> None:
    image = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
    mask = np.ones((8, 8), dtype=np.uint8)
    matrix, valid_pairs = build_masked_glcm(
        image, mask, gray_levels=8, distance=distance, angle_degrees=0
    )
    assert valid_pairs == 8 * (8 - distance)
    assert np.isclose(matrix.sum(), 1.0)


def test_mask_aware_pair_construction_excludes_background() -> None:
    image = np.array([[1, 2, 7], [3, 4, 7], [7, 7, 7]], dtype=np.uint8)
    mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    matrix, valid_pairs = build_masked_glcm(
        image, mask, gray_levels=8, distance=1, angle_degrees=0, symmetric=False, normed=False
    )
    assert valid_pairs == 2
    assert matrix[1, 2] == 1
    assert matrix[3, 4] == 1
    assert matrix[:, 7].sum() == 0
    assert matrix[7, :].sum() == 0


def test_symmetric_glcm_and_normalization() -> None:
    image = np.array([[1, 2]], dtype=np.uint8)
    mask = np.ones_like(image)
    matrix, valid_pairs = build_masked_glcm(image, mask, gray_levels=4, distance=1, angle_degrees=0)
    assert valid_pairs == 1
    assert np.isclose(matrix[1, 2], 0.5)
    assert np.isclose(matrix[2, 1], 0.5)
    assert np.isclose(matrix.sum(), 1.0)


def test_glcm_properties_match_known_synthetic_example() -> None:
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[1, 2] = 0.5
    matrix[2, 1] = 0.5
    props = compute_glcm_properties(matrix)
    assert props["contrast"] == pytest.approx(1.0)
    assert props["dissimilarity"] == pytest.approx(1.0)
    assert props["homogeneity"] == pytest.approx(0.5)
    assert props["ASM"] == pytest.approx(0.5)
    assert props["energy"] == pytest.approx(np.sqrt(0.5))
    assert np.isfinite(props["correlation"])


def test_final_output_schema_finite_and_deterministic() -> None:
    config = load_feature_config("glcm")
    image = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
            [0.4, 0.5, 0.6, 0.7],
        ],
        dtype=np.float32,
    )
    mask = np.ones((4, 4), dtype=np.uint8)
    roi = prepare_tumor_roi(_sample(image, mask))
    first = extract_glcm_features_from_roi(roi, config)
    second = extract_glcm_features_from_roi(roi, config)
    assert list(first) == GLCM_FEATURE_COLUMNS
    assert len(first) == 12
    assert first == second
    assert np.isfinite(list(first.values())).all()


def test_contextual_zero_or_nonzero_background_does_not_alter_results() -> None:
    config = load_feature_config("glcm")
    tumor = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    image_a = np.zeros((6, 6), dtype=np.float32)
    image_b = np.ones((6, 6), dtype=np.float32)
    image_a[2:4, 2:4] = tumor
    image_b[2:4, 2:4] = tumor
    roi_a = prepare_tumor_roi(_sample(image_a, mask))
    roi_b = prepare_tumor_roi(_sample(image_b, mask))
    assert extract_glcm_features_from_roi(roi_a, config) == extract_glcm_features_from_roi(roi_b, config)


def test_tiny_tumor_skips_invalid_large_distances_but_extracts_valid_pairs() -> None:
    config = load_feature_config("glcm")
    image = np.array([[0.1, 0.5]], dtype=np.float32)
    mask = np.ones_like(image, dtype=np.uint8)
    roi = prepare_tumor_roi(_sample(image, mask))
    features = extract_glcm_features_from_roi(roi, config)
    assert len(features) == 12
    assert np.isfinite(list(features.values())).all()


def test_all_invalid_pairs_raise_descriptive_failure() -> None:
    config = load_feature_config("glcm")
    image = np.array([[0.5]], dtype=np.float32)
    mask = np.ones_like(image, dtype=np.uint8)
    roi = prepare_tumor_roi(_sample(image, mask))
    with pytest.raises(GLCMExtractionError, match="no valid tumor pixel pairs"):
        extract_glcm_features_from_roi(roi, config)


@pytest.mark.parametrize("shape", [(256, 256), (512, 512)])
def test_source_shapes_supported(shape: tuple[int, int]) -> None:
    image = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    image[10:15, 10:15] = np.linspace(0.1, 0.9, 25).reshape(5, 5)
    mask[10:15, 10:15] = 1
    features = extract_glcm_features(_sample(image, mask))
    assert list(features.keys()) == ["sample_id", "patient_id", "label", "split"] + GLCM_FEATURE_COLUMNS


def test_feature_table_round_trip_succeeds(tmp_path) -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    image[2:6, 2:6] = np.linspace(0.1, 0.9, 16).reshape(4, 4)
    mask[2:6, 2:6] = 1
    row = extract_glcm_features(_sample(image, mask))
    frame = pd.DataFrame([row])
    split = frame.loc[:, ["sample_id", "patient_id", "label", "split"]]
    path = tmp_path / "glcm.csv"
    write_feature_table(frame, path, canonical_split=split)
    round_trip = read_feature_table(path, canonical_split=split)
    assert round_trip.shape == (1, 16)
    assert list(round_trip.columns) == ["sample_id", "patient_id", "label", "split"] + GLCM_FEATURE_COLUMNS
