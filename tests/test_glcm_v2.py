from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.feature_dataset import Phase1Sample
from src.features.config import load_feature_config
from src.features.feature_table import read_feature_table, write_feature_table
from src.features.glcm import extract_glcm_features, prepare_tumor_roi
from src.features.glcm_v2 import (
    GLCMV2ExtractionError,
    extract_glcm_v2_features,
    extract_glcm_v2_features_from_roi,
    glcm_v2_feature_columns,
)


def _sample(size: int = 32, mask_slice=slice(8, 24)) -> Phase1Sample:
    image = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    image = ((yy + 2 * xx) / float(3 * size)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[mask_slice, mask_slice] = 1
    return Phase1Sample(
        sample_id="1",
        patient_id="P1",
        label=1,
        split="test",
        image_normalized=image,
        tumor_mask=mask,
        image_raw=image.copy(),
        tumor_border=mask.copy(),
    )


def test_v2_config_loads() -> None:
    config = load_feature_config("glcm_v2")
    assert config["feature_set_name"] == "glcm_v2"
    assert config["feature_version"] == "v2"
    assert config["expected_feature_count"] == 84


def test_exact_feature_order_and_count() -> None:
    columns = glcm_v2_feature_columns()
    assert len(columns) == 84
    assert columns[:4] == [
        "glcm_contrast_d1_a0",
        "glcm_contrast_d1_a45",
        "glcm_contrast_d1_a90",
        "glcm_contrast_d1_a135",
    ]
    assert columns[12:14] == ["glcm_contrast_mean", "glcm_contrast_std"]
    assert columns[-2:] == ["glcm_asm_mean", "glcm_asm_std"]


def test_directional_values_and_aggregates_are_consistent() -> None:
    config = load_feature_config("glcm_v2")
    roi = prepare_tumor_roi(_sample())
    features = extract_glcm_v2_features_from_roi(roi, config)
    contrast_values = np.asarray(
        [
            features[f"glcm_contrast_d{distance}_a{angle}"]
            for distance in (1, 2, 4)
            for angle in (0, 45, 90, 135)
        ]
    )
    assert features["glcm_contrast_mean"] == pytest.approx(float(contrast_values.mean()))
    assert features["glcm_contrast_std"] == pytest.approx(float(contrast_values.std(ddof=0)))


def test_v1_aggregate_values_match_v2_aggregates() -> None:
    sample = _sample()
    v1 = extract_glcm_features(sample)
    v2 = extract_glcm_v2_features(sample)
    for property_name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm"):
        for aggregation in ("mean", "std"):
            key = f"glcm_{property_name}_{aggregation}"
            assert v2[key] == pytest.approx(v1[key], rel=1e-12, abs=1e-12)


def test_masked_background_invariance() -> None:
    base = _sample()
    changed = _sample()
    altered_image = changed.image_normalized.copy()
    altered_image[changed.tumor_mask == 0] = 1.0 - altered_image[changed.tumor_mask == 0]
    changed = Phase1Sample(
        sample_id=changed.sample_id,
        patient_id=changed.patient_id,
        label=changed.label,
        split=changed.split,
        image_normalized=altered_image,
        tumor_mask=changed.tumor_mask,
        image_raw=changed.image_raw,
        tumor_border=changed.tumor_border,
    )
    first = extract_glcm_v2_features(base)
    second = extract_glcm_v2_features(changed)
    for column in glcm_v2_feature_columns():
        assert second[column] == pytest.approx(first[column])


def test_no_nan_inf_and_deterministic() -> None:
    sample = _sample()
    first = extract_glcm_v2_features(sample)
    second = extract_glcm_v2_features(sample)
    values = np.asarray([first[column] for column in glcm_v2_feature_columns()], dtype=float)
    assert np.isfinite(values).all()
    assert first == second


def test_source_shape_supports_256_and_512() -> None:
    assert len(extract_glcm_v2_features(_sample(256))) == 88
    assert len(extract_glcm_v2_features(_sample(512))) == 88


def test_tiny_tumor_with_some_invalid_pairs_is_finite() -> None:
    sample = _sample(size=16, mask_slice=slice(7, 9))
    features = extract_glcm_v2_features(sample)
    values = np.asarray([features[column] for column in glcm_v2_feature_columns()], dtype=float)
    assert np.isfinite(values).all()
    assert len(values) == 84


def test_all_invalid_pair_case_raises() -> None:
    sample = _sample(size=16, mask_slice=slice(8, 9))
    with pytest.raises(GLCMV2ExtractionError, match="no valid tumor pixel pairs"):
        extract_glcm_v2_features(sample)


def test_feature_table_round_trip(tmp_path: Path) -> None:
    row = extract_glcm_v2_features(_sample())
    frame = pd.DataFrame([row])
    split = pd.DataFrame(
        [{"sample_id": 1, "patient_id": "P1", "label": 1, "split": "test"}]
    )
    output = tmp_path / "glcm_v2.csv"
    write_feature_table(frame, output, canonical_split=split)
    loaded = read_feature_table(output, canonical_split=split)
    assert loaded.shape == (1, 88)
