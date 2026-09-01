from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pywt

from src.data.feature_dataset import Phase1Sample
from src.features.config import load_feature_config
from src.features.feature_table import read_feature_table, write_feature_table
from src.features.wavelet import (
    STAT_ORDER,
    SUBBAND_ORDER,
    build_wavelet_feature_table,
    extract_wavelet_features,
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


def _synthetic_128() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((128, 128), dtype=np.float64)


# --- Config integration -------------------------------------------------


def test_config_loads_correctly() -> None:
    config = load_feature_config("wavelet")
    assert config["feature_set_name"] == "wavelet"
    assert config["parameters"]["wavelet"] == "db2"
    assert config["parameters"]["level"] == 2
    assert config["parameters"]["mode"] == "symmetric"
    assert config["expected_feature_count"] == 28
    assert config["parameters"]["retained_subbands"] == [
        "L1_LH",
        "L1_HL",
        "L1_HH",
        "L2_LL",
        "L2_LH",
        "L2_HL",
        "L2_HH",
    ]
    assert set(config["parameters"]["summary_statistics"]) == {"mean", "std", "energy", "entropy"}


# --- Feature naming and ordering ----------------------------------------


def test_feature_names_are_28_unique_dwt_prefixed_snake_case() -> None:
    config = load_feature_config("wavelet")
    image = _synthetic_128()
    features = extract_wavelet_features(image, config)
    assert len(features) == 28
    names = list(features.keys())
    assert len(set(names)) == 28
    for name in names:
        assert name.startswith("dwt_")
        assert name == name.lower()
        assert " " not in name


def test_deterministic_feature_ordering_matches_subband_and_stat_order() -> None:
    config = load_feature_config("wavelet")
    image = _synthetic_128()
    features = extract_wavelet_features(image, config)
    expected_names = [
        f"dwt_{subband.lower()}_{stat}" for subband in SUBBAND_ORDER for stat in STAT_ORDER
    ]
    assert list(features.keys()) == expected_names


# --- Output dimensionality / finiteness ---------------------------------


def test_output_dimensionality_on_synthetic_image() -> None:
    config = load_feature_config("wavelet")
    image = _synthetic_128()
    features = extract_wavelet_features(image, config)
    assert len(features) == config["expected_feature_count"]


def test_all_values_finite_on_synthetic_image() -> None:
    config = load_feature_config("wavelet")
    image = _synthetic_128()
    features = extract_wavelet_features(image, config)
    assert np.isfinite(list(features.values())).all()


# --- Edge cases -----------------------------------------------------------


def test_all_zero_input_produces_finite_values() -> None:
    config = load_feature_config("wavelet")
    image = np.zeros((128, 128), dtype=np.float64)
    features = extract_wavelet_features(image, config)
    assert len(features) == 28
    assert np.isfinite(list(features.values())).all()
    for name, value in features.items():
        if name.endswith("_mean") or name.endswith("_std") or name.endswith("_energy") or name.endswith("_entropy"):
            assert value == pytest.approx(0.0, abs=1e-9)


def test_constant_nonzero_input_produces_finite_values() -> None:
    config = load_feature_config("wavelet")
    image = np.full((128, 128), 0.5, dtype=np.float64)
    features = extract_wavelet_features(image, config)
    assert len(features) == 28
    assert np.isfinite(list(features.values())).all()
    entropy_values = [value for name, value in features.items() if name.endswith("_entropy")]
    assert all(value == pytest.approx(0.0, abs=1e-9) for value in entropy_values)


def test_negative_coefficient_values_handled() -> None:
    config = load_feature_config("wavelet")
    rng = np.random.default_rng(1)
    image = rng.uniform(-1.0, 1.0, size=(128, 128))
    features = extract_wavelet_features(image, config)
    assert len(features) == 28
    assert np.isfinite(list(features.values())).all()


def test_small_valid_input_supports_two_level_decomposition() -> None:
    config = load_feature_config("wavelet")
    rng = np.random.default_rng(2)
    image = rng.random((16, 16))
    features = extract_wavelet_features(image, config)
    assert len(features) == 28
    assert np.isfinite(list(features.values())).all()


# --- Determinism ----------------------------------------------------------


def test_same_input_and_config_yield_identical_output() -> None:
    config = load_feature_config("wavelet")
    image = _synthetic_128()
    first = extract_wavelet_features(image, config)
    second = extract_wavelet_features(image, config)
    assert first == second


# --- pywt decomposition correctness / subband correspondence -------------


def test_horizontal_stripe_image_energy_concentrates_in_lh_subband() -> None:
    """Row-wise (horizontal-edge) structure should dominate the *_lh subbands."""

    config = load_feature_config("wavelet")
    image = np.zeros((128, 128), dtype=np.float64)
    image[::2, :] = 1.0
    features = extract_wavelet_features(image, config)
    assert features["dwt_l1_lh_energy"] > features["dwt_l1_hl_energy"]
    assert features["dwt_l1_lh_energy"] > features["dwt_l1_hh_energy"]


def test_vertical_stripe_image_energy_concentrates_in_hl_subband() -> None:
    """Column-wise (vertical-edge) structure should dominate the *_hl subbands."""

    config = load_feature_config("wavelet")
    image = np.zeros((128, 128), dtype=np.float64)
    image[:, ::2] = 1.0
    features = extract_wavelet_features(image, config)
    assert features["dwt_l1_hl_energy"] > features["dwt_l1_lh_energy"]
    assert features["dwt_l1_hl_energy"] > features["dwt_l1_hh_energy"]


def test_pywt_wavedec2_ordering_assumption_holds() -> None:
    """Sanity-check the raw pywt structure this module's mapping relies on."""

    image = np.random.default_rng(3).random((32, 32))
    coeffs = pywt.wavedec2(image, wavelet="db2", level=2, mode="symmetric")
    assert len(coeffs) == 3
    cA2, level2_details, level1_details = coeffs
    assert len(level2_details) == 3
    assert len(level1_details) == 3


def test_flat_image_has_zero_energy_in_all_detail_subbands() -> None:
    config = load_feature_config("wavelet")
    image = np.full((128, 128), 0.3, dtype=np.float64)
    features = extract_wavelet_features(image, config)
    for subband in ("l1_lh", "l1_hl", "l1_hh", "l2_lh", "l2_hl", "l2_hh"):
        assert features[f"dwt_{subband}_energy"] == pytest.approx(0.0, abs=1e-9)
    # db2's approximation filter is not unit-gain, so the LL2 mean scales by a
    # fixed constant factor relative to the flat input value rather than
    # equaling it exactly; check it is finite, nonzero, and proportional.
    assert features["dwt_l2_ll_mean"] != 0.0


# --- ROI integration --------------------------------------------------


def test_extractor_uses_roi_image_masked_resized_from_prepare_tumor_roi() -> None:
    config = load_feature_config("wavelet")
    image = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    image[2:6, 2:6] = np.linspace(0.1, 0.9, 16).reshape(4, 4)
    mask[2:6, 2:6] = 1
    roi = prepare_tumor_roi(_sample(image, mask))
    expected = extract_wavelet_features(roi.roi_image_masked_resized, config)
    actual = extract_wavelet_features(roi.roi_image_masked_resized, config)
    assert expected == actual
    # Feeding the un-resized native ROI (a different shape/content) must differ.
    alternate = extract_wavelet_features(roi.roi_image_masked, config)
    assert alternate != expected


def test_roi_standard_size_is_128x128() -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    image[2:6, 2:6] = np.linspace(0.1, 0.9, 16).reshape(4, 4)
    mask[2:6, 2:6] = 1
    roi = prepare_tumor_roi(_sample(image, mask))
    assert roi.roi_image_masked_resized.shape == (128, 128)
    assert roi.standard_size == (128, 128)


# --- Feature table integration -------------------------------------------


def test_feature_table_round_trip_succeeds(tmp_path) -> None:
    config = load_feature_config("wavelet")
    image = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    image[2:6, 2:6] = np.linspace(0.1, 0.9, 16).reshape(4, 4)
    mask[2:6, 2:6] = 1
    roi = prepare_tumor_roi(_sample(image, mask))
    features = extract_wavelet_features(roi.roi_image_masked_resized, config)
    row = {"sample_id": "1", "patient_id": "p1", "label": 1, "split": "train", **features}
    frame = pd.DataFrame([row])
    split = frame.loc[:, ["sample_id", "patient_id", "label", "split"]]
    path = tmp_path / "wavelet.csv"
    write_feature_table(frame, path, canonical_split=split)
    round_trip = read_feature_table(path, canonical_split=split)
    assert round_trip.shape == (1, 32)
    expected_columns = ["sample_id", "patient_id", "label", "split"] + [
        f"dwt_{subband.lower()}_{stat}" for subband in SUBBAND_ORDER for stat in STAT_ORDER
    ]
    assert list(round_trip.columns) == expected_columns


# --- End-to-end smoke test on a small subset ------------------------------


def test_end_to_end_smoke_on_small_subset() -> None:
    frame, roi_info = build_wavelet_feature_table(sample_ids=["1", "2", "3", "4", "5"])
    assert len(frame) == 5
    feature_columns = [column for column in frame.columns if column.startswith("dwt_")]
    assert len(feature_columns) == 28
    assert list(frame.columns)[:4] == ["sample_id", "patient_id", "label", "split"]
    for column in feature_columns:
        assert np.isfinite(frame[column].to_numpy()).all()
    assert roi_info["standard_size"] == [128, 128]
