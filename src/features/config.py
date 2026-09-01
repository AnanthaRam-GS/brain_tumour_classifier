"""Load and validate frozen Phase 1 handcrafted feature configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_CONFIG_DIR = PROJECT_ROOT / "configs" / "features"

FEATURE_CONFIG_NAMES = ("glcm", "lbp", "wavelet", "hog", "gabor")
EXPERIMENTAL_FEATURE_CONFIG_NAMES = ("glcm_v2",)
SUPPORTED_FEATURE_CONFIG_NAMES = FEATURE_CONFIG_NAMES + EXPERIMENTAL_FEATURE_CONFIG_NAMES
REQUIRED_FIELDS = {
    "feature_set_name",
    "feature_version",
    "algorithm",
    "feature_prefix",
    "input_representation",
    "roi_policy",
    "parameters",
    "expected_feature_count",
}
SUPPORTED_ALGORITHMS = {
    "glcm": "gray_level_cooccurrence_matrix",
    "glcm_v2": "gray_level_cooccurrence_matrix",
    "lbp": "local_binary_pattern",
    "wavelet": "2d_discrete_wavelet_transform",
    "hog": "histogram_of_oriented_gradients",
    "gabor": "gabor_filter_bank",
}
EXPECTED_REPRESENTATIONS = {
    "glcm": "native_roi_image_and_mask",
    "glcm_v2": "native_roi_image_and_mask",
    "lbp": "native_roi_image_and_mask",
    "wavelet": "roi_image_masked_resized",
    "hog": "roi_image_masked_resized",
    "gabor": "roi_image_masked_resized",
}
GLCM_PROPERTIES = {"contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"}
AGGREGATIONS = {"mean", "std"}
LBP_METHODS = {"default", "ror", "uniform", "nri_uniform", "var"}
WAVELET_STATS = {"mean", "std", "energy", "entropy"}
GABOR_STATS = {"real_mean", "real_std", "magnitude_mean", "magnitude_std", "magnitude_energy"}


class FeatureConfigError(ValueError):
    """Raised when a feature configuration is missing or invalid."""


def get_feature_config_path(feature_set_name: str) -> Path:
    """Return the repository-relative path for a supported feature config."""

    name = str(feature_set_name).strip().lower()
    if name not in SUPPORTED_FEATURE_CONFIG_NAMES:
        raise FeatureConfigError(f"unsupported feature_set_name: {feature_set_name}")
    return FEATURE_CONFIG_DIR / f"{name}.yaml"


def load_feature_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load and validate a feature config by supported name or explicit path."""

    candidate = Path(name_or_path)
    path = get_feature_config_path(str(name_or_path)) if candidate.suffix == "" else candidate
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FeatureConfigError(f"feature config does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise FeatureConfigError("feature config must be a YAML mapping")
    return validate_feature_config(config)


def validate_feature_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate common and algorithm-specific feature config fields."""

    missing = sorted(REQUIRED_FIELDS.difference(config))
    if missing:
        raise FeatureConfigError(f"feature config missing field(s): {', '.join(missing)}")
    name = _nonblank(config["feature_set_name"], "feature_set_name")
    if name not in SUPPORTED_FEATURE_CONFIG_NAMES:
        raise FeatureConfigError(f"unsupported feature_set_name: {name}")
    if config["algorithm"] != SUPPORTED_ALGORITHMS[name]:
        raise FeatureConfigError(f"unsupported algorithm for {name}: {config['algorithm']}")
    _nonblank(config["feature_version"], "feature_version")
    _nonblank(config["feature_prefix"], "feature_prefix")
    _nonblank(config["roi_policy"], "roi_policy")
    if config["input_representation"] != EXPECTED_REPRESENTATIONS[name]:
        raise FeatureConfigError(
            f"unexpected input_representation for {name}: {config['input_representation']}"
        )
    if not isinstance(config["parameters"], dict):
        raise FeatureConfigError("parameters must be a mapping")
    expected_count = _positive_int(config["expected_feature_count"], "expected_feature_count")

    validators = {
        "glcm": _validate_glcm,
        "glcm_v2": _validate_glcm_v2,
        "lbp": _validate_lbp,
        "wavelet": _validate_wavelet,
        "hog": _validate_hog,
        "gabor": _validate_gabor,
    }
    derived_count = validators[name](config["parameters"])
    if expected_count != derived_count:
        raise FeatureConfigError(
            f"expected_feature_count {expected_count} does not match derived count {derived_count}"
        )
    return config


def load_all_feature_configs() -> dict[str, dict[str, Any]]:
    """Load all official Review 1 feature configurations."""

    return {name: load_feature_config(name) for name in FEATURE_CONFIG_NAMES}


def _nonblank(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise FeatureConfigError(f"{field} must be populated")
    return text


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise FeatureConfigError(f"{field} must be a positive integer")
    return value


def _number_list(values: Any, field: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise FeatureConfigError(f"{field} must be a non-empty list")
    numbers = []
    for value in values:
        if not isinstance(value, (int, float)) or value <= 0:
            raise FeatureConfigError(f"{field} values must be positive numbers")
        numbers.append(float(value))
    return numbers


def _positive_int_list(values: Any, field: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise FeatureConfigError(f"{field} must be a non-empty list")
    output = []
    for value in values:
        if not isinstance(value, int) or value <= 0:
            raise FeatureConfigError(f"{field} values must be positive integers")
        output.append(value)
    return output


def _angle_list(values: Any, field: str) -> list[int | float]:
    if not isinstance(values, list) or not values:
        raise FeatureConfigError(f"{field} must be a non-empty list")
    for value in values:
        if not isinstance(value, (int, float)) or value < 0 or value >= 180:
            raise FeatureConfigError(f"{field} values must be in [0, 180)")
    return values


def _require_subset(values: Any, allowed: set[str], field: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise FeatureConfigError(f"{field} must be a non-empty list")
    invalid = sorted(set(values).difference(allowed))
    if invalid:
        raise FeatureConfigError(f"unsupported {field}: {invalid}")
    return values


def _pair(values: Any, field: str) -> tuple[int, int]:
    if (
        not isinstance(values, list)
        or len(values) != 2
        or not all(isinstance(value, int) and value > 0 for value in values)
    ):
        raise FeatureConfigError(f"{field} must contain two positive integers")
    return int(values[0]), int(values[1])


def _validate_glcm(parameters: dict[str, Any]) -> int:
    gray_levels = _positive_int(parameters.get("gray_levels"), "gray_levels")
    if gray_levels <= 1:
        raise FeatureConfigError("gray_levels must be greater than 1")
    _positive_int_list(parameters.get("distances"), "distances")
    _angle_list(parameters.get("angles_degrees"), "angles_degrees")
    properties = _require_subset(parameters.get("properties"), GLCM_PROPERTIES, "properties")
    aggregation = _require_subset(parameters.get("aggregation"), AGGREGATIONS, "aggregation")
    quantization = parameters.get("quantization")
    if not isinstance(quantization, dict):
        raise FeatureConfigError("quantization must be a mapping")
    if quantization.get("method") != "uniform":
        raise FeatureConfigError("GLCM quantization method must be uniform")
    if quantization.get("levels") != gray_levels:
        raise FeatureConfigError("GLCM quantization levels must match gray_levels")
    source_range = quantization.get("source_range")
    if source_range != [0.0, 1.0]:
        raise FeatureConfigError("GLCM quantization source_range must be [0.0, 1.0]")
    mask_policy = parameters.get("mask_policy")
    if not isinstance(mask_policy, list) or not any(
        "BOTH pixels" in str(item) and "tumor_mask" in str(item) for item in mask_policy
    ):
        raise FeatureConfigError("GLCM mask_policy must require both pixels inside tumor_mask")
    return len(properties) * len(aggregation)


def _validate_glcm_v2(parameters: dict[str, Any]) -> int:
    properties, aggregation = _validate_glcm_common(parameters)
    distances = _positive_int_list(parameters.get("distances"), "distances")
    angles = _angle_list(parameters.get("angles_degrees"), "angles_degrees")
    if parameters.get("directional_features") is not True:
        raise FeatureConfigError("GLCM v2 must enable directional_features")
    return len(properties) * len(distances) * len(angles) + len(properties) * len(aggregation)


def _validate_glcm_common(parameters: dict[str, Any]) -> tuple[list[str], list[str]]:
    gray_levels = _positive_int(parameters.get("gray_levels"), "gray_levels")
    if gray_levels <= 1:
        raise FeatureConfigError("gray_levels must be greater than 1")
    _positive_int_list(parameters.get("distances"), "distances")
    _angle_list(parameters.get("angles_degrees"), "angles_degrees")
    properties = _require_subset(parameters.get("properties"), GLCM_PROPERTIES, "properties")
    aggregation = _require_subset(parameters.get("aggregation"), AGGREGATIONS, "aggregation")
    quantization = parameters.get("quantization")
    if not isinstance(quantization, dict):
        raise FeatureConfigError("quantization must be a mapping")
    if quantization.get("method") != "uniform":
        raise FeatureConfigError("GLCM quantization method must be uniform")
    if quantization.get("levels") != gray_levels:
        raise FeatureConfigError("GLCM quantization levels must match gray_levels")
    source_range = quantization.get("source_range")
    if source_range != [0.0, 1.0]:
        raise FeatureConfigError("GLCM quantization source_range must be [0.0, 1.0]")
    mask_policy = parameters.get("mask_policy")
    if not isinstance(mask_policy, list) or not any(
        "BOTH pixels" in str(item) and "tumor_mask" in str(item) for item in mask_policy
    ):
        raise FeatureConfigError("GLCM mask_policy must require both pixels inside tumor_mask")
    return properties, aggregation


def _validate_lbp(parameters: dict[str, Any]) -> int:
    p_value = _positive_int(parameters.get("P"), "P")
    radius = parameters.get("R")
    if not isinstance(radius, (int, float)) or radius <= 0:
        raise FeatureConfigError("R must be positive")
    method = parameters.get("method")
    if method not in LBP_METHODS:
        raise FeatureConfigError(f"unsupported LBP method: {method}")
    histogram = parameters.get("histogram")
    if not isinstance(histogram, dict) or histogram.get("use_pixels") != "tumor_mask_only":
        raise FeatureConfigError("LBP histogram must use tumor_mask_only pixels")
    if histogram.get("normalized") is not True:
        raise FeatureConfigError("LBP histogram must be normalized")
    expected_bins = parameters.get("expected_bin_count")
    derived_bins = p_value + 2 if method == "uniform" else expected_bins
    if expected_bins != derived_bins:
        raise FeatureConfigError("LBP expected_bin_count does not match method convention")
    return int(derived_bins)


def _validate_wavelet(parameters: dict[str, Any]) -> int:
    _nonblank(parameters.get("wavelet"), "wavelet")
    _positive_int(parameters.get("level"), "level")
    _nonblank(parameters.get("mode"), "mode")
    subbands = parameters.get("retained_subbands")
    required_subbands = ["L1_LH", "L1_HL", "L1_HH", "L2_LL", "L2_LH", "L2_HL", "L2_HH"]
    if subbands != required_subbands:
        raise FeatureConfigError("wavelet retained_subbands must include L1 details and LL2")
    stats = _require_subset(parameters.get("summary_statistics"), WAVELET_STATS, "summary_statistics")
    if set(stats) != WAVELET_STATS:
        raise FeatureConfigError("wavelet summary_statistics must include mean, std, energy, entropy")
    return len(required_subbands) * len(stats)


def _validate_hog(parameters: dict[str, Any]) -> int:
    input_h, input_w = _pair(parameters.get("input_size"), "input_size")
    orientations = _positive_int(parameters.get("orientations"), "orientations")
    cell_h, cell_w = _pair(parameters.get("pixels_per_cell"), "pixels_per_cell")
    block_h, block_w = _pair(parameters.get("cells_per_block"), "cells_per_block")
    if input_h % cell_h or input_w % cell_w:
        raise FeatureConfigError("HOG input_size must be divisible by pixels_per_cell")
    cells_h = input_h // cell_h
    cells_w = input_w // cell_w
    blocks_h = cells_h - block_h + 1
    blocks_w = cells_w - block_w + 1
    if blocks_h <= 0 or blocks_w <= 0:
        raise FeatureConfigError("HOG cells_per_block is incompatible with input_size")
    if parameters.get("feature_vector") is not True:
        raise FeatureConfigError("HOG feature_vector must be true")
    if not isinstance(parameters.get("transform_sqrt"), bool):
        raise FeatureConfigError("HOG transform_sqrt must be boolean")
    _nonblank(parameters.get("block_norm"), "block_norm")
    return blocks_h * blocks_w * block_h * block_w * orientations


def _validate_gabor(parameters: dict[str, Any]) -> int:
    frequencies = _number_list(parameters.get("frequencies"), "frequencies")
    orientations = _angle_list(parameters.get("orientations_degrees"), "orientations_degrees")
    if parameters.get("n_stds") != 3:
        raise FeatureConfigError("Gabor n_stds must be 3")
    if parameters.get("offset") != 0:
        raise FeatureConfigError("Gabor offset must be 0")
    stats = _require_subset(parameters.get("response_statistics"), GABOR_STATS, "response_statistics")
    if set(stats) != GABOR_STATS:
        raise FeatureConfigError("Gabor response_statistics must include all frozen statistics")
    return len(frequencies) * len(orientations) * len(stats)
