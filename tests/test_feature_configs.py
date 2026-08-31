from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from skimage.feature import local_binary_pattern

from src.features.config import (
    FEATURE_CONFIG_NAMES,
    FeatureConfigError,
    get_feature_config_path,
    load_all_feature_configs,
    load_feature_config,
    validate_feature_config,
)


def test_all_five_config_files_load() -> None:
    configs = load_all_feature_configs()

    assert set(configs) == {"glcm", "lbp", "wavelet", "hog", "gabor"}


def test_required_common_fields_exist() -> None:
    for config in load_all_feature_configs().values():
        assert {
            "feature_set_name",
            "feature_version",
            "algorithm",
            "feature_prefix",
            "input_representation",
            "roi_policy",
            "parameters",
            "expected_feature_count",
        }.issubset(config)


def test_feature_names_and_prefixes_are_unique() -> None:
    configs = load_all_feature_configs()

    assert len(set(configs)) == len(FEATURE_CONFIG_NAMES)
    prefixes = [config["feature_prefix"] for config in configs.values()]
    assert len(prefixes) == len(set(prefixes))


def test_expected_feature_counts() -> None:
    configs = load_all_feature_configs()

    assert configs["glcm"]["expected_feature_count"] == 12
    assert configs["lbp"]["expected_feature_count"] == 10
    assert configs["wavelet"]["expected_feature_count"] == 28
    assert configs["hog"]["expected_feature_count"] == 1764
    assert configs["gabor"]["expected_feature_count"] == 60


def test_lbp_expected_feature_count_matches_skimage_uniform_convention() -> None:
    config = load_feature_config("lbp")
    params = config["parameters"]
    # skimage uniform LBP labels non-uniform patterns as P + 1, so histogram
    # bins over integer codes are 0 through P + 1: P + 2 bins.
    assert params["expected_bin_count"] == params["P"] + 2
    image = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    assert int(local_binary_pattern(image, 8, 1, "uniform").max()) <= 9


def test_glcm_parameters_are_frozen() -> None:
    params = load_feature_config("glcm")["parameters"]

    assert params["gray_levels"] == 32
    assert params["distances"] == [1, 2, 4]
    assert params["angles_degrees"] == [0, 45, 90, 135]
    assert params["properties"] == [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]
    assert params["aggregation"] == ["mean", "std"]


def test_glcm_mask_policy_excludes_non_tumor_pairs() -> None:
    policy = " ".join(load_feature_config("glcm")["parameters"]["mask_policy"])

    assert "BOTH pixels" in policy
    assert "tumor_mask" in policy
    assert "do not include zero-filled contextual background" in policy


def test_lbp_parameters_are_frozen() -> None:
    params = load_feature_config("lbp")["parameters"]

    assert params["P"] == 8
    assert params["R"] == 1
    assert params["method"] == "uniform"


def test_wavelet_parameters_are_frozen_and_include_ll2() -> None:
    params = load_feature_config("wavelet")["parameters"]

    assert params["wavelet"] == "db2"
    assert params["level"] == 2
    assert "L2_LL" in params["retained_subbands"]
    assert params["retained_subbands"] == [
        "L1_LH",
        "L1_HL",
        "L1_HH",
        "L2_LL",
        "L2_LH",
        "L2_HL",
        "L2_HH",
    ]


def test_hog_128_compatibility_and_dimension_formula() -> None:
    params = load_feature_config("hog")["parameters"]

    assert params["input_size"] == [128, 128]
    cells_y = params["input_size"][0] // params["pixels_per_cell"][0]
    cells_x = params["input_size"][1] // params["pixels_per_cell"][1]
    blocks_y = cells_y - params["cells_per_block"][0] + 1
    blocks_x = cells_x - params["cells_per_block"][1] + 1
    count = (
        blocks_y
        * blocks_x
        * params["cells_per_block"][0]
        * params["cells_per_block"][1]
        * params["orientations"]
    )

    assert cells_y == cells_x == 8
    assert blocks_y == blocks_x == 7
    assert count == 1764


def test_gabor_filter_bank_shape_is_frozen() -> None:
    params = load_feature_config("gabor")["parameters"]

    assert len(params["frequencies"]) == 3
    assert params["frequencies"] == [0.10, 0.20, 0.30]
    assert len(params["orientations_degrees"]) == 4
    assert params["orientations_degrees"] == [0, 45, 90, 135]
    assert len(params["response_statistics"]) == 5


def test_invalid_feature_config_rejected() -> None:
    config = deepcopy(load_feature_config("glcm"))
    del config["feature_prefix"]

    with pytest.raises(FeatureConfigError, match="missing field"):
        validate_feature_config(config)


def test_invalid_algorithm_specific_parameter_rejected() -> None:
    config = deepcopy(load_feature_config("hog"))
    config["parameters"]["pixels_per_cell"] = [15, 16]

    with pytest.raises(FeatureConfigError, match="divisible"):
        validate_feature_config(config)


def test_loader_resolves_configs_repository_relatively() -> None:
    path = get_feature_config_path("glcm")

    assert path == Path.cwd() / "configs" / "features" / "glcm.yaml"
    assert load_feature_config("glcm")["feature_set_name"] == "glcm"
    assert load_feature_config("configs/features/glcm.yaml")["feature_set_name"] == "glcm"
