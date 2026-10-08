"""Parity between btdl's ported ROI geometry and Phase 1's native roi_image.

Phase 1 (src.preprocessing.roi) is imported here ONLY, with sys.path
modified inside this test module -- never from btdl runtime code.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from btdl import config
from btdl.preprocessing.roi import compute_roi

_REPO_ROOT = config.repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.data.feature_dataset import Phase1Sample
    from src.preprocessing.roi import prepare_tumor_roi

    _PHASE1_AVAILABLE = True
    _PHASE1_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only when Phase 1 is unavailable
    _PHASE1_AVAILABLE = False
    _PHASE1_IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    not _PHASE1_AVAILABLE, reason=f"Phase 1 (src.preprocessing.roi) unavailable: {_PHASE1_IMPORT_ERROR}"
)


@pytest.fixture(scope="module")
def cfg():
    return config.load_contract("input")


def _phase1_roi(image, mask):
    sample = Phase1Sample(
        sample_id="synthetic",
        patient_id="synthetic",
        label=1,
        split="train",
        image_normalized=image,
        tumor_mask=mask,
        image_raw=image,
        tumor_border=np.zeros((0,)),
    )
    return prepare_tumor_roi(sample)


def _synthetic_cases():
    rng = np.random.RandomState(0)
    cases = {}

    # Fully interior mask.
    image = rng.rand(100, 100).astype(np.float32)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:50, 40:50] = 1
    cases["interior"] = (image, mask)

    # Touches top-left corner.
    image = rng.rand(60, 60).astype(np.float32)
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[0:4, 0:4] = 1
    cases["top_left_corner"] = (image, mask)

    # Touches bottom-right corner.
    image = rng.rand(60, 60).astype(np.float32)
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[56:60, 56:60] = 1
    cases["bottom_right_corner"] = (image, mask)

    # Touches top edge only (not corners).
    image = rng.rand(60, 60).astype(np.float32)
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[0:3, 25:35] = 1
    cases["top_edge"] = (image, mask)

    # Non-square tight bbox.
    image = rng.rand(80, 80).astype(np.float32)
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:15, 20:50] = 1
    cases["non_square"] = (image, mask)

    # Odd tight-bbox side.
    image = rng.rand(50, 50).astype(np.float32)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:27, 20:27] = 1  # 7x7
    cases["odd_side"] = (image, mask)

    # Even tight-bbox side.
    image = rng.rand(50, 50).astype(np.float32)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:28, 20:28] = 1  # 8x8
    cases["even_side"] = (image, mask)

    # Single pixel mask.
    image = rng.rand(40, 40).astype(np.float32)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[20, 20] = 1
    cases["single_pixel"] = (image, mask)

    return cases


@pytest.mark.parametrize("case_name", sorted(_synthetic_cases().keys()))
def test_synthetic_parity_exact(cfg, case_name):
    image, mask = _synthetic_cases()[case_name]
    phase1 = _phase1_roi(image, mask)
    _, btdl_crop = compute_roi(image, mask, cfg)
    assert np.array_equal(phase1.roi_image, btdl_crop)


@pytest.mark.data
@pytest.mark.slow
def test_real_data_parity_all_samples():
    contract = config.load_contract("data")
    samples_dir = config.resolve_data_path(contract["samples_dir"])
    if not samples_dir.is_dir() or not any(samples_dir.glob("*.npz")):
        pytest.skip("real NPZ dataset not present on disk")

    input_cfg = config.load_contract("input")
    npz_paths = sorted(samples_dir.glob("*.npz"), key=lambda p: int(p.stem))

    mismatches = []
    bbox_mismatches = []
    padding_mismatches = []
    n_checked = 0

    for npz_path in npz_paths:
        with np.load(npz_path) as data:
            image = data["image_normalized"]
            mask = data["tumor_mask"]
            label = int(data["label"])
            patient_id = str(data["patient_id"])
            sample_id = str(data["sample_id"])

        phase1 = _phase1_roi(image, mask)
        geometry, btdl_crop = compute_roi(image, mask, input_cfg)
        n_checked += 1

        if not np.array_equal(phase1.roi_image, btdl_crop):
            mismatches.append(sample_id)
        if phase1.tight_bbox != geometry.tight_bbox:
            bbox_mismatches.append(sample_id)
        if phase1.square_bbox != geometry.crop_box:
            padding_mismatches.append(sample_id)

    assert n_checked == contract["expected"]["n_samples"]
    assert not mismatches, f"{len(mismatches)} crop mismatches: {mismatches[:20]}"
    assert not bbox_mismatches, f"{len(bbox_mismatches)} tight-bbox mismatches: {bbox_mismatches[:20]}"
    assert not padding_mismatches, (
        f"{len(padding_mismatches)} padded-square-bbox mismatches: {padding_mismatches[:20]}"
    )
