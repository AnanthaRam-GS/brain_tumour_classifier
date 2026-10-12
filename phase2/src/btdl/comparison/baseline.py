"""Read-only loader for the Phase 1 baseline referenced by study.yaml's
baseline_reference (glcm_xgboost_v2's test predictions).

Only ever read in "test" mode -- the baseline is itself a frozen Phase 1
test-set evaluation, so there is no val-preview equivalent of it."""

from pathlib import Path

import numpy as np
import pandas as pd

from btdl import config
from btdl.contracts import CLASS_NAMES_BY_INDEX

BASELINE_NAME = "glcm_xgboost_v2"


class BaselineAlignmentError(ValueError):
    """Raised when the baseline's predicted sample_ids don't exactly match
    the Phase 2 test split."""


def load_baseline_test_predictions(expected_sample_ids, *, repo_root=None) -> dict:
    """Loads <baseline_reference's directory>/test_predictions.csv, checks
    its sample_id set is EXACTLY (not a subset/superset of) expected_sample_ids,
    and returns {name, y_true_idx, probs, sample_ids, patient_ids}."""

    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    study = config.load_study()
    baseline_summary_relpath = Path(study["baseline_reference"])
    predictions_path = repo_root / baseline_summary_relpath.parent / "test_predictions.csv"
    if not predictions_path.is_file():
        raise BaselineAlignmentError(f"baseline predictions not found: {predictions_path}")

    frame = pd.read_csv(predictions_path, dtype={"sample_id": str, "patient_id": str})

    actual_ids = set(frame["sample_id"])
    expected_ids = set(expected_sample_ids)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids, key=lambda s: (len(s), s))[:5]
        extra = sorted(actual_ids - expected_ids, key=lambda s: (len(s), s))[:5]
        raise BaselineAlignmentError(
            f"baseline ({predictions_path}) sample_id set ({len(actual_ids)}) does not exactly "
            f"match the Phase 2 test split ({len(expected_ids)}); e.g. missing={missing} extra={extra}"
        )

    prob_columns = [f"prob_{name}" for name in CLASS_NAMES_BY_INDEX]
    missing_columns = [col for col in prob_columns + ["true_label", "sample_id", "patient_id"] if col not in frame]
    if missing_columns:
        raise BaselineAlignmentError(f"baseline predictions missing columns: {missing_columns}")

    label_to_index = {1: 0, 2: 1, 3: 2}
    y_true_idx = frame["true_label"].map(label_to_index).to_numpy()
    probs = frame[prob_columns].to_numpy(dtype=np.float64)

    return {
        "name": BASELINE_NAME,
        "y_true_idx": y_true_idx,
        "probs": probs,
        "sample_ids": tuple(frame["sample_id"]),
        "patient_ids": tuple(frame["patient_id"]),
    }
