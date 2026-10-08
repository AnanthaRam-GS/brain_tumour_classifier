"""Reload equivalence (C8): a freshly-reloaded best.pt's val probabilities
must match the val_predictions.csv saved at the best epoch during
training, within reload_equivalence_atol. Shared by cli/evaluate.py and
cli/final_test.py's --rehearsal mode."""

import pandas as pd

from btdl.contracts import CLASS_NAMES_BY_INDEX


class ReloadEquivalenceError(ValueError):
    """Raised when a sample is missing from the saved predictions, or the diff exceeds atol."""


def compute_reload_max_abs_diff(predictions, val_predictions_path) -> float:
    """predictions: a Predictions object from a fresh predict() call on val.
    val_predictions_path: the val_predictions.csv written during training."""

    saved = pd.read_csv(val_predictions_path, dtype={"sample_id": str})
    saved_by_id = {row.sample_id: row for row in saved.itertuples(index=False)}

    sample_ids = list(predictions.sample_ids)
    missing = [sample_id for sample_id in sample_ids if sample_id not in saved_by_id]
    if missing:
        raise ReloadEquivalenceError(
            f"{len(missing)} current sample_ids missing from {val_predictions_path}: {missing[:5]}"
        )

    max_abs_diff = 0.0
    for i, sample_id in enumerate(sample_ids):
        saved_row = saved_by_id[sample_id]
        for class_idx, name in enumerate(CLASS_NAMES_BY_INDEX):
            saved_prob = getattr(saved_row, f"prob_{name}")
            current_prob = float(predictions.probs[i, class_idx])
            max_abs_diff = max(max_abs_diff, abs(saved_prob - current_prob))

    return float(max_abs_diff)
