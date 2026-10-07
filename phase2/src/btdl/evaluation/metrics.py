"""Minimal per-epoch classification metrics.

Kept deliberately small for the training loop; a later prompt extends this
module with the full evaluation/report suite rather than adding a parallel
metrics module.
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

LABELS = (0, 1, 2)


class MetricsInputError(ValueError):
    """Raised when epoch_metrics' inputs fail validation."""


def _validate(y_true_idx, probs):
    y_true_idx = np.asarray(y_true_idx)
    probs = np.asarray(probs, dtype=np.float64)

    if probs.ndim != 2 or probs.shape[1] != 3:
        raise MetricsInputError(f"probs must have shape [N, 3], got {probs.shape}")
    if y_true_idx.shape[0] != probs.shape[0]:
        raise MetricsInputError(
            f"y_true_idx and probs have mismatched N: {y_true_idx.shape[0]} vs {probs.shape[0]}"
        )
    if not np.isfinite(probs).all():
        raise MetricsInputError("probs contains NaN or infinite values")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        max_abs_diff = float(np.max(np.abs(row_sums - 1.0)))
        raise MetricsInputError(f"probs rows must sum to 1 within 1e-5, max abs diff {max_abs_diff}")
    if not set(np.unique(y_true_idx).tolist()).issubset(set(LABELS)):
        raise MetricsInputError(f"y_true_idx values must be in {LABELS}, got {sorted(set(y_true_idx.tolist()))}")

    return y_true_idx, probs


def epoch_metrics(y_true_idx, probs) -> dict:
    """accuracy, balanced_accuracy, macro_f1, log_loss over internal class indices {0,1,2}."""

    y_true_idx, probs = _validate(y_true_idx, probs)
    y_pred_idx = np.argmax(probs, axis=1)

    return {
        "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_idx, y_pred_idx)),
        "macro_f1": float(
            f1_score(y_true_idx, y_pred_idx, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "log_loss": float(log_loss(y_true_idx, probs, labels=list(LABELS))),
    }
