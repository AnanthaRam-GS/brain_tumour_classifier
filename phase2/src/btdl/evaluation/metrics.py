"""Minimal per-epoch classification metrics.

Kept deliberately small for the training loop; a later prompt extends this
module with the full evaluation/report suite rather than adding a parallel
metrics module.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from btdl.contracts import CLASS_NAMES_BY_INDEX

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

    # Validated above to sum to 1 within 1e-5, but sklearn's log_loss applies
    # a stricter internal check and warns on float32-precision row sums; a
    # defensive renormalization (for this call only) avoids the noise
    # without loosening the validation tolerance itself.
    normalized_probs = probs / probs.sum(axis=1, keepdims=True)

    return {
        "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_idx, y_pred_idx)),
        "macro_f1": float(
            f1_score(y_true_idx, y_pred_idx, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "log_loss": float(log_loss(y_true_idx, normalized_probs, labels=list(LABELS))),
    }


def full_metrics(y_true_idx, probs) -> dict:
    """The full evaluation metric suite over internal class indices {0,1,2}.

    Predicted class = argmax(probs) per row; ties are broken by the lowest
    class index (numpy's documented argmax behavior -- the first occurrence
    of the maximum wins).

    Reuses epoch_metrics' validation (_validate) rather than reimplementing it.
    """

    y_true_idx, probs = _validate(y_true_idx, probs)
    y_pred_idx = np.argmax(probs, axis=1)
    normalized_probs = probs / probs.sum(axis=1, keepdims=True)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=list(LABELS), average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=list(LABELS), average="weighted", zero_division=0
    )
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=list(LABELS), average=None, zero_division=0
    )

    per_class = {}
    for index, name in enumerate(CLASS_NAMES_BY_INDEX):
        per_class[name] = {
            "precision": float(class_precision[index]),
            "recall": float(class_recall[index]),
            "f1": float(class_f1[index]),
            "support": int(class_support[index]),
        }

    confusion = sk_confusion_matrix(y_true_idx, y_pred_idx, labels=list(LABELS)).astype(int).tolist()

    roc_auc_per_class = {}
    roc_curves = {}
    for index, name in enumerate(CLASS_NAMES_BY_INDEX):
        y_true_binary = (y_true_idx == index).astype(int)
        if len(np.unique(y_true_binary)) < 2:
            # AUC/ROC undefined when a class is entirely absent or entirely
            # present in y_true -- reported as None/empty rather than guessed.
            roc_auc_per_class[name] = None
            roc_curves[name] = {"fpr": [], "tpr": [], "thresholds": []}
            continue
        roc_auc_per_class[name] = float(roc_auc_score(y_true_binary, normalized_probs[:, index]))
        fpr, tpr, thresholds = roc_curve(y_true_binary, normalized_probs[:, index])
        roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}

    try:
        macro_ovr = float(
            roc_auc_score(y_true_idx, normalized_probs, labels=list(LABELS), multi_class="ovr", average="macro")
        )
        weighted_ovr = float(
            roc_auc_score(
                y_true_idx, normalized_probs, labels=list(LABELS), multi_class="ovr", average="weighted"
            )
        )
    except ValueError:
        macro_ovr = None
        weighted_ovr = None
    else:
        # sklearn warns (rather than raising) when a class is absent and
        # folds it into the average as NaN; report that as undefined (None),
        # consistent with how an absent class's per-class AUC is reported.
        if not np.isfinite(macro_ovr):
            macro_ovr = None
        if not np.isfinite(weighted_ovr):
            weighted_ovr = None

    return {
        "n": int(len(y_true_idx)),
        "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_idx, y_pred_idx)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "roc_auc": {
            "macro_ovr": macro_ovr,
            "weighted_ovr": weighted_ovr,
            "per_class": roc_auc_per_class,
        },
        "log_loss": float(log_loss(y_true_idx, normalized_probs, labels=list(LABELS))),
        "roc_curves": roc_curves,
    }
