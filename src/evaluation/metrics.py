"""Shared multiclass metrics for Phase 1 classical ML experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

CLASS_ORDER = [1, 2, 3]
CLASS_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


class EvaluationError(ValueError):
    """Raised when evaluation inputs or outputs violate the shared contract."""


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute shared multiclass metrics with fixed class order [1, 2, 3]."""

    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape:
        raise EvaluationError(f"y_true and y_pred shapes differ: {true.shape} != {pred.shape}")
    if not set(np.unique(true)).issubset(CLASS_ORDER):
        raise EvaluationError("y_true contains labels outside [1, 2, 3]")
    if not set(np.unique(pred)).issubset(CLASS_ORDER):
        raise EvaluationError("y_pred contains labels outside [1, 2, 3]")

    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    roc_auc = {
        "macro_ovr": None,
        "weighted_ovr": None,
        "available": False,
        "reason": "score values unavailable",
    }
    if y_score is not None:
        scores = np.asarray(y_score, dtype=float)
        if scores.shape != (len(true), len(CLASS_ORDER)):
            raise EvaluationError(
                f"y_score must have shape ({len(true)}, {len(CLASS_ORDER)}), got {scores.shape}"
            )
        if not np.isfinite(scores).all():
            raise EvaluationError("y_score contains NaN or infinite values")
        try:
            binarized = label_binarize(true, classes=CLASS_ORDER)
            roc_auc = {
                "macro_ovr": float(
                    roc_auc_score(binarized, scores, average="macro", multi_class="ovr")
                ),
                "weighted_ovr": float(
                    roc_auc_score(binarized, scores, average="weighted", multi_class="ovr")
                ),
                "available": True,
                "reason": "",
            }
        except ValueError as exc:
            roc_auc["reason"] = str(exc)

    return {
        "metrics": {
            "accuracy": float(accuracy_score(true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            "macro_precision": float(
                precision_score(true, pred, labels=CLASS_ORDER, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(true, pred, labels=CLASS_ORDER, average="macro", zero_division=0)
            ),
            "macro_f1": float(f1_score(true, pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
            "weighted_precision": float(
                precision_score(true, pred, labels=CLASS_ORDER, average="weighted", zero_division=0)
            ),
            "weighted_recall": float(
                recall_score(true, pred, labels=CLASS_ORDER, average="weighted", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(true, pred, labels=CLASS_ORDER, average="weighted", zero_division=0)
            ),
        },
        "per_class_metrics": {
            CLASS_NAMES[label]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASS_ORDER)
        },
        "confusion_matrix": confusion_matrix(true, pred, labels=CLASS_ORDER).astype(int).tolist(),
        "roc_auc": roc_auc,
    }
