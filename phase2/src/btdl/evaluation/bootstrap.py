"""Patient-level bootstrap confidence intervals (docs/DECISIONS.md D14).

Resamples PATIENTS with replacement (taking all of a resampled patient's
slices), not individual samples -- slices from the same patient are
correlated, so resampling at the sample level would understate variance.
Uses its own np.random.Generator; never the global RNG.
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score

from btdl.contracts import CLASS_NAMES_BY_INDEX
from btdl.evaluation.metrics import LABELS, full_metrics

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "meningioma_f1",
    "glioma_f1",
    "pituitary_f1",
    "macro_ovr_auc",
    "log_loss",
)


def _extract_metric_values(metrics: dict) -> dict:
    values = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_ovr_auc": metrics["roc_auc"]["macro_ovr"],
        "log_loss": metrics["log_loss"],
    }
    for name, stats in metrics["per_class"].items():
        values[f"{name}_f1"] = stats["f1"]
    return values


def _fast_metric_values(y_true_idx, probs) -> dict:
    """Same metric set as _extract_metric_values(full_metrics(...)), computed
    directly via sklearn (skipping confusion_matrix/roc_curves/precision/
    recall, which bootstrap never needs) -- full_metrics() per resample made
    n_resamples=2000 bootstraps minutes slow; this keeps it to seconds."""

    y_pred_idx = np.argmax(probs, axis=1)
    normalized_probs = probs / probs.sum(axis=1, keepdims=True)

    values = {
        "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_idx, y_pred_idx)),
        "macro_f1": float(
            f1_score(y_true_idx, y_pred_idx, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_idx, y_pred_idx, labels=list(LABELS), average="weighted", zero_division=0)
        ),
        "log_loss": float(log_loss(y_true_idx, normalized_probs, labels=list(LABELS))),
    }
    per_class_f1 = f1_score(y_true_idx, y_pred_idx, labels=list(LABELS), average=None, zero_division=0)
    for index, name in enumerate(CLASS_NAMES_BY_INDEX):
        values[f"{name}_f1"] = float(per_class_f1[index])

    try:
        macro_ovr = float(
            roc_auc_score(y_true_idx, normalized_probs, labels=list(LABELS), multi_class="ovr", average="macro")
        )
        if not np.isfinite(macro_ovr):
            macro_ovr = None
    except ValueError:
        macro_ovr = None
    values["macro_ovr_auc"] = macro_ovr

    return values


def _patient_index_map(patient_ids: np.ndarray) -> dict:
    unique_patients = np.unique(patient_ids)
    return {patient: np.where(patient_ids == patient)[0] for patient in unique_patients}


def _resample_indices(unique_patients, patient_to_indices, rng) -> np.ndarray:
    sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    return np.concatenate([patient_to_indices[patient] for patient in sampled_patients])


def _percentile_ci(samples, alpha):
    low_pct = 100 * (alpha / 2)
    high_pct = 100 * (1 - alpha / 2)
    return float(np.percentile(samples, low_pct)), float(np.percentile(samples, high_pct))


def patient_bootstrap(
    y_true_idx, probs, patient_ids, *, n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Per-metric {point, ci_low, ci_high, n_valid}.

    A resample where a metric is undefined (e.g. a class absent, so its AUC
    is undefined) is skipped for THAT metric only; n_valid reports how many
    of n_resamples actually contributed to that metric's CI.
    """

    y_true_idx = np.asarray(y_true_idx)
    probs = np.asarray(probs)
    patient_ids = np.asarray(patient_ids)

    patient_to_indices = _patient_index_map(patient_ids)
    unique_patients = np.array(list(patient_to_indices.keys()))

    point_values = _extract_metric_values(full_metrics(y_true_idx, probs))

    rng = np.random.default_rng(seed)
    collected = {name: [] for name in point_values}

    for _ in range(n_resamples):
        idx = _resample_indices(unique_patients, patient_to_indices, rng)
        values = _fast_metric_values(y_true_idx[idx], probs[idx])
        for name, value in values.items():
            if value is not None and np.isfinite(value):
                collected[name].append(value)

    result = {}
    for name, point_value in point_values.items():
        samples = collected[name]
        n_valid = len(samples)
        if n_valid == 0:
            result[name] = {"point": point_value, "ci_low": None, "ci_high": None, "n_valid": 0}
            continue
        ci_low, ci_high = _percentile_ci(samples, alpha)
        result[name] = {"point": point_value, "ci_low": ci_low, "ci_high": ci_high, "n_valid": n_valid}

    return result


def paired_patient_bootstrap(
    y_true_idx,
    probs_a,
    probs_b,
    patient_ids,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Per-metric {point, ci_low, ci_high, n_valid, frac_a_greater} for metric(a) - metric(b),
    on IDENTICAL resamples for both a and b."""

    y_true_idx = np.asarray(y_true_idx)
    probs_a = np.asarray(probs_a)
    probs_b = np.asarray(probs_b)
    patient_ids = np.asarray(patient_ids)

    patient_to_indices = _patient_index_map(patient_ids)
    unique_patients = np.array(list(patient_to_indices.keys()))

    point_a = _extract_metric_values(full_metrics(y_true_idx, probs_a))
    point_b = _extract_metric_values(full_metrics(y_true_idx, probs_b))
    point_diff = {
        name: (point_a[name] - point_b[name])
        if (point_a[name] is not None and point_b[name] is not None)
        else None
        for name in point_a
    }

    rng = np.random.default_rng(seed)
    diffs = {name: [] for name in point_a}
    a_greater_count = {name: 0 for name in point_a}

    for _ in range(n_resamples):
        idx = _resample_indices(unique_patients, patient_to_indices, rng)
        values_a = _fast_metric_values(y_true_idx[idx], probs_a[idx])
        values_b = _fast_metric_values(y_true_idx[idx], probs_b[idx])
        for name in point_diff:
            va, vb = values_a[name], values_b[name]
            if va is None or vb is None or not np.isfinite(va) or not np.isfinite(vb):
                continue
            diffs[name].append(va - vb)
            if va > vb:
                a_greater_count[name] += 1

    result = {}
    for name, point_value in point_diff.items():
        samples = diffs[name]
        n_valid = len(samples)
        if n_valid == 0:
            result[name] = {
                "point": point_value,
                "ci_low": None,
                "ci_high": None,
                "n_valid": 0,
                "frac_a_greater": None,
            }
            continue
        ci_low, ci_high = _percentile_ci(samples, alpha)
        result[name] = {
            "point": point_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_valid": n_valid,
            "frac_a_greater": a_greater_count[name] / n_valid,
        }

    return result
