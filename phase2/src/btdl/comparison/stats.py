"""Seed aggregation, pooled/paired patient bootstrap, and calibration (D14, D19).

Reuses evaluation/bootstrap.py's patient-resampling primitives
(patient_index_map, resample_indices, percentile_ci, fast_metric_values,
extract_metric_values) rather than reimplementing patient-level resampling
-- the only new logic here is averaging a metric ACROSS SEEDS inside each
resample, which single-seed evaluation never needed.
"""

import numpy as np

from btdl.contracts import CLASS_NAMES_BY_INDEX
from btdl.evaluation.bootstrap import (
    extract_metric_values,
    fast_metric_values,
    patient_index_map,
    percentile_ci,
    resample_indices,
)
from btdl.evaluation.metrics import LABELS, full_metrics

HEADLINE_METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "log_loss",
)


def _nanmean(values):
    valid = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def _nanstd(values):
    valid = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else (0.0 if len(valid) == 1 else None)


# ---- seed aggregation (headline + per-class) -----------------------------------------------


def seed_aggregate_headline(seeds_metrics: list) -> dict:
    """seeds_metrics: list of full_metrics()-shaped dicts (one per seed).
    Returns {metric_name: {"mean": ..., "std": ..., "values": [...]}} for
    HEADLINE_METRIC_NAMES plus macro/weighted AUC."""

    def _get(metrics, name):
        if name == "macro_auc":
            return metrics["roc_auc"]["macro_ovr"]
        if name == "weighted_auc":
            return metrics["roc_auc"]["weighted_ovr"]
        return metrics[name]

    names = HEADLINE_METRIC_NAMES + ("macro_auc", "weighted_auc")
    result = {}
    for name in names:
        values = [_get(m, name) for m in seeds_metrics]
        result[name] = {"mean": _nanmean(values), "std": _nanstd(values), "values": values}
    return result


def seed_aggregate_per_class(seeds_metrics: list) -> dict:
    """{class_name: {metric: {"mean":..., "std":..., "values":[...]}}}."""

    result = {}
    for class_name in CLASS_NAMES_BY_INDEX:
        result[class_name] = {}
        for metric in ("precision", "recall", "f1"):
            values = [m["per_class"][class_name][metric] for m in seeds_metrics]
            result[class_name][metric] = {"mean": _nanmean(values), "std": _nanstd(values), "values": values}
    return result


def seed_aggregate_stratified(seeds_stratified: list) -> dict:
    """seeds_stratified: list of stratified_report()-shaped dicts (one per
    seed). {tertile: {class_name: {"mean":..., "std":..., "values":[...]}}}
    for per_class_recall, seed-averaged."""

    tertile_names = list(seeds_stratified[0]["tertiles"].keys())
    result = {}
    for tertile in tertile_names:
        result[tertile] = {}
        for class_name in CLASS_NAMES_BY_INDEX:
            values = [s["tertiles"][tertile]["per_class_recall"].get(class_name) for s in seeds_stratified]
            result[tertile][class_name] = {"mean": _nanmean(values), "std": _nanstd(values), "values": values}
    return result


# ---- multi-seed patient bootstrap (pooled + paired) ----------------------------------------


def multi_seed_patient_bootstrap(
    y_true_idx, probs_by_seed: dict, patient_ids, *, n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Per-metric {point, ci_low, ci_high, n_valid}: point/CI of the metric
    AVERAGED ACROSS SEEDS, computed on identical patient resamples for every
    seed. probs_by_seed: {seed: probs array}, all sharing y_true_idx/patient_ids
    (the same held-out set, predicted by each of the model's seed-trained
    checkpoints)."""

    y_true_idx = np.asarray(y_true_idx)
    patient_ids = np.asarray(patient_ids)
    probs_by_seed = {s: np.asarray(p) for s, p in probs_by_seed.items()}

    patient_to_indices = patient_index_map(patient_ids)
    unique_patients = np.array(list(patient_to_indices.keys()))

    point_per_seed = [extract_metric_values(full_metrics(y_true_idx, probs)) for probs in probs_by_seed.values()]
    metric_names = list(point_per_seed[0].keys())
    point_values = {name: _nanmean([p[name] for p in point_per_seed]) for name in metric_names}

    rng = np.random.default_rng(seed)
    collected = {name: [] for name in metric_names}

    for _ in range(n_resamples):
        idx = resample_indices(unique_patients, patient_to_indices, rng)
        per_seed_values = [fast_metric_values(y_true_idx[idx], probs[idx]) for probs in probs_by_seed.values()]
        for name in metric_names:
            seed_avg = _nanmean([v[name] for v in per_seed_values])
            if seed_avg is not None and np.isfinite(seed_avg):
                collected[name].append(seed_avg)

    result = {}
    for name, point_value in point_values.items():
        samples = collected[name]
        n_valid = len(samples)
        if n_valid == 0:
            result[name] = {"point": point_value, "ci_low": None, "ci_high": None, "n_valid": 0}
            continue
        ci_low, ci_high = percentile_ci(samples, alpha)
        result[name] = {"point": point_value, "ci_low": ci_low, "ci_high": ci_high, "n_valid": n_valid}
    return result


def paired_multi_seed_patient_bootstrap(
    y_true_idx,
    probs_by_seed_a: dict,
    probs_by_seed_b: dict,
    patient_ids,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Per-metric {point, ci_low, ci_high, n_valid, frac_a_greater} for
    seed-averaged-metric(A) - seed-averaged-metric(B), on IDENTICAL patient
    resamples for both A and B (and across every seed within each). A
    model compared against ITSELF (same probs_by_seed for both) gives
    point == 0.0 and ci_low == ci_high == 0.0 exactly."""

    y_true_idx = np.asarray(y_true_idx)
    patient_ids = np.asarray(patient_ids)
    probs_by_seed_a = {s: np.asarray(p) for s, p in probs_by_seed_a.items()}
    probs_by_seed_b = {s: np.asarray(p) for s, p in probs_by_seed_b.items()}

    patient_to_indices = patient_index_map(patient_ids)
    unique_patients = np.array(list(patient_to_indices.keys()))

    point_per_seed_a = [extract_metric_values(full_metrics(y_true_idx, p)) for p in probs_by_seed_a.values()]
    point_per_seed_b = [extract_metric_values(full_metrics(y_true_idx, p)) for p in probs_by_seed_b.values()]
    metric_names = list(point_per_seed_a[0].keys())
    point_a = {name: _nanmean([p[name] for p in point_per_seed_a]) for name in metric_names}
    point_b = {name: _nanmean([p[name] for p in point_per_seed_b]) for name in metric_names}
    point_diff = {
        name: (point_a[name] - point_b[name]) if (point_a[name] is not None and point_b[name] is not None) else None
        for name in metric_names
    }

    rng = np.random.default_rng(seed)
    diffs = {name: [] for name in metric_names}
    a_greater_count = {name: 0 for name in metric_names}

    for _ in range(n_resamples):
        idx = resample_indices(unique_patients, patient_to_indices, rng)
        values_a = [fast_metric_values(y_true_idx[idx], p[idx]) for p in probs_by_seed_a.values()]
        values_b = [fast_metric_values(y_true_idx[idx], p[idx]) for p in probs_by_seed_b.values()]
        for name in metric_names:
            avg_a = _nanmean([v[name] for v in values_a])
            avg_b = _nanmean([v[name] for v in values_b])
            if avg_a is None or avg_b is None:
                continue
            diffs[name].append(avg_a - avg_b)
            if avg_a > avg_b:
                a_greater_count[name] += 1

    result = {}
    for name, point_value in point_diff.items():
        samples = diffs[name]
        n_valid = len(samples)
        if n_valid == 0:
            result[name] = {"point": point_value, "ci_low": None, "ci_high": None, "n_valid": 0, "frac_a_greater": None}
            continue
        ci_low, ci_high = percentile_ci(samples, alpha)
        result[name] = {
            "point": point_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_valid": n_valid,
            "frac_a_greater": a_greater_count[name] / n_valid,
        }
    return result


# ---- calibration (D19): expected calibration error, top-label, 15 equal-width bins ---------


def expected_calibration_error(y_true_idx, probs, *, n_bins: int = 15) -> dict:
    """Top-label ECE: bins the PREDICTED class's confidence into n_bins
    equal-width bins over [0, 1]; ECE = sum_bins (|bin|/N) * |acc(bin) -
    conf(bin)|. Returns {"ece": float, "bins": [{"bin_low", "bin_high",
    "count", "accuracy", "confidence"}, ...]} -- bins' reliability-diagram
    data in bin order."""

    y_true_idx = np.asarray(y_true_idx)
    probs = np.asarray(probs)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true_idx).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidences, bin_edges[1:-1], right=True), 0, n_bins - 1)

    n = len(confidences)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {"bin_low": float(bin_edges[b]), "bin_high": float(bin_edges[b + 1]), "count": 0,
                 "accuracy": None, "confidence": None}
            )
            continue
        accuracy = float(correct[mask].mean())
        confidence = float(confidences[mask].mean())
        ece += (count / n) * abs(accuracy - confidence)
        bins.append(
            {"bin_low": float(bin_edges[b]), "bin_high": float(bin_edges[b + 1]), "count": count,
             "accuracy": accuracy, "confidence": confidence}
        )

    return {"ece": float(ece), "bins": bins}


def seed_aggregate_calibration(seeds_data: list, *, n_bins: int = 15) -> dict:
    """Per-seed ECE + mean/std across seeds. seeds_data: list of
    {"y_true_idx", "probs"}-shaped dicts (one per seed)."""

    per_seed = [expected_calibration_error(d["y_true_idx"], d["probs"], n_bins=n_bins) for d in seeds_data]
    ece_values = [r["ece"] for r in per_seed]
    return {
        "mean": _nanmean(ece_values),
        "std": _nanstd(ece_values),
        "values": ece_values,
        "per_seed_bins": [r["bins"] for r in per_seed],
    }
