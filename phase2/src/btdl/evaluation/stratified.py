"""Size-stratified evaluation (small/medium/large crop_side tertiles).

Tertile edges always come from the frozen, train-only
phase2/artifacts/contract/size_tertiles.json (src/btdl/cli/size_tertiles.py)
-- never recomputed from val/test geometry.
"""

import numpy as np

from btdl.contracts import CLASS_NAMES_BY_INDEX
from btdl.evaluation.metrics import full_metrics

TERTILE_NAMES = ("small", "medium", "large")


def classify_tertile(crop_side, edges) -> str:
    low, high = edges
    if crop_side <= low:
        return "small"
    if crop_side <= high:
        return "medium"
    return "large"


def stratified_report(predictions, geometry, tertiles: dict) -> dict:
    """predictions: a Predictions-like object (sample_ids, y_true_idx, probs).
    geometry: a DataFrame with sample_id, crop_side columns.
    tertiles: {"edges": [low, high], ...} as written by cli/size_tertiles.py.
    """

    edges = tertiles["edges"]
    geometry_by_id = dict(zip(geometry["sample_id"].astype(str), geometry["crop_side"]))

    sample_ids = list(predictions.sample_ids)
    crop_sides = np.array([geometry_by_id[sample_id] for sample_id in sample_ids])
    tertile_labels = np.array([classify_tertile(cs, edges) for cs in crop_sides])

    tertile_report = {}
    for tertile in TERTILE_NAMES:
        mask = tertile_labels == tertile
        n = int(mask.sum())
        if n == 0:
            tertile_report[tertile] = {"n": 0, "accuracy": None, "macro_f1": None, "per_class_recall": {}}
            continue
        sub_metrics = full_metrics(predictions.y_true_idx[mask], predictions.probs[mask])
        tertile_report[tertile] = {
            "n": n,
            "accuracy": sub_metrics["accuracy"],
            "macro_f1": sub_metrics["macro_f1"],
            "per_class_recall": {name: stats["recall"] for name, stats in sub_metrics["per_class"].items()},
        }

    class_tertile_counts = {name: {tertile: 0 for tertile in TERTILE_NAMES} for name in CLASS_NAMES_BY_INDEX}
    for class_idx, class_name in enumerate(CLASS_NAMES_BY_INDEX):
        for tertile in TERTILE_NAMES:
            mask = (tertile_labels == tertile) & (predictions.y_true_idx == class_idx)
            class_tertile_counts[class_name][tertile] = int(mask.sum())

    return {"tertiles": tertile_report, "class_tertile_counts": class_tertile_counts}
