"""Standardized evaluation artifacts.

write_evaluation() writes six files into out_dir: metrics.json,
predictions.csv, confusion_matrix.csv, roc_curves.json, bootstrap_ci.json,
stratified.json. validate_evaluation_dir() checks presence and basic schema
of every file -- a future comparison step only reads directories that pass
this.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from btdl import config
from btdl.contracts import CLASS_NAMES_BY_INDEX, index_to_label
from btdl.evaluation.bootstrap import patient_bootstrap
from btdl.evaluation.metrics import full_metrics
from btdl.evaluation.stratified import TERTILE_NAMES, classify_tertile, stratified_report

RESULTS_SCHEMA_VERSION = "1.0.0"

EVALUATION_FILES = (
    "metrics.json",
    "predictions.csv",
    "confusion_matrix.csv",
    "roc_curves.json",
    "bootstrap_ci.json",
    "stratified.json",
)

PREDICTIONS_COLUMNS = (
    ("sample_id", "patient_id", "true_label", "true_name", "pred_label", "pred_name")
    + tuple(f"prob_{name}" for name in CLASS_NAMES_BY_INDEX)
    + ("correct", "crop_side", "size_tertile")
)


class EvaluationValidationError(ValueError):
    """Raised when an evaluation directory is missing a file or fails a schema check."""


def _load_geometry_and_tertiles(repo_root):
    geometry = pd.read_csv(
        repo_root / "phase2" / "artifacts" / "contract" / "roi_geometry.csv", dtype={"sample_id": str}
    )
    with (repo_root / "phase2" / "artifacts" / "contract" / "size_tertiles.json").open() as handle:
        tertiles = json.load(handle)
    return geometry, tertiles


def _write_json(obj, path) -> None:
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_evaluation(out_dir, *, predictions, run_metadata: dict, split: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = config.repo_root()
    geometry, tertiles = _load_geometry_and_tertiles(repo_root)
    geometry_by_id = dict(zip(geometry["sample_id"], geometry["crop_side"]))

    full = full_metrics(predictions.y_true_idx, predictions.probs)
    roc_curves = full.pop("roc_curves")

    metrics_payload = dict(full)
    metrics_payload["split"] = split
    metrics_payload["results_schema_version"] = RESULTS_SCHEMA_VERSION
    metrics_payload["contract_conformant"] = run_metadata.get("contract_conformant")
    metrics_payload["run"] = {
        "model_name": run_metadata.get("model_name"),
        "model_version": run_metadata.get("model_version"),
        "seed": run_metadata.get("seed"),
        "lr": run_metadata.get("lr"),
        "best_epoch": run_metadata.get("best_epoch"),
        "git_commit": run_metadata.get("git_commit"),
        "contract_dir_sha256": run_metadata.get("contract_dir_sha256"),
    }
    _write_json(metrics_payload, out_dir / "metrics.json")

    predicted_idx = predictions.probs.argmax(axis=1)
    with open(out_dir / "predictions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTIONS_COLUMNS))
        writer.writeheader()
        for i, sample_id in enumerate(predictions.sample_ids):
            true_idx = int(predictions.y_true_idx[i])
            pred_idx = int(predicted_idx[i])
            crop_side = geometry_by_id.get(sample_id)
            size_tertile = classify_tertile(crop_side, tertiles["edges"]) if crop_side is not None else None
            row = {
                "sample_id": sample_id,
                "patient_id": predictions.patient_ids[i],
                "true_label": index_to_label(true_idx),
                "true_name": CLASS_NAMES_BY_INDEX[true_idx],
                "pred_label": index_to_label(pred_idx),
                "pred_name": CLASS_NAMES_BY_INDEX[pred_idx],
                "correct": true_idx == pred_idx,
                "crop_side": crop_side,
                "size_tertile": size_tertile,
            }
            for class_idx, name in enumerate(CLASS_NAMES_BY_INDEX):
                row[f"prob_{name}"] = float(predictions.probs[i, class_idx])
            writer.writerow(row)

    confusion_frame = pd.DataFrame(
        full["confusion_matrix"], index=list(CLASS_NAMES_BY_INDEX), columns=list(CLASS_NAMES_BY_INDEX)
    )
    confusion_frame.to_csv(out_dir / "confusion_matrix.csv")

    _write_json(roc_curves, out_dir / "roc_curves.json")

    bootstrap_result = patient_bootstrap(
        predictions.y_true_idx, predictions.probs, np.asarray(predictions.patient_ids)
    )
    _write_json(bootstrap_result, out_dir / "bootstrap_ci.json")

    stratified_result = stratified_report(predictions, geometry, tertiles)
    _write_json(stratified_result, out_dir / "stratified.json")

    return metrics_payload


def validate_evaluation_dir(path) -> None:
    path = Path(path)

    missing = [name for name in EVALUATION_FILES if not (path / name).is_file()]
    if missing:
        raise EvaluationValidationError(f"missing evaluation files in {path}: {missing}")

    with (path / "metrics.json").open() as handle:
        metrics = json.load(handle)
    required_metrics_keys = {
        "n",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "per_class",
        "confusion_matrix",
        "roc_auc",
        "log_loss",
        "split",
        "results_schema_version",
        "run",
        "contract_conformant",
    }
    missing_keys = required_metrics_keys - metrics.keys()
    if missing_keys:
        raise EvaluationValidationError(f"metrics.json missing keys: {sorted(missing_keys)}")

    predictions_frame = pd.read_csv(path / "predictions.csv")
    missing_columns = set(PREDICTIONS_COLUMNS) - set(predictions_frame.columns)
    if missing_columns:
        raise EvaluationValidationError(f"predictions.csv missing columns: {sorted(missing_columns)}")

    confusion_frame = pd.read_csv(path / "confusion_matrix.csv", index_col=0)
    if confusion_frame.shape != (3, 3):
        raise EvaluationValidationError(
            f"confusion_matrix.csv must be 3x3 (with header row and index), got {confusion_frame.shape}"
        )

    with (path / "roc_curves.json").open() as handle:
        roc_curves = json.load(handle)
    if set(roc_curves.keys()) != set(CLASS_NAMES_BY_INDEX):
        raise EvaluationValidationError(
            f"roc_curves.json keys must be exactly {set(CLASS_NAMES_BY_INDEX)}, got {set(roc_curves.keys())}"
        )
    for name, curve in roc_curves.items():
        if not {"fpr", "tpr", "thresholds"}.issubset(curve.keys()):
            raise EvaluationValidationError(f"roc_curves.json[{name!r}] missing fpr/tpr/thresholds")

    with (path / "bootstrap_ci.json").open() as handle:
        bootstrap = json.load(handle)
    if not bootstrap:
        raise EvaluationValidationError("bootstrap_ci.json is empty")
    for name, stats in bootstrap.items():
        if not {"point", "ci_low", "ci_high", "n_valid"}.issubset(stats.keys()):
            raise EvaluationValidationError(f"bootstrap_ci.json[{name!r}] missing required keys")

    with (path / "stratified.json").open() as handle:
        stratified = json.load(handle)
    if "tertiles" not in stratified or "class_tertile_counts" not in stratified:
        raise EvaluationValidationError("stratified.json missing 'tertiles' or 'class_tertile_counts'")
    if set(stratified["tertiles"].keys()) != set(TERTILE_NAMES):
        raise EvaluationValidationError(
            f"stratified.json tertiles must be exactly {set(TERTILE_NAMES)}, got {set(stratified['tertiles'].keys())}"
        )
