"""Read-only reporting helpers for the faculty-demo walkthrough notebooks
(phase2/notebooks/make_walkthrough.py).

Every function here only READS existing artifacts (contracts, the split,
the ROI cache, run directories, exported results) and formats them for
display -- it never trains a model, runs inference, computes a metric
from scratch, or constructs a dataset for the "test" split. This module
never calls RoiDataset at all (it reads the ROI cache and predictions.csv
directly); tests/test_walkthrough.py's AST guard statically forbids this
module from ever calling RoiDataset with split="test" or allow_test=True,
should one ever be added here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from btdl import config
from btdl.cli.register_frozen import FROZEN_REGISTRATION_DIR_RELPATH
from btdl.cli.unblinding_status import compute_status
from btdl.comparison.baseline import load_baseline_test_predictions
from btdl.comparison.loaders import ComparisonInputError, load_test_model
from btdl.comparison.stats import (
    paired_multi_seed_patient_bootstrap,
    seed_aggregate_calibration,
    seed_aggregate_headline,
    seed_aggregate_per_class,
    seed_aggregate_stratified,
)
from btdl.contracts import CLASS_NAMES_BY_INDEX, label_to_index
from btdl.data.roi_cache import open_roi_cache
from btdl.data.split import SplitIntegrityError, load_split, validate_split
from btdl.evaluation.metrics import full_metrics
from btdl.models.registry import build_model, get_spec
from btdl.preprocessing.model_input import to_model_input
from btdl.preprocessing.resize import resize_roi
from btdl.preprocessing.roi import compute_roi
from btdl.runs import run_dir_for

BASELINE_COMPARISON_METRICS = ("macro_f1", "balanced_accuracy", "meningioma_f1", "macro_ovr_auc")

QA_DIR_RELPATH = "phase2/artifacts/qa"
RESULTS_DIR_RELPATH = "phase2/artifacts/results"
SELECTION_DIR_RELPATH = "phase2/artifacts/selection"
TEST_ACCESS_LOG_RELPATH = "phase2/artifacts/test_access_log.csv"


class WalkthroughError(ValueError):
    """Raised when a walkthrough helper cannot find the artifact it needs."""


# ---- shared small readers ---------------------------------------------------------------


def _read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _manifest_frame(repo_root):
    data_contract = config.load_contract("data")
    manifest_path = repo_root / data_contract["manifest"]
    return pd.read_csv(manifest_path, dtype={"sample_id": str, "patient_id": str})


def _selection(model_name, repo_root):
    path = repo_root / SELECTION_DIR_RELPATH / f"{model_name}.json"
    if not path.is_file():
        raise WalkthroughError(f"no selection file at {path} -- run cli/select_lr.py first")
    return _read_json(path)


def _final_seeds():
    return sorted(config.load_contract("training")["final_seeds"])


def _final_run_dir(model_name, seed, repo_root):
    selection = _selection(model_name, repo_root)
    return run_dir_for(model_name, selection["selected_lr"], seed, smoke=False)


# ---- 1. dataset_overview ------------------------------------------------------------------


def dataset_overview() -> dict:
    """Split/patient/class tables and the patient-level overlap check."""

    repo_root = config.repo_root()
    manifest = _manifest_frame(repo_root)
    split_index = load_split()

    overlap_ok = True
    overlap_detail = "no patient overlap between train/val/test"
    try:
        validate_split(split_index, manifest)
    except SplitIntegrityError as exc:
        overlap_ok = False
        overlap_detail = str(exc)

    rows = [{"split": sample["split"], "patient_id": sample["patient_id"]} for _, sample in split_index.items()]
    split_patient_frame = pd.DataFrame(rows)

    split_table = (
        split_patient_frame.groupby("split")
        .agg(n_samples=("patient_id", "size"), n_patients=("patient_id", "nunique"))
        .reindex(["train", "val", "test"])
    )

    class_table = manifest.pivot_table(index="split", columns="label", aggfunc="size", fill_value=0).reindex(
        ["train", "val", "test"]
    )

    return {
        "split_table": split_table,
        "class_table": class_table,
        "n_samples": len(manifest),
        "n_patients": manifest["patient_id"].nunique(),
        "patient_overlap_ok": overlap_ok,
        "patient_overlap_detail": overlap_detail,
    }


# ---- 2. show_roi_pipeline -------------------------------------------------------------------


def show_roi_pipeline(sample_id: str) -> dict:
    """Raw normalised image + mask (if the NPZ is locally cached) and the
    cached 224 ROI + 3-channel tensor stats for one sample. Never touches
    the test split: `sample_id` must belong to train or val."""

    repo_root = config.repo_root()
    split_index = load_split()
    sample_id = str(sample_id)
    if sample_id not in split_index:
        raise WalkthroughError(f"{sample_id!r} is not in the split index")
    entry = split_index[sample_id]
    if entry["split"] == "test":
        raise WalkthroughError(
            f"{sample_id!r} belongs to the test split -- show_roi_pipeline only displays train/val samples"
        )

    data_contract = config.load_contract("data")
    input_cfg = config.load_contract("input")
    samples_dir = config.resolve_data_path(data_contract["samples_dir"])
    npz_path = samples_dir / f"{sample_id}.npz"

    result = {
        "sample_id": sample_id,
        "split": entry["split"],
        "patient_id": entry["patient_id"],
        "label": entry["label"],
    }

    if npz_path.is_file():
        with np.load(npz_path) as handle:
            image = np.asarray(handle["image_normalized"], dtype=np.float32)
            mask = np.asarray(handle["tumor_mask"])
        geometry, crop = compute_roi(image, mask, input_cfg)
        resized = resize_roi(crop, input_cfg)
        result.update(
            {
                "cache_only_checkout": False,
                "raw_image": image,
                "raw_mask": mask,
                "roi_geometry": geometry,
                "recomputed_roi_224": resized,
            }
        )
    else:
        result["cache_only_checkout"] = True
        result["note"] = "raw NPZ cache not present locally -- showing the cached ROI only"

    cache = open_roi_cache()
    roi224 = np.asarray(cache[sample_id], dtype=np.float32)
    tensor = to_model_input(roi224, input_cfg)

    result["cached_roi_224"] = roi224
    result["model_input_tensor_shape"] = tuple(tensor.shape)
    result["model_input_tensor_stats"] = {
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }
    return result


# ---- 3. show_qa_sheets -----------------------------------------------------------------------


def show_qa_sheets() -> dict:
    """Paths to the already-generated QA contact sheets (never regenerates them)."""

    repo_root = config.repo_root()
    qa_dir = repo_root / QA_DIR_RELPATH
    sheets = {}
    for name in ("augmentation_sheet.png", "roi_contact_sheet.png"):
        path = qa_dir / name
        sheets[name] = path if path.is_file() else None
    return sheets


# ---- 4. model_overview -----------------------------------------------------------------------


def model_overview(model: str, *, instantiate: bool = False) -> dict:
    """Spec, parameter counts, module tree, pretrained-weights id.

    instantiate=False (default) reports spec fields only, without building
    the model (so it never needs network access for pretrained weights).
    Pass instantiate=True to also report parameter counts and the module
    tree (builds the model with pretrained=False to avoid any download)."""

    spec = get_spec(model)
    result = {
        "name": spec.name,
        "version": spec.version,
        "weights_id": spec.weights_id,
        "reference_only": spec.reference_only,
        "description": spec.description,
    }
    if instantiate:
        built = build_model(model, pretrained=False)
        result["total_parameters"] = sum(p.numel() for p in built.parameters())
        result["trainable_parameters"] = sum(p.numel() for p in built.parameters() if p.requires_grad)
        result["module_tree"] = str(built)
    return result


# ---- 5. protocol_overview --------------------------------------------------------------------


def protocol_overview(model: str) -> dict:
    """Grid table + selection JSON, seed table (mean +/- sample std), and
    the training-history frame for every final-seed run."""

    repo_root = config.repo_root()
    selection = _selection(model, repo_root)
    grid_frame = pd.DataFrame(selection["candidates"]).sort_values("lr").reset_index(drop=True)

    seeds = _final_seeds()
    seed_rows = []
    histories = {}
    for seed in seeds:
        run_dir = _final_run_dir(model, seed, repo_root)
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = _read_json(metadata_path)
        seed_rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "best_epoch": metadata.get("best_epoch"),
                "best_val_macro_f1": metadata.get("best_val_macro_f1"),
                "best_val_loss": metadata.get("best_val_loss"),
            }
        )
        history_path = run_dir / "history.csv"
        if history_path.is_file():
            histories[seed] = pd.read_csv(history_path)

    seed_frame = pd.DataFrame(seed_rows)
    summary = {}
    if not seed_frame.empty:
        for column in ("best_val_macro_f1", "best_val_loss"):
            summary[column] = {"mean": seed_frame[column].mean(), "std": seed_frame[column].std(ddof=1)}

    return {
        "selected_lr": selection["selected_lr"],
        "grid_table": grid_frame,
        "seed_table": seed_frame,
        "seed_summary": summary,
        "training_curves": histories,
    }


# ---- 6. split_results -------------------------------------------------------------------------


def _eval_dir_for(model, split, seed, repo_root):
    if split in ("val", "rehearsal_val"):
        run_dir = _final_run_dir(model, seed, repo_root)
        return run_dir / "eval" / split
    if split == "test":
        return repo_root / RESULTS_DIR_RELPATH / model / f"seed{seed}" / "eval" / "test"
    raise WalkthroughError(f"split must be one of val/rehearsal_val/test, got {split!r}")


def split_results(model: str, split: str) -> dict:
    """Per-seed and seed-aggregated metrics/predictions for one split,
    reading only already-written eval artifacts (never runs inference)."""

    repo_root = config.repo_root()
    seeds = _final_seeds()

    per_seed = {}
    missing_seeds = []
    for seed in seeds:
        eval_dir = _eval_dir_for(model, split, seed, repo_root)
        metrics_path = eval_dir / "metrics.json"
        if not metrics_path.is_file():
            missing_seeds.append(seed)
            continue
        per_seed[seed] = {
            "metrics": _read_json(metrics_path),
            "predictions": pd.read_csv(eval_dir / "predictions.csv"),
            "bootstrap_ci": _read_json(eval_dir / "bootstrap_ci.json"),
            "stratified": _read_json(eval_dir / "stratified.json"),
            "roc_curves": _read_json(eval_dir / "roc_curves.json"),
            "confusion_matrix": pd.read_csv(eval_dir / "confusion_matrix.csv", index_col=0),
        }

    if missing_seeds and not per_seed:
        return {
            "available": False,
            "missing_seeds": missing_seeds,
            "note": f"no {split!r} results available yet for {model!r}",
        }

    seeds_metrics = [per_seed[seed]["metrics"] for seed in per_seed]
    seeds_stratified = [per_seed[seed]["stratified"] for seed in per_seed]
    probs_by_seed = {
        seed: per_seed[seed]["predictions"][[f"prob_{name}" for name in CLASS_NAMES_BY_INDEX]].to_numpy()
        for seed in per_seed
    }

    aggregates = {
        "headline": seed_aggregate_headline(seeds_metrics),
        "per_class": seed_aggregate_per_class(seeds_metrics),
        "stratified": seed_aggregate_stratified(seeds_stratified),
    }

    calibration_inputs = []
    for seed, data in per_seed.items():
        y_true_idx = data["predictions"]["true_label"].map(label_to_index)
        calibration_inputs.append({"y_true_idx": y_true_idx.to_numpy(), "probs": probs_by_seed[seed]})
    if calibration_inputs:
        aggregates["calibration"] = seed_aggregate_calibration(calibration_inputs)

    return {
        "available": True,
        "missing_seeds": missing_seeds,
        "per_seed": per_seed,
        "aggregates": aggregates,
    }


# ---- 7. baseline_comparison -------------------------------------------------------------------


def baseline_comparison(model: str) -> dict:
    """Test-mode-only comparison against Phase 1's GLCM v2 baseline,
    computed directly from THIS model's own exported test-mode seeds
    (phase2/artifacts/results/<model>/seed<seed>/) and the baseline's test
    predictions -- no dependency on the cross-model
    phase2/artifacts/comparison/comparison.json (which requires every
    study model to be registered+tested first, via
    cli.compare.py --mode test / unblinding_status). Reuses the same
    comparison primitives cli/compare.py itself uses (btdl.comparison.
    baseline/loaders/stats), just scoped to one model instead of the
    whole study.

    Returns a headline table (this model's seed mean +/- std vs. the
    baseline's point value) and paired-difference bootstrap CIs (this
    model's seed-averaged metric minus the baseline's, on identical
    patient resamples), for macro_f1, balanced_accuracy, meningioma_f1,
    and macro_ovr_auc."""

    repo_root = config.repo_root()
    try:
        seeds_data = load_test_model(model, repo_root=repo_root)
    except ComparisonInputError as exc:
        return {"available": False, "note": str(exc)}

    seeds_metrics = [d["metrics"] for d in seeds_data]
    probs_by_seed = {d["seed"]: d["probs"] for d in seeds_data}
    model_headline = seed_aggregate_headline(seeds_metrics)
    model_per_class = seed_aggregate_per_class(seeds_metrics)

    expected_sample_ids = set(seeds_data[0]["sample_ids"])
    baseline_raw = load_baseline_test_predictions(expected_sample_ids, repo_root=repo_root)
    baseline_metrics = full_metrics(baseline_raw["y_true_idx"], baseline_raw["probs"])
    baseline_headline = seed_aggregate_headline([baseline_metrics])
    baseline_per_class = seed_aggregate_per_class([baseline_metrics])

    evaluation_cfg = config.load_contract("evaluation")
    paired = paired_multi_seed_patient_bootstrap(
        seeds_data[0]["y_true_idx"],
        probs_by_seed,
        {"baseline": baseline_raw["probs"]},
        seeds_data[0]["patient_ids"],
        n_resamples=evaluation_cfg["bootstrap"]["n_resamples"],
        seed=evaluation_cfg["bootstrap"]["seed"],
        alpha=evaluation_cfg["bootstrap"]["alpha"],
    )

    def _stats_for(name, headline, per_class):
        if name == "meningioma_f1":
            return per_class["meningioma"]["f1"]
        if name == "macro_ovr_auc":
            return headline["macro_auc"]
        return headline[name]

    headline_table = {}
    for name in BASELINE_COMPARISON_METRICS:
        model_stats = _stats_for(name, model_headline, model_per_class)
        baseline_stats = _stats_for(name, baseline_headline, baseline_per_class)
        headline_table[name] = {
            "model_mean": model_stats["mean"],
            "model_std": model_stats["std"],
            "baseline": baseline_stats["mean"],
        }

    return {
        "available": True,
        "model": model,
        "baseline_name": baseline_raw["name"],
        "headline": headline_table,
        "paired_vs_baseline": {name: paired[name] for name in BASELINE_COMPARISON_METRICS},
    }


# ---- 8. example_predictions -------------------------------------------------------------------


def example_predictions(model: str, split: str, n: int = 6) -> dict:
    """n example predictions (images + probabilities) from an already
    written predictions.csv, including at least one misclassified example
    if any exist. Never runs new inference."""

    repo_root = config.repo_root()
    seeds = _final_seeds()

    predictions_path = None
    used_seed = None
    for seed in seeds:
        eval_dir = _eval_dir_for(model, split, seed, repo_root)
        candidate = eval_dir / "predictions.csv"
        if candidate.is_file():
            predictions_path = candidate
            used_seed = seed
            break
    if predictions_path is None:
        return {"available": False, "note": f"no {split!r} predictions available yet for {model!r}"}

    predictions = pd.read_csv(predictions_path, dtype={"sample_id": str})
    correct = predictions[predictions["correct"]]
    incorrect = predictions[~predictions["correct"]]

    chosen_rows = []
    if not incorrect.empty:
        chosen_rows.append(incorrect.iloc[0])
    remaining = max(n - len(chosen_rows), 0)
    pool = correct if not correct.empty else predictions
    chosen_rows.extend(row for _, row in pool.head(remaining).iterrows())
    chosen = pd.DataFrame(chosen_rows).head(n).reset_index(drop=True)

    cache = open_roi_cache()
    examples = []
    for _, row in chosen.iterrows():
        sample_id = row["sample_id"]
        image = np.asarray(cache[sample_id], dtype=np.float32) if sample_id in cache.sample_id_to_row else None
        examples.append({"row": row.to_dict(), "image": image})

    return {"available": True, "seed": used_seed, "examples": examples}


# ---- 9. test_access_status --------------------------------------------------------------------


def test_access_status(model: str) -> dict:
    """Per-seed: has final_test logged this run, when, and has it been
    exported -- reading only the tracked log/results artifacts."""

    repo_root = config.repo_root()
    seeds = _final_seeds()

    log_path = repo_root / TEST_ACCESS_LOG_RELPATH
    log_rows = []
    if log_path.is_file():
        log_rows = pd.read_csv(log_path).to_dict("records") if log_path.stat().st_size else []

    frozen_path = repo_root / FROZEN_REGISTRATION_DIR_RELPATH / f"{model}.json"
    registration = _read_json(frozen_path) if frozen_path.is_file() else None

    per_seed = {}
    for seed in seeds:
        run_dir = None
        if registration is not None:
            entry = next((e for e in registration.get("seeds", []) if e.get("seed") == seed), None)
            run_dir = entry.get("run_dir") if entry else None
        logged_row = next(
            (row for row in log_rows if row.get("model_name") == model and row.get("seed") == seed), None
        )
        exported = (repo_root / RESULTS_DIR_RELPATH / model / f"seed{seed}").is_dir()
        per_seed[seed] = {
            "registered": run_dir is not None,
            "run_dir": run_dir,
            "logged": logged_row is not None,
            "logged_at": logged_row.get("utc_timestamp") if logged_row else None,
            "exported": exported,
        }

    overall_status = compute_status()
    model_row = next((row for row in overall_status["models"] if row["model_name"] == model), None)

    return {"per_seed": per_seed, "study_wide_ready": overall_status["ready"], "model_status_row": model_row}
