"""Orchestrates the comparison pipeline: loads data, computes statistics,
writes tables/figures/comparison.json/RESULTS.md.

Deterministic by construction: bootstrap settings (n_resamples/seed/alpha)
come from the committed evaluation.yaml contract, every input is read in a
fixed (sorted) order, and every dict is JSON-dumped with sort_keys=True --
two runs over identical inputs produce byte-identical tables and
comparison.json.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from btdl import config
from btdl.cli.unblinding_status import compute_status
from btdl.comparison import baseline as baseline_mod
from btdl.comparison import loaders, plots, stats, tables
from btdl.evaluation.metrics import full_metrics

TEST_OUTPUT_DIR_RELPATH = "phase2/artifacts/comparison"
PREVIEW_OUTPUT_DIR_RELPATH = "phase2/runs/_preview/comparison"
PREVIEW_LABEL = plots.PREVIEW_LABEL

D15_LIMITATIONS = (
    "Metrics are computed on a single frozen test split; they describe this "
    "split, not a population-level guarantee.",
    "Three seeds per architecture bound, but do not eliminate, training-run "
    "variance (see each model's seed std in the headline table).",
    "Paired bootstrap differences are reported as confidence intervals, not "
    "hypothesis-test p-values; no multiple-comparison correction has been "
    "applied across the many pairs compared here -- treat any single "
    "pairwise CI narrowly, not as an isolated significance claim.",
    "The Phase 1 baseline (GLCM features + XGBoost) and the Phase 2 "
    "architectures were evaluated on the identical test split but via "
    "different pipelines (classical features vs. end-to-end CNNs); the "
    "comparison is valid but the two are not drop-in substitutes for each "
    "other's development process.",
)


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _predictions_hash(seed_data: dict) -> str:
    """A deterministic hash of exactly what was read for one seed (sample
    order, labels, probabilities) -- used in comparison.json's input_hashes,
    independent of file-system path or mtime."""

    digest = hashlib.sha256()
    digest.update(",".join(seed_data["sample_ids"]).encode("utf-8"))
    digest.update(np.asarray(seed_data["y_true_idx"]).tobytes())
    digest.update(np.ascontiguousarray(seed_data["probs"]).tobytes())
    return digest.hexdigest()


def _sum_confusion(seeds_data: list) -> list:
    total = np.zeros((3, 3), dtype=int)
    for seed_data in seeds_data:
        total += np.asarray(full_metrics(seed_data["y_true_idx"], seed_data["probs"])["confusion_matrix"])
    return total.tolist()


class CompareError(ValueError):
    """Raised when `compare` refuses to run (not READY, missing models, etc)."""


def build_comparison(mode: str, *, repo_root=None) -> dict:
    if mode not in ("test", "val-preview"):
        raise ValueError(f"mode must be 'test' or 'val-preview', got {mode!r}")

    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    study = config.load_study()
    evaluation_cfg = config.load_contract("evaluation")
    n_resamples = evaluation_cfg["bootstrap"]["n_resamples"]
    bootstrap_seed = evaluation_cfg["bootstrap"]["seed"]
    alpha = evaluation_cfg["bootstrap"]["alpha"]

    if mode == "test":
        status = compute_status()
        if not status["ready"]:
            not_ready = [row["model_name"] for row in status["models"] if not row["ready"]]
            raise CompareError(
                "refusing: unblinding_status is NOT READY -- not registered/ready: "
                f"{not_ready} (D18: test mode requires every study model to be frozen-registered)"
            )

    data_by_model, missing = loaders.load_study_models(mode, repo_root=repo_root)
    if mode == "test" and missing:
        raise CompareError(f"refusing: not all study models are exported for test mode: {missing}")

    # ---- per-model seed aggregation ----
    per_model = {}
    for model_name in sorted(data_by_model.keys()):
        seeds_data = data_by_model[model_name]
        seeds_metrics = [d["metrics"] for d in seeds_data]
        probs_by_seed = {d["seed"]: d["probs"] for d in seeds_data}
        per_model[model_name] = {
            "seeds": [d["seed"] for d in seeds_data],
            "headline": stats.seed_aggregate_headline(seeds_metrics),
            "per_class": stats.seed_aggregate_per_class(seeds_metrics),
            "stratified": stats.seed_aggregate_stratified([d["stratified"] for d in seeds_data]),
            "calibration": stats.seed_aggregate_calibration(seeds_data, n_bins=15),
            "pooled_bootstrap": stats.multi_seed_patient_bootstrap(
                seeds_data[0]["y_true_idx"],
                probs_by_seed,
                seeds_data[0]["patient_ids"],
                n_resamples=n_resamples,
                seed=bootstrap_seed,
                alpha=alpha,
            ),
            "confusion_matrix_sum": _sum_confusion(seeds_data),
        }

    # ---- Phase 1 baseline (test mode only) ----
    baseline_name = None
    if mode == "test" and data_by_model:
        any_seeds = next(iter(data_by_model.values()))
        expected_ids = set(any_seeds[0]["sample_ids"])
        baseline_raw = baseline_mod.load_baseline_test_predictions(expected_ids, repo_root=repo_root)
        baseline_name = baseline_raw["name"]
        baseline_metrics = full_metrics(baseline_raw["y_true_idx"], baseline_raw["probs"])
        per_model[baseline_name] = {
            "seeds": ["baseline"],
            "headline": stats.seed_aggregate_headline([baseline_metrics]),
            "per_class": stats.seed_aggregate_per_class([baseline_metrics]),
            "confusion_matrix_sum": baseline_metrics["confusion_matrix"],
            "is_baseline": True,
        }
    else:
        baseline_raw = None

    # ---- paired comparisons: every model pair + each model vs baseline ----
    model_names = sorted(data_by_model.keys())
    pair_list = [(a, b) for i, a in enumerate(model_names) for b in model_names[i + 1 :]]
    if baseline_name is not None:
        pair_list += [(name, baseline_name) for name in model_names]

    def _probs_by_seed_for(name):
        if name == baseline_name:
            return {"baseline": baseline_raw["probs"]}
        return {d["seed"]: d["probs"] for d in data_by_model[name]}

    def _y_true_and_patients_for(name):
        if name == baseline_name:
            return baseline_raw["y_true_idx"], baseline_raw["patient_ids"]
        seeds_data = data_by_model[name]
        return seeds_data[0]["y_true_idx"], seeds_data[0]["patient_ids"]

    paired = {}
    for a, b in pair_list:
        y_true_a, patients_a = _y_true_and_patients_for(a)
        paired[f"{a}_vs_{b}"] = stats.paired_multi_seed_patient_bootstrap(
            y_true_a,
            _probs_by_seed_for(a),
            _probs_by_seed_for(b),
            patients_a,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
            alpha=alpha,
        )

    efficiency = loaders.load_central_efficiency(repo_root=repo_root)

    input_hashes = {}
    for model_name in sorted(data_by_model.keys()):
        for seed_data in data_by_model[model_name]:
            input_hashes[f"{model_name}:seed{seed_data['seed']}"] = _predictions_hash(seed_data)
    if baseline_raw is not None:
        input_hashes[baseline_name] = _predictions_hash(
            {"sample_ids": baseline_raw["sample_ids"], "y_true_idx": baseline_raw["y_true_idx"], "probs": baseline_raw["probs"]}
        )

    return {
        "mode": mode,
        "label": PREVIEW_LABEL if mode == "val-preview" else None,
        "study_version": study["study_version"],
        "present_models": model_names,
        "missing_models": missing,
        "baseline": baseline_name,
        "per_model": per_model,
        "paired": paired,
        "efficiency": efficiency,
        "bootstrap_settings": {"n_resamples": n_resamples, "seed": bootstrap_seed, "alpha": alpha},
        "input_hashes": input_hashes,
    }, data_by_model


def _render_results_md(result: dict, table_frames: dict, figure_paths: list, out_dir: Path) -> str:
    mode = result["mode"]
    lines = ["# Phase 2 cross-model comparison", ""]
    if mode == "val-preview":
        lines += [f"**{PREVIEW_LABEL}**", ""]
    def _table_md(stem: str) -> str:
        path = Path(out_dir) / f"{stem}.md"
        return path.read_text() if path.is_file() else ""

    lines += [
        f"Mode: `{mode}`. Study version: `{result['study_version']}`. "
        f"Models present: {result['present_models']}. Missing: {result['missing_models']}.",
        "",
        "## Headline",
        "",
        _table_md("headline"),
        "## Per-class",
        "",
        _table_md("per_class"),
        "## Paired comparisons",
        "",
        _table_md("paired"),
        "## Size-stratified recall",
        "",
        _table_md("stratified"),
        "## Calibration (D19, supplementary, descriptive only)",
        "",
        _table_md("calibration"),
        "## Efficiency",
        "",
        _table_md("efficiency"),
        "## Figures",
        "",
    ]
    for path in figure_paths:
        rel = Path(path).relative_to(out_dir) if Path(path).is_relative_to(out_dir) else Path(path).name
        lines.append(f"- [{Path(path).name}]({rel})")
    lines += [
        "",
        "## Limitations (D15)",
        "",
    ]
    for item in D15_LIMITATIONS:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Narrative",
        "",
        "**TODO (not auto-generated):** interpretation of the headline results.",
        "",
        "**TODO (not auto-generated):** discussion of the paired comparisons' practical significance.",
        "",
        "**TODO (not auto-generated):** calibration discussion (D19, descriptive/supplementary only --",
        "see D19 in DECISIONS.md: calibration never influenced model selection).",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_compare(mode: str, *, repo_root=None) -> dict:
    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    result, data_by_model = build_comparison(mode, repo_root=repo_root)

    if mode == "test":
        out_dir = repo_root / TEST_OUTPUT_DIR_RELPATH
        results_md_path = repo_root / "phase2" / "docs" / "RESULTS.md"
    else:
        out_dir = repo_root / PREVIEW_OUTPUT_DIR_RELPATH
        results_md_path = out_dir / "RESULTS_PREVIEW.md"

    table_frames = tables.write_all_tables(result, out_dir)
    figure_dir = out_dir / "figures"
    figure_paths = plots.write_all_figures(result, data_by_model, figure_dir, mode=mode)

    comparison_json_path = out_dir / "comparison.json"
    comparison_json_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_json_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    results_md_path.parent.mkdir(parents=True, exist_ok=True)
    results_md_path.write_text(_render_results_md(result, table_frames, figure_paths, out_dir))

    return {
        "result": result,
        "out_dir": out_dir,
        "comparison_json_path": comparison_json_path,
        "results_md_path": results_md_path,
        "figure_paths": figure_paths,
    }
