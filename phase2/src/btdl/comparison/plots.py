"""Comparison figures (matplotlib, Agg backend)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btdl.contracts import CLASS_NAMES_BY_INDEX
from btdl.evaluation.metrics import LABELS, full_metrics

PREVIEW_LABEL = "VALIDATION PREVIEW -- NOT RESULTS"


def _maybe_preview_suffix(title: str, mode: str) -> str:
    return f"{title}\n{PREVIEW_LABEL}" if mode == "val-preview" else title


def plot_headline_with_ci(result: dict, out_dir: Path, *, mode: str) -> Path:
    """One bar per model (+ baseline, test mode) for macro_f1, with the
    pooled-bootstrap 95% CI as an error bar."""

    model_names = sorted(result["per_model"].keys())
    means, los, his = [], [], []
    for name in model_names:
        data = result["per_model"][name]
        headline_mean = data["headline"]["macro_f1"]["mean"]
        pooled = data.get("pooled_bootstrap")
        if pooled and pooled.get("macro_f1", {}).get("ci_low") is not None:
            lo, hi = pooled["macro_f1"]["ci_low"], pooled["macro_f1"]["ci_high"]
        else:
            lo, hi = headline_mean, headline_mean
        means.append(headline_mean)
        los.append(headline_mean - lo)
        his.append(hi - headline_mean)

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(model_names)), 4.5))
    ax.bar(model_names, means, yerr=[los, his], capsize=4, color="#4C72B0")
    ax.set_ylabel("macro F1")
    ax.set_ylim(0, 1.0)
    ax.set_title(_maybe_preview_suffix("Headline macro-F1 (seed mean, pooled-bootstrap 95% CI)", mode), fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()

    out_path = Path(out_dir) / "headline_macro_f1.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_per_class_f1(result: dict, out_dir: Path, *, mode: str) -> Path:
    """Grouped bars: one group per model, one bar per class; meningioma bars
    highlighted (it's the minority/hardest class)."""

    model_names = sorted(result["per_model"].keys())
    n_classes = len(CLASS_NAMES_BY_INDEX)
    x = np.arange(len(model_names))
    width = 0.8 / n_classes

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(model_names)), 4.5))
    for class_index, class_name in enumerate(CLASS_NAMES_BY_INDEX):
        values = [result["per_model"][m]["per_class"][class_name]["f1"]["mean"] for m in model_names]
        offset = (class_index - (n_classes - 1) / 2) * width
        color = "#C44E52" if class_name == "meningioma" else "#55A868"
        ax.bar(x + offset, values, width=width, label=class_name, color=color, alpha=1.0 if class_name == "meningioma" else 0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_ylabel("F1 (seed mean)")
    ax.set_ylim(0, 1.0)
    ax.set_title(_maybe_preview_suffix("Per-class F1 (meningioma highlighted)", mode), fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = Path(out_dir) / "per_class_f1.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_confusion_matrix_grid(result: dict, out_dir: Path, *, mode: str) -> Path:
    """One row-normalized confusion matrix per model, counts SUMMED across
    seeds (each test/val sample appears once per seed, so this is a sum of
    `n_seeds` independent passes over the same held-out set, not n distinct
    samples)."""

    model_names = sorted(result["per_model"].keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        cm = np.asarray(result["per_model"][name]["confusion_matrix_sum"], dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES_BY_INDEX, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(CLASS_NAMES_BY_INDEX, fontsize=7)
        ax.set_title(name, fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.suptitle(_maybe_preview_suffix("Confusion matrices (row-normalized, seed-summed counts)", mode), fontsize=10)
    fig.tight_layout()

    out_path = Path(out_dir) / "confusion_matrix_grid.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_roc_overlay(data_by_model: dict, out_dir: Path, *, mode: str) -> Path:
    """One macro ROC curve per model, using the SEED-AVERAGED probabilities
    (averaged per sample across the model's seeds) -- for plotting only; the
    headline/pooled-bootstrap AUC numbers use each seed's own metrics, never
    this averaged curve."""

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model_name in sorted(data_by_model.keys()):
        seeds_data = data_by_model[model_name]
        probs_stack = np.stack([d["probs"] for d in seeds_data], axis=0)
        avg_probs = probs_stack.mean(axis=0)
        y_true_idx = seeds_data[0]["y_true_idx"]
        metrics = full_metrics(y_true_idx, avg_probs)
        macro_auc = metrics["roc_auc"]["macro_ovr"]

        # Macro ROC via a simple average of per-class curves at a shared fpr
        # grid (full_metrics() doesn't return per-class curves, only AUCs).
        from sklearn.metrics import roc_curve

        fpr_grid = np.linspace(0, 1, 100)
        tpr_interp_sum = np.zeros_like(fpr_grid)
        n_valid = 0
        for class_index in range(3):
            y_true_binary = (y_true_idx == class_index).astype(int)
            if len(np.unique(y_true_binary)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true_binary, avg_probs[:, class_index])
            tpr_interp_sum += np.interp(fpr_grid, fpr, tpr)
            n_valid += 1
        if n_valid == 0:
            continue
        macro_tpr = tpr_interp_sum / n_valid
        label = f"{model_name} (AUC={macro_auc:.3f})" if macro_auc is not None else model_name
        ax.plot(fpr_grid, macro_tpr, label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        _maybe_preview_suffix(
            "ROC overlay (macro, seed-averaged probabilities -- for plotting only)", mode
        ),
        fontsize=9,
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out_path = Path(out_dir) / "roc_overlay.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_training_curve_grid(data_by_model: dict, out_dir: Path, *, mode: str) -> Path:
    """val_macro_f1 vs epoch, all seeds overlaid, one panel per model."""

    model_names = sorted(data_by_model.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, model_name in zip(axes, model_names):
        for seed_data in data_by_model[model_name]:
            history_path = seed_data.get("history_path")
            if history_path is None or not Path(history_path).is_file():
                continue
            history = pd.read_csv(history_path)
            ax.plot(history["epoch"], history["val_macro_f1"], label=f"seed {seed_data['seed']}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val macro F1")
        ax.set_title(model_name, fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(_maybe_preview_suffix("Training curves (val macro-F1, all seeds)", mode), fontsize=10)
    fig.tight_layout()

    out_path = Path(out_dir) / "training_curves_grid.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_macro_f1_vs_parameters(result: dict, out_dir: Path, *, mode: str) -> Path:
    model_names = sorted(result["per_model"].keys())
    params_by_device = {}
    for payload in result.get("efficiency", {}).values():
        for model_name, model_result in payload.get("results", {}).items():
            params_by_device.setdefault(model_name, model_result["parameters"]["total"])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name in model_names:
        if name not in params_by_device:
            continue
        macro_f1 = result["per_model"][name]["headline"]["macro_f1"]["mean"]
        ax.scatter(params_by_device[name], macro_f1)
        ax.annotate(name, (params_by_device[name], macro_f1), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("parameters (total)")
    ax.set_ylabel("macro F1 (seed mean)")
    ax.set_title(_maybe_preview_suffix("Macro-F1 vs parameters", mode))
    fig.tight_layout()

    out_path = Path(out_dir) / "macro_f1_vs_parameters.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_macro_f1_vs_cpu_latency(result: dict, out_dir: Path, *, mode: str) -> Path:
    model_names = sorted(result["per_model"].keys())
    cpu_payload = result.get("efficiency", {}).get("cpu", {}).get("results", {})

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name in model_names:
        model_result = cpu_payload.get(name)
        if model_result is None:
            continue
        ms_per_img = model_result["inference"].get("1", {}).get("ms_per_image")
        if ms_per_img is None:
            continue
        macro_f1 = result["per_model"][name]["headline"]["macro_f1"]["mean"]
        ax.scatter(ms_per_img, macro_f1)
        ax.annotate(name, (ms_per_img, macro_f1), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("CPU ms/image (batch=1)")
    ax.set_ylabel("macro F1 (seed mean)")
    ax.set_title(_maybe_preview_suffix("Macro-F1 vs CPU latency", mode))
    fig.tight_layout()

    out_path = Path(out_dir) / "macro_f1_vs_cpu_latency.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_reliability_diagrams(result: dict, out_dir: Path, *, mode: str) -> Path:
    """One reliability diagram per model, using the FIRST seed's bins (noted
    in the caption/title) for a single representative curve per model."""

    model_names = sorted(result["per_model"].keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, name in zip(axes, model_names):
        calibration = result["per_model"][name].get("calibration")
        if not calibration or not calibration.get("per_seed_bins"):
            continue
        bins = calibration["per_seed_bins"][0]
        first_seed = result["per_model"][name]["seeds"][0]
        centers = [(b["bin_low"] + b["bin_high"]) / 2 for b in bins if b["count"] > 0]
        accuracies = [b["accuracy"] for b in bins if b["count"] > 0]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.plot(centers, accuracies, marker="o", markersize=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{name} (seed {first_seed})", fontsize=9)

    fig.suptitle(_maybe_preview_suffix("Reliability diagrams (first seed, 15 equal-width bins)", mode), fontsize=10)
    fig.tight_layout()

    out_path = Path(out_dir) / "reliability_diagrams.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def write_all_figures(result: dict, data_by_model: dict, out_dir: Path, *, mode: str) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_headline_with_ci(result, out_dir, mode=mode),
        plot_per_class_f1(result, out_dir, mode=mode),
        plot_confusion_matrix_grid(result, out_dir, mode=mode),
        plot_roc_overlay(data_by_model, out_dir, mode=mode),
        plot_training_curve_grid(data_by_model, out_dir, mode=mode),
        plot_macro_f1_vs_parameters(result, out_dir, mode=mode),
        plot_macro_f1_vs_cpu_latency(result, out_dir, mode=mode),
        plot_reliability_diagrams(result, out_dir, mode=mode),
    ]
    return [p for p in paths if p is not None]
