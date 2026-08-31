"""Matplotlib figures for a saved Phase 1 experiment result.

Reads the standard ``reports/experiments/<experiment_name>/`` outputs
(``metrics.json``, ``predictions.csv``, and, if present,
``hyperparameter_search.json``) and renders one PNG per plot: confusion
matrix, per-class precision/recall/F1, ROC curves, and the hyperparameter
search grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from src.evaluation.metrics import CLASS_NAMES, CLASS_ORDER

CLASS_COLORS = {1: "#3d6fb4", 2: "#7d5ba6", 3: "#1f8a70"}


def _load_experiment(experiment_dir: Path) -> dict[str, Any]:
    with (experiment_dir / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    predictions = pd.read_csv(experiment_dir / "predictions.csv")
    search_path = experiment_dir / "hyperparameter_search.json"
    search = None
    if search_path.is_file():
        with search_path.open(encoding="utf-8") as handle:
            search = json.load(handle)
    return {"metrics": metrics, "predictions": predictions, "search": search}


def _load_split_metadata(split_metadata_json: Path | str) -> dict[str, Any] | None:
    path = Path(split_metadata_json)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _plot_confusion_matrix(ax: plt.Axes, confusion: list[list[int]]) -> None:
    matrix = np.asarray(confusion)
    row_sums = matrix.sum(axis=1, keepdims=True)
    shares = matrix / row_sums
    names = [CLASS_NAMES[label] for label in CLASS_ORDER]

    ax.imshow(shares, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test)")
    for i in range(len(names)):
        for j in range(len(names)):
            color = "white" if shares[i, j] > 0.55 else "black"
            ax.text(
                j, i, f"{matrix[i, j]}\n{shares[i, j]:.0%}",
                ha="center", va="center", fontsize=9, color=color,
            )


def _plot_per_class_metrics(ax: plt.Axes, per_class_metrics: dict[str, Any]) -> None:
    names = [CLASS_NAMES[label] for label in CLASS_ORDER]
    metrics_order = ("precision", "recall", "f1")
    x = np.arange(len(metrics_order))
    width = 0.25
    for offset, label in zip((-1, 0, 1), CLASS_ORDER):
        name = CLASS_NAMES[label]
        values = [per_class_metrics[name][metric] for metric in metrics_order]
        ax.bar(
            x + offset * width, values, width,
            label=f"{name} (n={per_class_metrics[name]['support']})",
            color=CLASS_COLORS[label],
        )
    ax.set_xticks(x, [metric.capitalize() for metric in metrics_order])
    ax.set_ylim(0, 1)
    ax.set_title("Per-class precision / recall / F1")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)


def _plot_roc_curves(ax: plt.Axes, predictions: pd.DataFrame) -> None:
    for label in CLASS_ORDER:
        name = CLASS_NAMES[label]
        prob_column = f"prob_{name}"
        if prob_column not in predictions.columns:
            continue
        y_true = (predictions["true_label"] == label).astype(int)
        fpr, tpr, _ = roc_curve(y_true, predictions[prob_column])
        auc_value = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLORS[label], label=f"{name} (AUC={auc_value:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, one-vs-rest")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)


def _plot_hyperparameter_search(ax: plt.Axes, search: dict[str, Any] | None) -> None:
    if search is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "no hyperparameter_search.json found", ha="center", va="center")
        return
    trials = pd.DataFrame(search["all_trials"])
    trials["gamma_label"] = trials["gamma"].astype(str)
    c_values = sorted(trials["C"].unique())
    gamma_labels = list(dict.fromkeys(trials["gamma_label"]))
    grid = np.full((len(c_values), len(gamma_labels)), np.nan)
    for _, row in trials.iterrows():
        i = c_values.index(row["C"])
        j = gamma_labels.index(row["gamma_label"])
        grid[i, j] = row["val_macro_f1"]

    im = ax.imshow(grid, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(len(gamma_labels)), gamma_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(c_values)), [f"C={c:g}" for c in c_values])
    ax.set_title("Validation macro F1 by C / gamma")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    chosen = search["chosen_hyperparameters"]
    chosen_gamma_label = str(chosen["gamma"])
    if chosen_gamma_label in gamma_labels:
        ci = c_values.index(chosen["C"])
        gj = gamma_labels.index(chosen_gamma_label)
        ax.add_patch(
            plt.Rectangle((gj - 0.5, ci - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5)
        )
    for i in range(len(c_values)):
        for j in range(len(gamma_labels)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7)


def _plot_class_distribution(ax: plt.Axes, split_metadata: dict[str, Any] | None) -> None:
    if split_metadata is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "no split_metadata.json found", ha="center", va="center")
        return
    counts_by_split = split_metadata["sample_class_counts_by_split"]
    splits = ("train", "val", "test")
    names = [CLASS_NAMES[label] for label in CLASS_ORDER]
    x = np.arange(len(splits))
    width = 0.25
    for offset, label in zip((-1, 0, 1), CLASS_ORDER):
        name = CLASS_NAMES[label]
        values = [counts_by_split[split][name] for split in splits]
        totals = [sum(counts_by_split[split].values()) for split in splits]
        bars = ax.bar(x + offset * width, values, width, label=name, color=CLASS_COLORS[label])
        for bar, value, total in zip(bars, values, totals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value}\n{value / total:.0%}",
                ha="center", va="bottom", fontsize=7,
            )
    ax.set_xticks(x, [split.capitalize() for split in splits])
    ax.set_ylabel("Sample count")
    ax.set_title("Class distribution by split")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.15)


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_experiment_figures(
    experiment_dir: Path | str,
    output_dir: Path | str,
    *,
    split_metadata_json: Path | str = "data/splits/split_metadata.json",
) -> dict[str, Path]:
    """Render one PNG per plot into ``output_dir`` and return their paths."""

    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)
    data = _load_experiment(experiment_dir)
    metrics = data["metrics"]
    split_metadata = _load_split_metadata(split_metadata_json)

    written: dict[str, Path] = {}

    figure, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    _plot_confusion_matrix(ax, metrics["confusion_matrix"])
    written["confusion_matrix"] = _save(figure, output_dir / "confusion_matrix.png")

    figure, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    _plot_per_class_metrics(ax, metrics["per_class_metrics"])
    written["per_class_metrics"] = _save(figure, output_dir / "per_class_metrics.png")

    figure, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    _plot_roc_curves(ax, data["predictions"])
    written["roc_curves"] = _save(figure, output_dir / "roc_curves.png")

    figure, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    _plot_hyperparameter_search(ax, data["search"])
    written["hyperparameter_search"] = _save(figure, output_dir / "hyperparameter_search.png")

    figure, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    _plot_class_distribution(ax, split_metadata)
    written["class_distribution"] = _save(figure, output_dir / "class_distribution.png")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-metadata-json", type=Path, default=Path("data/splits/split_metadata.json")
    )
    args = parser.parse_args()
    written = plot_experiment_figures(
        args.experiment_dir, args.output_dir, split_metadata_json=args.split_metadata_json
    )
    for name, path in written.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
