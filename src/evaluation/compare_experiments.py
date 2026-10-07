"""Compare every Phase 1 experiment under reports/experiments/ and plot it.

Handles both the shared build_result_dict schema and the older
LBP-logistic-regression schema, recomputing per-class metrics/confusion
matrix for the latter from its predictions.csv via evaluate_predictions so
everything lands on one common shape.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import CLASS_NAMES, evaluate_predictions

EXPERIMENTS_ROOT = Path("reports/experiments")
OUTPUT_DIR = Path("reports/figures/model_comparison")
CLASS_LIST = list(CLASS_NAMES.values())


def _standard_summary(metrics):
    return {
        "accuracy": metrics["metrics"]["accuracy"],
        "balanced_accuracy": metrics["metrics"]["balanced_accuracy"],
        "macro_f1": metrics["metrics"]["macro_f1"],
        "weighted_f1": metrics["metrics"]["weighted_f1"],
        "roc_auc_macro": metrics["roc_auc"].get("macro_ovr"),
        "per_class_f1": {name: metrics["per_class_metrics"][name]["f1"] for name in CLASS_LIST},
        "confusion_matrix": metrics["confusion_matrix"],
        "feature_count": metrics["feature_count"],
        "model_name": metrics["model_name"],
        "feature_set": metrics["feature_set"],
    }


def _legacy_summary(metrics, predictions_path):
    test = metrics["test_metrics"]
    predictions = pd.read_csv(predictions_path)
    if "split" in predictions.columns:
        predictions = predictions[predictions["split"] == "test"]
    evaluation = evaluate_predictions(predictions["label"], predictions["predicted_label"])
    return {
        "accuracy": test["accuracy"],
        "balanced_accuracy": test["balanced_accuracy"],
        "macro_f1": test["macro_f1"],
        "weighted_f1": test["weighted_f1"],
        "roc_auc_macro": test.get("roc_auc_ovr_macro"),
        "per_class_f1": {name: evaluation["per_class_metrics"][name]["f1"] for name in CLASS_LIST},
        "confusion_matrix": evaluation["confusion_matrix"],
        "feature_count": metrics["feature_count"],
        "model_name": metrics["model"],
        "feature_set": metrics["feature_set"],
    }


def _find_metrics_file(exp_dir):
    for name in ("metrics.json", "test_metrics.json"):
        candidate = exp_dir / name
        if candidate.is_file():
            return candidate
    return None


def load_all_experiments():
    summaries = {}
    for exp_dir in sorted(EXPERIMENTS_ROOT.iterdir()):
        metrics_path = _find_metrics_file(exp_dir)
        if metrics_path is None:
            continue
        metrics = json.loads(metrics_path.read_text())
        if "metrics" in metrics:
            summaries[exp_dir.name] = _standard_summary(metrics)
        else:
            summaries[exp_dir.name] = _legacy_summary(metrics, exp_dir / "predictions.csv")
    return summaries


def build_comparison_table(summaries):
    rows = []
    for name, summary in summaries.items():
        rows.append(
            {
                "experiment": name,
                "feature_set": summary["feature_set"],
                "model": summary["model_name"],
                "feature_count": summary["feature_count"],
                "accuracy": summary["accuracy"],
                "balanced_accuracy": summary["balanced_accuracy"],
                "macro_f1": summary["macro_f1"],
                "weighted_f1": summary["weighted_f1"],
                "roc_auc_macro": summary["roc_auc_macro"],
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def plot_overall_metrics(table, output_dir):
    metrics_to_plot = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    x = np.arange(len(table))
    width = 0.2
    figure, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for i, metric in enumerate(metrics_to_plot):
        bars = ax.bar(x + (i - 1.5) * width, table[metric], width, label=metric.replace("_", " "))
        ax.bar_label(bars, labels=[f"{v * 100:.0f}%" for v in table[metric]], fontsize=6, padding=2)
    ax.set_xticks(x, table["experiment"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Overall test-set metrics by experiment")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    figure.savefig(output_dir / "overall_metrics.png", dpi=150)
    plt.close(figure)


def plot_per_class_f1(summaries, output_dir):
    names = list(summaries.keys())
    x = np.arange(len(CLASS_LIST))
    width = 0.8 / len(names)
    figure, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for i, name in enumerate(names):
        values = [summaries[name]["per_class_f1"][cls] for cls in CLASS_LIST]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * width, values, width, label=name)
        ax.bar_label(bars, labels=[f"{v * 100:.0f}%" for v in values], fontsize=6, padding=2)
    ax.set_xticks(x, CLASS_LIST)
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 by experiment")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    figure.savefig(output_dir / "per_class_f1.png", dpi=150)
    plt.close(figure)


def plot_cost_vs_benefit(table, output_dir):
    figure, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(table["feature_count"], table["macro_f1"], s=80, color="#1f8a70")
    for _, row in table.iterrows():
        ax.annotate(row["experiment"], (row["feature_count"], row["macro_f1"]), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Feature count (log scale)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Feature cost vs. macro F1")
    ax.grid(alpha=0.3)
    figure.savefig(output_dir / "cost_vs_benefit.png", dpi=150)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = load_all_experiments()
    table = build_comparison_table(summaries)
    table.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)
    plot_overall_metrics(table, OUTPUT_DIR)
    plot_per_class_f1(summaries, OUTPUT_DIR)
    plot_cost_vs_benefit(table, OUTPUT_DIR)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
