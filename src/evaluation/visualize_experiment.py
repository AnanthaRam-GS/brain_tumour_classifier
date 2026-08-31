"""Presentation-ready visualizations for saved classification experiments."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

CLASS_ORDER = [1, 2, 3]
CLASS_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}
PROBABILITY_COLUMNS = {
    1: "prob_meningioma",
    2: "prob_glioma",
    3: "prob_pituitary",
}


class VisualizationError(ValueError):
    """Raised when saved experiment outputs are missing or inconsistent."""


def row_normalize_confusion_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized percentages for a confusion matrix."""

    counts = np.asarray(matrix, dtype=float)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise VisualizationError("confusion matrix must be square")
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0) * 100.0


def prediction_confidence(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return max predicted probability and correctness for each prediction row."""

    required = {"true_label", "predicted_label", *PROBABILITY_COLUMNS.values()}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise VisualizationError(f"prediction table missing probability column(s): {missing}")
    output = predictions.copy(deep=True)
    probabilities = output[list(PROBABILITY_COLUMNS.values())].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise VisualizationError("prediction probabilities contain NaN or infinite values")
    output["confidence"] = probabilities.max(axis=1)
    output["correct"] = output["true_label"].astype(int).eq(output["predicted_label"].astype(int))
    return output


def compute_roc_curves(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute one-vs-rest ROC curves from stored project-label probabilities."""

    missing = sorted(set(PROBABILITY_COLUMNS.values()).difference(predictions.columns))
    if missing:
        raise VisualizationError(f"prediction table missing probability column(s): {missing}")
    y_true = predictions["true_label"].astype(int).to_numpy()
    scores = predictions[[PROBABILITY_COLUMNS[label] for label in CLASS_ORDER]].to_numpy(dtype=float)
    if not set(np.unique(y_true)).issubset(CLASS_ORDER):
        raise VisualizationError("true labels must be in {1, 2, 3}")
    if not np.isfinite(scores).all():
        raise VisualizationError("prediction probabilities contain NaN or infinite values")
    y_bin = label_binarize(y_true, classes=CLASS_ORDER)
    curves: dict[str, Any] = {}
    fpr_grid = np.linspace(0.0, 1.0, 200)
    tpr_values = []
    for index, label in enumerate(CLASS_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, index], scores[:, index])
        curves[CLASS_NAMES[label]] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": float(auc(fpr, tpr)),
        }
        tpr_values.append(np.interp(fpr_grid, fpr, tpr))
    mean_tpr = np.mean(tpr_values, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    curves["Macro average"] = {
        "fpr": fpr_grid,
        "tpr": mean_tpr,
        "auc": float(auc(fpr_grid, mean_tpr)),
    }
    return curves


def human_feature_name(feature: str) -> str:
    """Convert glcm_feature_stat names into compact presentation labels."""

    if feature.startswith("glcm_"):
        feature = feature[len("glcm_") :]
    parts = feature.split("_")
    return " ".join(part.upper() if part == "asm" else part.capitalize() for part in parts)


def _save(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path, *, dpi: int = 300) -> Path:
    counts = np.asarray(matrix, dtype=int)
    normalized = row_normalize_confusion_matrix(counts)
    labels = [CLASS_NAMES[label] for label in CLASS_ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3), labels=labels)
    ax.set_yticks(range(3), labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test Confusion Matrix")
    for row in range(3):
        for col in range(3):
            color = "white" if normalized[row, col] >= 50 else "black"
            ax.text(
                col,
                row,
                f"{counts[row, col]}\n{normalized[row, col]:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )
    fig.colorbar(image, ax=ax, label="Row-normalized percent")
    _save(fig, output_path, dpi=dpi)
    return output_path


def plot_per_class_metrics(metrics: dict[str, Any], output_path: Path, *, dpi: int = 300) -> Path:
    classes = ["meningioma", "glioma", "pituitary"]
    display = ["Meningioma", "Glioma", "Pituitary"]
    measures = ["precision", "recall", "f1"]
    values = np.array([[metrics["per_class_metrics"][cls][m] for m in measures] for cls in classes])
    x = np.arange(len(display))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for index, measure in enumerate(measures):
        bars = ax.bar(x + (index - 1) * width, values[:, index], width, label=measure.capitalize())
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
    ax.set_xticks(x, labels=display)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test Per-Class Metrics")
    ax.legend()
    _save(fig, output_path, dpi=dpi)
    return output_path


def plot_validation_vs_test(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    output_path: Path,
    *,
    dpi: int = 300,
) -> Path:
    keys = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]
    labels = [label for _, label in keys] + ["Macro ROC-AUC", "Weighted ROC-AUC"]
    validation = [validation_metrics["metrics"][key] for key, _ in keys]
    test = [test_metrics["metrics"][key] for key, _ in keys]
    validation += [
        validation_metrics["roc_auc"]["macro_ovr"],
        validation_metrics["roc_auc"]["weighted_ovr"],
    ]
    test += [test_metrics["roc_auc"]["macro_ovr"], test_metrics["roc_auc"]["weighted_ovr"]]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    ax.bar(x - width / 2, validation, width, label="Validation")
    ax.bar(x + width / 2, test, width, label="Test")
    ax.set_xticks(x, labels=labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Validation vs Test Metrics")
    ax.legend()
    _save(fig, output_path, dpi=dpi)
    return output_path


def plot_feature_importance(importance: pd.DataFrame, output_path: Path, *, dpi: int = 300) -> Path:
    table = importance.copy(deep=True).sort_values("importance", ascending=True)
    labels = [f"{human_feature_name(f)}\n({f})" for f in table["feature"]]
    fig, ax = plt.subplots(figsize=(9.0, 6.4), constrained_layout=True)
    ax.barh(labels, table["importance"].astype(float))
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost GLCM Feature Importance")
    _save(fig, output_path, dpi=dpi)
    return output_path


def plot_top_feature_distributions(
    features: pd.DataFrame,
    importance: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 4,
    dpi: int = 300,
) -> Path:
    table = features.copy(deep=True)
    top_features = importance.sort_values("importance", ascending=False)["feature"].head(top_n).tolist()
    class_values = [1, 2, 3]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    for ax, feature in zip(axes.ravel(), top_features, strict=True):
        grouped = [table.loc[table["label"].eq(label), feature].to_numpy(dtype=float) for label in class_values]
        ax.boxplot(grouped, tick_labels=[CLASS_NAMES[label] for label in class_values], showfliers=False)
        ax.set_title(human_feature_name(feature))
        ax.set_ylabel("Original GLCM value")
    fig.suptitle("Top GLCM Feature Distributions by Class", fontsize=14)
    _save(fig, output_path, dpi=dpi)
    return output_path


def plot_roc_curves(predictions: pd.DataFrame, output_path: Path, *, dpi: int = 300) -> tuple[Path, dict[str, float]]:
    curves = compute_roc_curves(predictions)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    for name, curve in curves.items():
        ax.plot(curve["fpr"], curve["tpr"], label=f"{name} AUC={curve['auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test One-vs-Rest ROC Curves")
    ax.legend(loc="lower right")
    _save(fig, output_path, dpi=dpi)
    return output_path, {name: values["auc"] for name, values in curves.items()}


def plot_prediction_confidence(predictions: pd.DataFrame, output_path: Path, *, dpi: int = 300) -> tuple[Path, dict[str, float]]:
    table = prediction_confidence(predictions)
    correct = table.loc[table["correct"], "confidence"].to_numpy(dtype=float)
    incorrect = table.loc[~table["correct"], "confidence"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    ax.boxplot([correct, incorrect], tick_labels=["Correct", "Incorrect"], showfliers=False)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Max predicted probability")
    ax.set_title("Test Prediction Confidence")
    stats = {
        "correct_mean": float(np.mean(correct)),
        "correct_median": float(np.median(correct)),
        "incorrect_mean": float(np.mean(incorrect)),
        "incorrect_median": float(np.median(incorrect)),
    }
    _save(fig, output_path, dpi=dpi)
    return output_path, stats


def plot_candidate_search(candidates: pd.DataFrame, output_path: Path, *, dpi: int = 300) -> Path:
    table = candidates.copy(deep=True).sort_values("candidate_id")
    colors = ["tab:orange" if bool(selected) else "tab:blue" for selected in table["selected"]]
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.bar(table["candidate_id"].astype(int), table["validation_macro_f1"].astype(float), color=colors)
    selected = table[table["selected"].astype(bool)].iloc[0]
    ax.annotate(
        "Selected",
        xy=(selected["candidate_id"], selected["validation_macro_f1"]),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "0.25"},
    )
    ax.set_xlabel("Candidate ID")
    ax.set_ylabel("Validation macro F1")
    ax.set_title("XGBoost Candidate Validation Macro F1")
    _save(fig, output_path, dpi=dpi)
    return output_path


def load_experiment_inputs(experiment_dir: Path | str, features_path: Path | str) -> dict[str, Any]:
    root = Path(experiment_dir)
    return {
        "test_predictions": pd.read_csv(root / "test_predictions.csv"),
        "test_metrics": json.loads((root / "test_metrics.json").read_text(encoding="utf-8")),
        "validation_metrics": json.loads((root / "validation_metrics.json").read_text(encoding="utf-8")),
        "test_confusion": pd.read_csv(root / "test_confusion_matrix.csv").iloc[:, 1:].to_numpy(dtype=int),
        "feature_importance": pd.read_csv(root / "feature_importance.csv"),
        "validation_candidates": pd.read_csv(root / "validation_candidates.csv"),
        "features": pd.read_csv(features_path),
    }


def generate_experiment_visualizations(
    *,
    experiment_dir: Path | str = "reports/experiments/glcm_xgboost_v1",
    features: Path | str = "data/features/glcm/glcm_features.csv",
    output_dir: Path | str | None = None,
    dpi: int = 300,
) -> dict[str, Any]:
    """Generate all Review 1 figures from saved experiment outputs."""

    root = Path(experiment_dir)
    figure_dir = Path(output_dir) if output_dir is not None else root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_experiment_inputs(root, features)
    manifest: list[dict[str, Any]] = []

    def add(filename: str, title: str, purpose: str, source_files: list[str], key_message: str) -> Path:
        path = figure_dir / filename
        manifest.append(
            {
                "filename": filename,
                "title": title,
                "purpose": purpose,
                "source_files": source_files,
                "key_message": key_message,
            }
        )
        return path

    plot_confusion_matrix(
        inputs["test_confusion"],
        add(
            "01_test_confusion_matrix.png",
            "Test Confusion Matrix",
            "Show class-specific classification errors.",
            ["test_confusion_matrix.csv"],
            "Meningioma has the weakest recall and is often predicted as glioma or pituitary.",
        ),
        dpi=dpi,
    )
    plot_per_class_metrics(
        inputs["test_metrics"],
        add(
            "02_per_class_metrics.png",
            "Test Per-Class Metrics",
            "Compare precision, recall, and F1 by tumor class.",
            ["test_metrics.json"],
            "Glioma has the highest test F1, while meningioma has the lowest.",
        ),
        dpi=dpi,
    )
    plot_validation_vs_test(
        inputs["validation_metrics"],
        inputs["test_metrics"],
        add(
            "03_validation_vs_test_metrics.png",
            "Validation vs Test Metrics",
            "Compare validation and held-out test performance.",
            ["validation_metrics.json", "test_metrics.json"],
            "Validation and test macro F1 are very close.",
        ),
        dpi=dpi,
    )
    plot_feature_importance(
        inputs["feature_importance"],
        add(
            "04_glcm_feature_importance.png",
            "XGBoost GLCM Feature Importance",
            "Show all stored XGBoost feature importance values.",
            ["feature_importance.csv"],
            "Homogeneity mean is the highest-importance GLCM feature in this run.",
        ),
        dpi=dpi,
    )
    top4 = inputs["feature_importance"].sort_values("importance", ascending=False)["feature"].head(4).tolist()
    plot_top_feature_distributions(
        inputs["features"],
        inputs["feature_importance"],
        add(
            "05_top_glcm_feature_distributions.png",
            "Top GLCM Feature Distributions by Class",
            "Show original extracted values for the top four GLCM features by class.",
            ["data/features/glcm/glcm_features.csv", "feature_importance.csv"],
            f"Top features shown: {', '.join(top4)}.",
        ),
        dpi=dpi,
    )
    _, roc_aucs = plot_roc_curves(
        inputs["test_predictions"],
        add(
            "06_test_roc_curves.png",
            "Test One-vs-Rest ROC Curves",
            "Show class probability separation from saved test predictions.",
            ["test_predictions.csv"],
            "All one-vs-rest ROC curves are computed from saved XGBoost probabilities.",
        ),
        dpi=dpi,
    )
    _, confidence_stats = plot_prediction_confidence(
        inputs["test_predictions"],
        add(
            "07_prediction_confidence.png",
            "Test Prediction Confidence",
            "Compare maximum predicted probability for correct and incorrect predictions.",
            ["test_predictions.csv"],
            "Incorrect predictions still often have moderately high confidence.",
        ),
        dpi=dpi,
    )
    selected = inputs["validation_candidates"].loc[inputs["validation_candidates"]["selected"].astype(bool)].iloc[0]
    plot_candidate_search(
        inputs["validation_candidates"],
        add(
            "08_xgboost_candidate_validation_macro_f1.png",
            "XGBoost Candidate Validation Macro F1",
            "Show validation-driven hyperparameter selection.",
            ["validation_candidates.csv"],
            f"Candidate {int(selected['candidate_id'])} had the best validation macro F1.",
        ),
        dpi=dpi,
    )

    manifest_payload = {
        "experiment_dir": str(root),
        "figure_dir": str(figure_dir),
        "figures": manifest,
        "roc_auc": roc_aucs,
        "confidence": confidence_stats,
    }
    manifest_path = figure_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"figure_count": len(manifest), "figure_dir": str(figure_dir)}, indent=2))
    return manifest_payload
