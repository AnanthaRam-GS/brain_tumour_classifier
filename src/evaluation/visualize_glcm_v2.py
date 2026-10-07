"""Presentation visualizations for promoted GLCM v2 vs preserved GLCM v1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
_MPL_CONFIG_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "cv-glcm-v2-visualizations-matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

CLASS_ORDER = [1, 2, 3]
CLASS_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}
CLASS_KEYS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
PROBABILITY_COLUMNS = {
    1: "prob_meningioma",
    2: "prob_glioma",
    3: "prob_pituitary",
}
PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm"]


class GLCMV2VisualizationError(ValueError):
    """Raised when v2 visualization source files are missing or inconsistent."""


def load_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def confusion_matrix_from_csv(path: Path | str) -> np.ndarray:
    frame = pd.read_csv(path)
    matrix = frame.drop(columns=[frame.columns[0]]).to_numpy(dtype=int)
    if matrix.shape != (3, 3):
        raise GLCMV2VisualizationError(f"confusion matrix must be 3x3, got {matrix.shape}")
    return matrix


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    counts = np.asarray(matrix, dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0) * 100.0


def probability_scores(predictions: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(PROBABILITY_COLUMNS.values()).difference(predictions.columns))
    if missing:
        raise GLCMV2VisualizationError(f"missing probability columns: {missing}")
    scores = predictions[[PROBABILITY_COLUMNS[label] for label in CLASS_ORDER]].to_numpy(float)
    if not np.isfinite(scores).all():
        raise GLCMV2VisualizationError("probabilities contain NaN or infinite values")
    return scores


def roc_auc_by_class(predictions: pd.DataFrame) -> dict[str, float]:
    y_true = predictions["true_label"].astype(int).to_numpy()
    scores = probability_scores(predictions)
    y_bin = label_binarize(y_true, classes=CLASS_ORDER)
    values = {}
    macro_curves = []
    grid = np.linspace(0.0, 1.0, 300)
    for index, label in enumerate(CLASS_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, index], scores[:, index])
        values[CLASS_NAMES[label]] = float(auc(fpr, tpr))
        macro_curves.append(np.interp(grid, fpr, tpr))
    values["Macro"] = float(auc(grid, np.mean(macro_curves, axis=0)))
    return values


def prediction_confidence_stats(predictions: pd.DataFrame) -> dict[str, float]:
    table = predictions.copy(deep=True)
    table["confidence"] = probability_scores(table).max(axis=1)
    table["correct"] = table["true_label"].astype(int).eq(table["predicted_label"].astype(int))
    correct = table.loc[table["correct"], "confidence"].to_numpy(float)
    incorrect = table.loc[~table["correct"], "confidence"].to_numpy(float)
    return {
        "correct_mean": float(correct.mean()),
        "correct_median": float(np.median(correct)),
        "incorrect_mean": float(incorrect.mean()),
        "incorrect_median": float(np.median(incorrect)),
    }


def human_feature_name(feature: str) -> str:
    name = feature.removeprefix("glcm_")
    parts = name.split("_")
    if len(parts) >= 3 and parts[-2].startswith("d") and parts[-1].startswith("a"):
        prop = " ".join(parts[:-2]).upper() if parts[0] == "asm" else " ".join(parts[:-2]).capitalize()
        return f"{prop} (d={parts[-2][1:]}, {parts[-1][1:]} deg)"
    return " ".join(part.upper() if part == "asm" else part.capitalize() for part in parts)


def feature_property(feature: str) -> str:
    body = feature.removeprefix("glcm_")
    for prop in PROPERTIES:
        if body == prop or body.startswith(f"{prop}_"):
            return prop
    raise GLCMV2VisualizationError(f"cannot determine GLCM property for {feature}")


def feature_distance(feature: str) -> str | None:
    for part in feature.split("_"):
        if part in {"d1", "d2", "d4"}:
            return part
    return None


def feature_angle(feature: str) -> str | None:
    for part in feature.split("_"):
        if part in {"a0", "a45", "a90", "a135"}:
            return part
    return None


def aggregate_importance_by_property(importance: pd.DataFrame) -> pd.DataFrame:
    table = importance.copy(deep=True)
    table["property"] = table["feature"].map(feature_property)
    return table.groupby("property", as_index=False)["importance"].sum()


def aggregate_importance_by_distance(importance: pd.DataFrame) -> pd.DataFrame:
    table = importance.copy(deep=True)
    table["distance"] = table["feature"].map(feature_distance)
    return table.dropna(subset=["distance"]).groupby("distance", as_index=False)["importance"].sum()


def aggregate_importance_by_angle(importance: pd.DataFrame) -> pd.DataFrame:
    table = importance.copy(deep=True)
    table["angle"] = table["feature"].map(feature_angle)
    return table.dropna(subset=["angle"]).groupby("angle", as_index=False)["importance"].sum()


def best_macro_f1_by_weighting(candidates: pd.DataFrame) -> pd.DataFrame:
    return (
        candidates.groupby("weighting_mode", as_index=False)["validation_macro_f1"]
        .max()
        .sort_values("validation_macro_f1", ascending=False, kind="mergesort")
    )


def _save(fig: plt.Figure, path: Path, *, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(matrix: np.ndarray, path: Path, title: str) -> Path:
    norm = row_normalize(matrix)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    labels = [CLASS_NAMES[label] for label in CLASS_ORDER]
    ax.set_xticks(range(3), labels=labels)
    ax.set_yticks(range(3), labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{matrix[row, col]}\n{norm[row, col]:.1f}%", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Row-normalized percent")
    return _save(fig, path)


def plot_per_class_metrics(metrics: dict[str, Any], path: Path) -> Path:
    measures = ["precision", "recall", "f1"]
    x = np.arange(3)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for offset, measure in enumerate(measures):
        values = [metrics["per_class_metrics"][CLASS_KEYS[label]][measure] for label in CLASS_ORDER]
        bars = ax.bar(x + (offset - 1) * width, values, width, label=measure.capitalize())
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, labels=[CLASS_NAMES[label] for label in CLASS_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("GLCM v2 XGBoost Test Per-Class Metrics")
    ax.legend()
    return _save(fig, path)


def plot_overall_comparison(v1: dict[str, Any], v2: dict[str, Any], path: Path) -> Path:
    specs = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]
    labels = [label for _, label in specs] + ["Macro ROC-AUC", "Weighted ROC-AUC"]
    v1_values = [v1["metrics"][key] for key, _ in specs] + [v1["roc_auc"]["macro_ovr"], v1["roc_auc"]["weighted_ovr"]]
    v2_values = [v2["metrics"][key] for key, _ in specs] + [v2["roc_auc"]["macro_ovr"], v2["roc_auc"]["weighted_ovr"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    width = 0.36
    ax.bar(x - width / 2, v1_values, width, label="V1")
    bars = ax.bar(x + width / 2, v2_values, width, label="V2")
    for index, bar in enumerate(bars):
        delta = v2_values[index] - v1_values[index]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{delta:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels=labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("GLCM v1 vs v2 Overall Test Metrics")
    ax.legend()
    return _save(fig, path)


def plot_class_f1_comparison(v1: dict[str, Any], v2: dict[str, Any], path: Path) -> Path:
    labels = [CLASS_NAMES[label] for label in CLASS_ORDER]
    v1_values = [v1["per_class_metrics"][CLASS_KEYS[label]]["f1"] for label in CLASS_ORDER]
    v2_values = [v2["per_class_metrics"][CLASS_KEYS[label]]["f1"] for label in CLASS_ORDER]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    width = 0.36
    ax.bar(x - width / 2, v1_values, width, label="V1 F1")
    ax.bar(x + width / 2, v2_values, width, label="V2 F1")
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("Class-Wise Test F1: GLCM v1 vs v2")
    ax.legend()
    return _save(fig, path)


def plot_confusion_comparison(v1_matrix: np.ndarray, v2_matrix: np.ndarray, path: Path) -> Path:
    labels = [CLASS_NAMES[label] for label in CLASS_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, matrix, title in zip(axes, [v1_matrix, v2_matrix], ["V1", "V2"], strict=True):
        norm = row_normalize(matrix)
        image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(3), labels=labels, rotation=20, ha="right")
        ax.set_yticks(range(3), labels=labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        for row in range(3):
            for col in range(3):
                ax.text(col, row, f"{matrix[row, col]}\n{norm[row, col]:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Row-normalized percent")
    fig.suptitle("Test Confusion Matrix Comparison")
    return _save(fig, path)


def plot_feature_importance_top20(importance: pd.DataFrame, path: Path) -> Path:
    table = importance.sort_values("importance", ascending=False).head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.barh([human_feature_name(f) for f in table["feature"]], table["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost v2 - Top GLCM Feature Importances")
    return _save(fig, path)


def plot_importance_bars(table: pd.DataFrame, label_col: str, title: str, path: Path) -> Path:
    ordered = table.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    ax.barh(ordered[label_col], ordered["importance"])
    ax.set_xlabel("Summed importance")
    ax.set_title(title)
    return _save(fig, path)


def plot_top_feature_distributions(features: pd.DataFrame, importance: pd.DataFrame, path: Path) -> Path:
    top = importance.sort_values("importance", ascending=False)["feature"].head(6).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, feature in zip(axes.ravel(), top, strict=True):
        groups = [features.loc[features["label"].eq(label), feature].to_numpy(float) for label in CLASS_ORDER]
        ax.boxplot(groups, tick_labels=[CLASS_NAMES[label] for label in CLASS_ORDER], showfliers=False)
        ax.set_title(human_feature_name(feature), fontsize=10)
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Top GLCM v2 Feature Distributions by Class")
    return _save(fig, path)


def plot_roc_curves(predictions: pd.DataFrame, path: Path) -> tuple[Path, dict[str, float]]:
    y_true = predictions["true_label"].astype(int).to_numpy()
    scores = probability_scores(predictions)
    y_bin = label_binarize(y_true, classes=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    aucs = {}
    grid = np.linspace(0, 1, 300)
    macro_tprs = []
    for index, label in enumerate(CLASS_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, index], scores[:, index])
        aucs[CLASS_NAMES[label]] = float(auc(fpr, tpr))
        macro_tprs.append(np.interp(grid, fpr, tpr))
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES[label]} AUC={aucs[CLASS_NAMES[label]]:.3f}")
    macro = np.mean(macro_tprs, axis=0)
    aucs["Macro"] = float(auc(grid, macro))
    ax.plot(grid, macro, label=f"Macro AUC={aucs['Macro']:.3f}", linewidth=2.2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.4", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("GLCM v2 Test ROC Curves")
    ax.legend(loc="lower right")
    return _save(fig, path), aucs


def plot_auc_comparison(v1_predictions: pd.DataFrame, v2_predictions: pd.DataFrame, path: Path) -> tuple[Path, dict[str, dict[str, float]]]:
    v1 = roc_auc_by_class(v1_predictions)
    v2 = roc_auc_by_class(v2_predictions)
    labels = ["Meningioma", "Glioma", "Pituitary", "Macro"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    width = 0.36
    ax.bar(x - width / 2, [v1[label] for label in labels], width, label="V1 AUC")
    ax.bar(x + width / 2, [v2[label] for label in labels], width, label="V2 AUC")
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_title("Class ROC-AUC: GLCM v1 vs v2")
    ax.legend()
    return _save(fig, path), {"v1": v1, "v2": v2}


def plot_prediction_confidence(predictions: pd.DataFrame, path: Path) -> tuple[Path, dict[str, float]]:
    stats = prediction_confidence_stats(predictions)
    table = predictions.copy(deep=True)
    table["confidence"] = probability_scores(table).max(axis=1)
    table["correct"] = table["true_label"].astype(int).eq(table["predicted_label"].astype(int))
    groups = [table.loc[table["correct"], "confidence"], table.loc[~table["correct"], "confidence"]]
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    ax.boxplot(groups, tick_labels=["Correct", "Incorrect"], showfliers=False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Max predicted probability")
    ax.set_title("GLCM v2 Prediction Confidence")
    return _save(fig, path), stats


def plot_confidence_comparison(v1_predictions: pd.DataFrame, v2_predictions: pd.DataFrame, path: Path) -> tuple[Path, dict[str, float]]:
    frames = []
    for version, predictions in [("V1", v1_predictions), ("V2", v2_predictions)]:
        table = predictions.copy(deep=True)
        table["confidence"] = probability_scores(table).max(axis=1)
        table["correct"] = table["true_label"].astype(int).eq(table["predicted_label"].astype(int))
        frames.append(table.assign(group=version + " " + np.where(table["correct"], "correct", "incorrect")))
    combined = pd.concat(frames, ignore_index=True)
    order = ["V1 correct", "V1 incorrect", "V2 correct", "V2 incorrect"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.boxplot([combined.loc[combined["group"].eq(group), "confidence"] for group in order], tick_labels=order, showfliers=False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Max predicted probability")
    ax.set_title("Prediction Confidence: V1 vs V2")
    stats = {group: float(combined.loc[combined["group"].eq(group), "confidence"].median()) for group in order}
    return _save(fig, path), stats


def plot_correlation_heatmap(features: pd.DataFrame, path: Path) -> Path:
    feature_cols = [c for c in features.columns if c not in ("sample_id", "patient_id", "label", "split")]
    corr = features[feature_cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ticks = np.arange(0, len(feature_cols), 14)
    ax.set_xticks(ticks, labels=[str(t + 1) for t in ticks], rotation=90)
    ax.set_yticks(ticks, labels=[str(t + 1) for t in ticks])
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Feature index")
    ax.set_title("GLCM v2 Feature Correlation Heatmap")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    return _save(fig, path)


def property_correlation_summary(features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in features.columns if c not in ("sample_id", "patient_id", "label", "split")]
    rows = []
    corr = features[feature_cols].corr().abs()
    for prop in PROPERTIES:
        cols = [c for c in feature_cols if feature_property(c) == prop]
        values = corr.loc[cols, cols].to_numpy()
        upper = values[np.triu_indices_from(values, k=1)]
        rows.append({"property": prop, "mean_abs_correlation": float(upper.mean())})
    return pd.DataFrame(rows)


def plot_property_correlation_summary(features: pd.DataFrame, path: Path) -> tuple[Path, pd.DataFrame]:
    table = property_correlation_summary(features)
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    ax.bar(table["property"], table["mean_abs_correlation"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean absolute within-property correlation")
    ax.set_title("GLCM v2 Property-Level Correlation Summary")
    return _save(fig, path), table


def plot_candidate_search(candidates: pd.DataFrame, path: Path) -> Path:
    table = candidates.sort_values("candidate_id")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for mode, subset in table.groupby("weighting_mode", sort=False):
        ax.scatter(subset["candidate_id"], subset["validation_macro_f1"], label=mode, s=28)
    selected = table[table["selected"].astype(bool)]
    ax.scatter(selected["candidate_id"], selected["validation_macro_f1"], color="black", s=95, marker="*", label="selected")
    ax.set_xlabel("Candidate ID")
    ax.set_ylabel("Validation macro F1")
    ax.set_title("GLCM v2 XGBoost Validation Candidate Search")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_weighting_strategy(candidates: pd.DataFrame, path: Path) -> tuple[Path, pd.DataFrame]:
    table = best_macro_f1_by_weighting(candidates)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = ax.bar(table["weighting_mode"], table["validation_macro_f1"])
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_ylim(0, max(0.7, float(table["validation_macro_f1"].max()) + 0.05))
    ax.set_ylabel("Best validation macro F1")
    ax.set_title("Best Validation Macro F1 by Weighting Strategy")
    ax.tick_params(axis="x", labelrotation=20)
    return _save(fig, path), table


def plot_representation_diagram(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.3")
    ax.text(0.18, 0.75, "V1\n3 distances x 4 angles x 6 properties", ha="center", va="center", bbox=box, fontsize=11)
    ax.text(0.18, 0.45, "mean/std aggregation", ha="center", va="center", bbox=box, fontsize=11)
    ax.text(0.18, 0.18, "12 features", ha="center", va="center", bbox=box, fontsize=12, fontweight="bold")
    ax.text(0.70, 0.75, "V2\n3 distances x 4 angles x 6 properties", ha="center", va="center", bbox=box, fontsize=11)
    ax.text(0.70, 0.45, "72 directional features\n+\n12 aggregate features", ha="center", va="center", bbox=box, fontsize=11)
    ax.text(0.70, 0.18, "84 features", ha="center", va="center", bbox=box, fontsize=12, fontweight="bold")
    for x in (0.18, 0.70):
        ax.annotate("", xy=(x, 0.56), xytext=(x, 0.68), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.annotate("", xy=(x, 0.27), xytext=(x, 0.37), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("GLCM v1 vs v2 Feature Representation")
    return _save(fig, path)


def plot_improvement_summary(v1: dict[str, Any], v2: dict[str, Any], path: Path) -> Path:
    rows = [
        ("Features", 12, 84, ""),
        ("Accuracy", v1["metrics"]["accuracy"], v2["metrics"]["accuracy"], "%"),
        ("Macro F1", v1["metrics"]["macro_f1"], v2["metrics"]["macro_f1"], "%"),
        ("Balanced Accuracy", v1["metrics"]["balanced_accuracy"], v2["metrics"]["balanced_accuracy"], "%"),
        ("Meningioma F1", v1["per_class_metrics"]["meningioma"]["f1"], v2["per_class_metrics"]["meningioma"]["f1"], "%"),
        ("Macro ROC-AUC", v1["roc_auc"]["macro_ovr"], v2["roc_auc"]["macro_ovr"], ""),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.axis("off")
    y = 0.86
    ax.text(0.5, 0.96, "GLCM v1 to v2 Improvement Summary", ha="center", fontsize=15, fontweight="bold")
    ax.text(0.18, y, "Metric", fontweight="bold")
    ax.text(0.46, y, "V1", fontweight="bold")
    ax.text(0.64, y, "V2", fontweight="bold")
    ax.text(0.80, y, "Delta", fontweight="bold")
    for metric, old, new, fmt in rows:
        y -= 0.12
        if fmt == "%":
            old_text, new_text, delta_text = f"{old*100:.2f}%", f"{new*100:.2f}%", f"{(new-old)*100:+.2f} pts"
        else:
            old_text, new_text, delta_text = f"{old}", f"{new}", f"{new-old:+.4f}" if isinstance(old, float) else f"{new-old:+}"
        ax.text(0.18, y, metric)
        ax.text(0.46, y, old_text)
        ax.text(0.64, y, new_text)
        ax.text(0.80, y, delta_text)
    return _save(fig, path)


def generate_glcm_v2_visualizations(
    *,
    v2_experiment_dir: Path | str = "reports/experiments/glcm_xgboost_v2",
    v1_experiment_dir: Path | str = "reports/experiments/glcm_xgboost_v1",
    v2_features: Path | str = "data/features/glcm_v2/glcm_v2_features.csv",
    output_dir: Path | str | None = None,
    expected_test_count: int | None = 498,
    expected_feature_rows: int | None = 3064,
    expected_feature_count: int | None = 84,
) -> dict[str, Any]:
    v2_root = Path(v2_experiment_dir)
    v1_root = Path(v1_experiment_dir)
    figure_root = Path(output_dir) if output_dir else v2_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    v1_metrics = load_json(v1_root / "test_metrics.json")
    v2_metrics = load_json(v2_root / "test_metrics.json")
    v1_predictions = pd.read_csv(v1_root / "test_predictions.csv")
    v2_predictions = pd.read_csv(v2_root / "test_predictions.csv")
    v1_matrix = confusion_matrix_from_csv(v1_root / "test_confusion_matrix.csv")
    v2_matrix = confusion_matrix_from_csv(v2_root / "test_confusion_matrix.csv")
    importance = pd.read_csv(v2_root / "feature_importance.csv")
    candidates = pd.read_csv(v2_root / "validation_candidates.csv")
    features = pd.read_csv(v2_features, dtype={"patient_id": str})

    if expected_test_count is not None and int(v2_matrix.sum()) != expected_test_count:
        raise GLCMV2VisualizationError(
            f"v2 test confusion matrix does not sum to {expected_test_count}"
        )
    feature_cols = [c for c in features.columns if c not in ("sample_id", "patient_id", "label", "split")]
    if expected_feature_rows is not None and len(features) != expected_feature_rows:
        raise GLCMV2VisualizationError("v2 feature table has unexpected row count")
    if expected_feature_count is not None and (
        len(feature_cols) != expected_feature_count or len(importance) != expected_feature_count
    ):
        raise GLCMV2VisualizationError("v2 features or importance file has unexpected shape")

    manifest = []

    def add(path: Path, title: str, purpose: str, sources: list[str], message: str) -> None:
        manifest.append(
            {
                "filename": path.name,
                "title": title,
                "purpose": purpose,
                "source_files": sources,
                "key_message": message,
            }
        )

    path = plot_confusion(v2_matrix, figure_root / "01_v2_test_confusion_matrix.png", "GLCM v2 XGBoost Test Confusion Matrix")
    add(path, "GLCM v2 test confusion matrix", "Show v2 class-specific test errors.", [str(v2_root / "test_confusion_matrix.csv")], "Meningioma recall improved over v1 but remains weakest.")
    path = plot_per_class_metrics(v2_metrics, figure_root / "02_v2_per_class_metrics.png")
    add(path, "GLCM v2 per-class metrics", "Compare v2 precision, recall, and F1 by class.", [str(v2_root / "test_metrics.json")], "Glioma has the strongest F1; meningioma remains lowest.")
    path = plot_overall_comparison(v1_metrics, v2_metrics, figure_root / "03_v1_vs_v2_overall_metrics.png")
    add(path, "V1 vs V2 overall test metrics", "Compare overall held-out test metrics.", [str(v1_root / "test_metrics.json"), str(v2_root / "test_metrics.json")], "V2 improves accuracy, balanced accuracy, macro F1, weighted F1, and ROC-AUC.")
    path = plot_class_f1_comparison(v1_metrics, v2_metrics, figure_root / "04_v1_vs_v2_class_f1.png")
    add(path, "V1 vs V2 class F1", "Compare per-class test F1.", [str(v1_root / "test_metrics.json"), str(v2_root / "test_metrics.json")], "Meningioma F1 improves most; pituitary F1 decreases slightly.")
    path = plot_confusion_comparison(v1_matrix, v2_matrix, figure_root / "05_v1_vs_v2_confusion_comparison.png")
    add(path, "V1 vs V2 confusion matrix comparison", "Show how test confusion patterns changed.", [str(v1_root / "test_confusion_matrix.csv"), str(v2_root / "test_confusion_matrix.csv")], "Correct meningioma and glioma counts increased; pituitary correct count decreased.")
    path = plot_feature_importance_top20(importance, figure_root / "06_v2_feature_importance_top20.png")
    add(path, "XGBoost v2 top feature importances", "Show top directional GLCM features.", [str(v2_root / "feature_importance.csv")], "Top features are dominated by homogeneity, ASM, and energy descriptors.")
    prop = aggregate_importance_by_property(importance)
    path = plot_importance_bars(prop.assign(property=prop["property"].str.title().str.replace("Asm", "ASM")), "property", "Aggregated Importance by GLCM Property", figure_root / "07_v2_importance_by_property.png")
    add(path, "Importance by property", "Aggregate XGBoost importance by GLCM property family.", [str(v2_root / "feature_importance.csv")], "Homogeneity-family features contribute the largest summed importance.")
    dist = aggregate_importance_by_distance(importance)
    path = plot_importance_bars(dist, "distance", "Directional Importance by Distance", figure_root / "08_v2_importance_by_distance.png")
    add(path, "Importance by distance", "Aggregate directional importance by GLCM distance.", [str(v2_root / "feature_importance.csv")], "Coarser d=4 and mid-scale d=2 information are prominent.")
    angle = aggregate_importance_by_angle(importance)
    path = plot_importance_bars(angle, "angle", "Directional Importance by Angle", figure_root / "09_v2_importance_by_angle.png")
    add(path, "Importance by angle", "Aggregate directional importance by angle.", [str(v2_root / "feature_importance.csv")], "Directional importance is not uniform across angles.")
    path = plot_top_feature_distributions(features, importance, figure_root / "10_v2_top_feature_distributions.png")
    add(path, "Top feature distributions by class", "Visualize original feature values by class.", [str(v2_features), str(v2_root / "feature_importance.csv")], "Top features show overlap with some class-level shifts.")
    path, v2_auc = plot_roc_curves(v2_predictions, figure_root / "11_v2_test_roc_curves.png")
    add(path, "GLCM v2 ROC curves", "Show one-vs-rest ROC curves from saved test probabilities.", [str(v2_root / "test_predictions.csv")], "V2 probabilities separate classes better than random for all classes.")
    path, auc_comparison = plot_auc_comparison(v1_predictions, v2_predictions, figure_root / "12_v1_vs_v2_class_auc.png")
    add(path, "V1 vs V2 class AUC", "Compare one-vs-rest AUC by class and macro average.", [str(v1_root / "test_predictions.csv"), str(v2_root / "test_predictions.csv")], "V2 improves macro AUC and most class probability separation.")
    path, v2_conf = plot_prediction_confidence(v2_predictions, figure_root / "13_v2_prediction_confidence.png")
    add(path, "V2 prediction confidence", "Compare confidence for correct and incorrect v2 predictions.", [str(v2_root / "test_predictions.csv")], "Correct predictions tend to have higher median confidence.")
    path, confidence_comparison = plot_confidence_comparison(v1_predictions, v2_predictions, figure_root / "14_v1_vs_v2_prediction_confidence.png")
    add(path, "V1 vs V2 prediction confidence", "Compare confidence distributions across versions.", [str(v1_root / "test_predictions.csv"), str(v2_root / "test_predictions.csv")], "Confidence distributions should not be interpreted as calibration.")
    path = plot_correlation_heatmap(features, figure_root / "15_v2_feature_correlation_heatmap.png")
    add(path, "GLCM v2 feature correlation heatmap", "Visualize redundancy across 84 features.", [str(v2_features)], "The expanded representation contains many correlated descriptors.")
    path, prop_corr = plot_property_correlation_summary(features, figure_root / "16_v2_property_correlation_summary.png")
    add(path, "Property correlation summary", "Summarize within-property redundancy.", [str(v2_features)], "Within-property correlations are high for several GLCM families.")
    path = plot_candidate_search(candidates, figure_root / "17_v2_candidate_search.png")
    add(path, "Validation candidate search", "Show validation macro F1 across candidates.", [str(v2_root / "validation_candidates.csv")], "Candidate 24 was selected by validation macro F1 policy.")
    path, weighting = plot_weighting_strategy(candidates, figure_root / "18_v2_weighting_strategy_comparison.png")
    add(path, "Weighting strategy comparison", "Compare best validation macro F1 per weighting mode.", [str(v2_root / "validation_candidates.csv")], "The unweighted strategy achieved the best validation macro F1.")
    path = plot_representation_diagram(figure_root / "19_v1_vs_v2_feature_representation.png")
    add(path, "V1 vs V2 feature representation", "Explain how v2 retains directional GLCM information.", [], "V2 expands 12 aggregate features to 84 total features.")
    path = plot_improvement_summary(v1_metrics, v2_metrics, figure_root / "20_v1_to_v2_improvement_summary.png")
    add(path, "V1 to V2 improvement summary", "Summarize the main test-set improvements.", [str(v1_root / "test_metrics.json"), str(v2_root / "test_metrics.json")], "V2 improves macro F1 and meningioma F1 while preserving v1 separately.")

    manifest_path = figure_root / "figure_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    return {
        "figure_dir": str(figure_root),
        "figure_count": len(manifest),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "v2_auc": v2_auc,
        "auc_comparison": auc_comparison,
        "v2_confidence": v2_conf,
        "confidence_comparison": confidence_comparison,
        "property_importance": prop.to_dict(orient="records"),
        "distance_importance": dist.to_dict(orient="records"),
        "angle_importance": angle.to_dict(orient="records"),
        "property_correlation": prop_corr.to_dict(orient="records"),
        "weighting_best_macro_f1": weighting.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-experiment-dir", type=Path, default=Path("reports/experiments/glcm_xgboost_v2"))
    parser.add_argument("--v1-experiment-dir", type=Path, default=Path("reports/experiments/glcm_xgboost_v1"))
    parser.add_argument("--v2-features", type=Path, default=Path("data/features/glcm_v2/glcm_v2_features.csv"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary = generate_glcm_v2_visualizations(
        v2_experiment_dir=args.v2_experiment_dir,
        v1_experiment_dir=args.v1_experiment_dir,
        v2_features=args.v2_features,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
