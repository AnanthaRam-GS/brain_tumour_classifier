"""Evaluation plots (matplotlib, Agg backend). PNGs saved into out_dir."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btdl.contracts import CLASS_NAMES_BY_INDEX


def plot_confusion_matrix(confusion_matrix, out_dir) -> list:
    """Two panels: raw counts, and row-normalized (recall-per-row). Returns written paths."""

    cm = np.asarray(confusion_matrix, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, data, title, fmt in (
        (axes[0], cm, "Confusion matrix (counts)", "{:.0f}"),
        (axes[1], normalized, "Confusion matrix (row-normalized)", "{:.2f}"),
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES_BY_INDEX, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES_BY_INDEX)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = Path(out_dir) / "confusion_matrix.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return [out_path]


def plot_roc_curves(roc_curves: dict, roc_auc: dict, out_dir) -> list:
    """3 classes on one axes, with macro AUC in the legend."""

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name in CLASS_NAMES_BY_INDEX:
        curve = roc_curves.get(name, {})
        fpr = curve.get("fpr", [])
        tpr = curve.get("tpr", [])
        class_auc = roc_auc.get("per_class", {}).get(name)
        label = f"{name} (AUC={class_auc:.3f})" if class_auc is not None else f"{name} (AUC=n/a)"
        if fpr and tpr:
            ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    macro_ovr = roc_auc.get("macro_ovr")
    title = f"ROC curves (macro OvR AUC={macro_ovr:.3f})" if macro_ovr is not None else "ROC curves (macro OvR AUC=n/a)"
    ax.set_title(title)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out_path = Path(out_dir) / "roc_curves.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return [out_path]


def plot_training_curves(history_csv_path, out_dir) -> list:
    """Loss panel (train_aug_loss, val_loss) and F1 panel (train_aug_macro_f1,
    val_macro_f1), both with the best epoch marked."""

    history = pd.read_csv(history_csv_path)
    best_epoch = None
    if "val_macro_f1" in history.columns and len(history) > 0:
        best_epoch = int(history.loc[history["val_macro_f1"].idxmax(), "epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss = axes[0]
    ax_loss.plot(history["epoch"], history["train_aug_loss"], label="train_aug_loss")
    ax_loss.plot(history["epoch"], history["val_loss"], label="val_loss")
    if best_epoch is not None:
        ax_loss.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best epoch={best_epoch}")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(fontsize=8)

    ax_f1 = axes[1]
    ax_f1.plot(history["epoch"], history["train_aug_macro_f1"], label="train_aug_macro_f1")
    ax_f1.plot(history["epoch"], history["val_macro_f1"], label="val_macro_f1")
    if best_epoch is not None:
        ax_f1.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best epoch={best_epoch}")
    ax_f1.set_xlabel("epoch")
    ax_f1.set_ylabel("macro F1")
    ax_f1.set_title("Macro F1")
    ax_f1.legend(fontsize=8)

    fig.tight_layout()
    out_path = Path(out_dir) / "training_curves.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return [out_path]

