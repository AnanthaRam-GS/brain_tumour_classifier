"""
LBP + Logistic Regression
Brain Tumor Classification

Input:
    data/features/lbp/lbp_features.csv

Output:
    results/lbp_logistic_regression/
        metrics.json
        predictions.csv
        confusion_matrix.png
        metrics_comparison.png
        class_distribution.png
        roc_curve.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize


# ============================================================
# Configuration
# ============================================================

FEATURE_FILE = Path(
    "data/features/lbp/lbp_features.csv"
)

OUTPUT_DIR = Path(
    "results/lbp_logistic_regression"
)

RANDOM_STATE = 42

CLASS_NAMES = {
    1: "Meningioma",
    2: "Glioma",
    3: "Pituitary",
}


# ============================================================
# Utility functions
# ============================================================

def print_dataset_information(df: pd.DataFrame) -> None:
    """Print detailed information about the feature dataset."""

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(f"Total samples       : {len(df)}")
    print(f"Total columns       : {len(df.columns)}")

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("lbp_")
    ]

    print(f"LBP features        : {len(feature_columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nSplit distribution:")
    print(df["split"].value_counts().sort_index())

    print("\nClass distribution:")
    class_counts = df["label"].value_counts().sort_index()

    for label, count in class_counts.items():
        print(
            f"  {label} - "
            f"{CLASS_NAMES.get(label, 'Unknown')}: "
            f"{count}"
        )

    print("\nClass distribution by split:")

    distribution = pd.crosstab(
        df["split"],
        df["label"],
    )

    distribution = distribution.rename(
        columns=CLASS_NAMES
    )

    print(distribution)

    print("\nMissing values:")

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if missing.empty:
        print("  No missing values")
    else:
        print(missing)

    print("\nFeature statistics:")

    print(
        df[feature_columns].describe().round(4)
    )


def plot_class_distribution(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot class distribution."""

    counts = (
        df["label"]
        .map(CLASS_NAMES)
        .value_counts()
        .reindex(
            ["Meningioma", "Glioma", "Pituitary"]
        )
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        counts.index,
        counts.values,
    )

    plt.title(
        "LBP Dataset Class Distribution"
    )

    plt.xlabel("Tumor Class")
    plt.ylabel("Number of Samples")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray | None = None,
) -> dict:

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    if y_probability is not None:

        try:

            y_true_binary = label_binarize(
                y_true,
                classes=[1, 2, 3],
            )

            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true_binary,
                    y_probability,
                    multi_class="ovr",
                    average="macro",
                )
            )

        except ValueError:

            metrics["roc_auc_ovr_macro"] = None

    return metrics


def print_metrics(
    name: str,
    metrics: dict,
) -> None:

    print("\n" + "=" * 70)
    print(f"{name.upper()} RESULTS")
    print("=" * 70)

    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted Precision : "
        f"{metrics['weighted_precision']:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{metrics['weighted_recall']:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{metrics['weighted_f1']:.4f}"
    )

    if metrics.get("roc_auc_ovr_macro") is not None:

        print(
            f"ROC-AUC (OvR)      : "
            f"{metrics['roc_auc_ovr_macro']:.4f}"
        )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[1, 2, 3],
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(matrix)

    plt.title(
        "LBP + Logistic Regression "
        "Confusion Matrix"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        [0, 1, 2],
        ["Meningioma", "Glioma", "Pituitary"],
        rotation=30,
    )

    plt.yticks(
        [0, 1, 2],
        ["Meningioma", "Glioma", "Pituitary"],
    )

    for i in range(3):
        for j in range(3):

            plt.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def plot_metric_comparison(
    validation_metrics: dict,
    test_metrics: dict,
    output_path: Path,
) -> None:

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    display_names = [
        "Accuracy",
        "Balanced Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
    ]

    validation_values = [
        validation_metrics[name]
        for name in metric_names
    ]

    test_values = [
        test_metrics[name]
        for name in metric_names
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        validation_values,
        width,
        label="Validation",
    )

    plt.bar(
        x + width / 2,
        test_values,
        width,
        label="Test",
    )

    plt.xticks(
        x,
        display_names,
        rotation=30,
        ha="right",
    )

    plt.ylabel("Score")
    plt.title(
        "LBP + Logistic Regression "
        "Validation vs Test Performance"
    )

    plt.ylim(0, 1)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:

    y_binary = label_binarize(
        y_true,
        classes=[1, 2, 3],
    )

    plt.figure(figsize=(8, 6))

    for index, label in enumerate([1, 2, 3]):

        fpr, tpr, _ = roc_curve(
            y_binary[:, index],
            probabilities[:, index],
        )

        auc = roc_auc_score(
            y_binary[:, index],
            probabilities[:, index],
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{CLASS_NAMES[label]} (AUC={auc:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "LBP + Logistic Regression ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


# ============================================================
# Main training
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading LBP feature dataset...")

    df = pd.read_csv(FEATURE_FILE)

    print_dataset_information(df)

    # --------------------------------------------------------
    # Plot dataset information
    # --------------------------------------------------------

    plot_class_distribution(
        df,
        OUTPUT_DIR / "class_distribution.png",
    )

    # --------------------------------------------------------
    # Select feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("lbp_")
    ]

    if len(feature_columns) != 10:

        raise ValueError(
            f"Expected 10 LBP features, "
            f"found {len(feature_columns)}"
        )

    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    train_df = df[df["split"] == "train"].copy()

    validation_df = df[
        df["split"].isin(
            ["validation", "val"]
        )
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print("\n" + "=" * 70)
    print("SPLIT INFORMATION")
    print("=" * 70)

    print(
        f"Training samples   : {len(train_df)}"
    )

    print(
        f"Validation samples : {len(validation_df)}"
    )

    print(
        f"Test samples       : {len(test_df)}"
    )

    # --------------------------------------------------------
    # X and y
    # --------------------------------------------------------

    X_train = train_df[
        feature_columns
    ].to_numpy(dtype=float)

    y_train = train_df[
        "label"
    ].to_numpy()

    X_validation = validation_df[
        feature_columns
    ].to_numpy(dtype=float)

    y_validation = validation_df[
        "label"
    ].to_numpy()

    X_test = test_df[
        feature_columns
    ].to_numpy(dtype=float)

    y_test = test_df[
        "label"
    ].to_numpy()

    # --------------------------------------------------------
    # Imputer
    #
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_validation = imputer.transform(
        X_validation
    )

    X_test = imputer.transform(
        X_test
    )

    # --------------------------------------------------------
    # Standardization
    #
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_validation = scaler.transform(
        X_validation
    )

    X_test = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_predictions = model.predict(
        X_validation
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
        validation_probabilities,
    )

    print_metrics(
        "Validation",
        validation_metrics,
    )

    print("\nValidation classification report:")

    print(
        classification_report(
            y_validation,
            validation_predictions,
            labels=[1, 2, 3],
            target_names=[
                "Meningioma",
                "Glioma",
                "Pituitary",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )

    test_probabilities = model.predict_proba(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities,
    )

    print_metrics(
        "Test",
        test_metrics,
    )

    print("\nTest classification report:")

    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[1, 2, 3],
            target_names=[
                "Meningioma",
                "Glioma",
                "Pituitary",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\nTest confusion matrix:")

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[1, 2, 3],
    )

    print(matrix)

    plot_confusion_matrix(
        y_test,
        test_predictions,
        OUTPUT_DIR / "confusion_matrix.png",
    )

    # --------------------------------------------------------
    # Validation vs test plot
    # --------------------------------------------------------

    plot_metric_comparison(
        validation_metrics,
        test_metrics,
        OUTPUT_DIR / "metrics_comparison.png",
    )

    # --------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------

    plot_roc_curve(
        y_test,
        test_probabilities,
        OUTPUT_DIR / "roc_curve.png",
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions = test_df[
        [
            "sample_id",
            "patient_id",
            "label",
            "split",
        ]
    ].copy()

    predictions[
        "predicted_label"
    ] = test_predictions

    predictions[
        "predicted_class"
    ] = [
        CLASS_NAMES[label]
        for label in test_predictions
    ]

    for index, label in enumerate(
        [1, 2, 3]
    ):

        predictions[
            f"probability_{CLASS_NAMES[label].lower()}"
        ] = test_probabilities[:, index]

    predictions.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results = {
        "model": "LogisticRegression",
        "feature_set": "LBP",
        "feature_count": len(feature_columns),
        "classes": CLASS_NAMES,
        "random_state": RANDOM_STATE,
        "train_samples": len(train_df),
        "validation_samples": len(validation_df),
        "test_samples": len(test_df),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    with (
        OUTPUT_DIR / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Feature set : LBP ({len(feature_columns)} features)"
    )

    print(
        "Model       : Logistic Regression"
    )

    print(
        f"Train       : {len(train_df)} samples"
    )

    print(
        f"Validation  : {len(validation_df)} samples"
    )

    print(
        f"Test        : {len(test_df)} samples"
    )

    print(
        f"\nFinal Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Final Test Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    if test_metrics.get(
        "roc_auc_ovr_macro"
    ) is not None:

        print(
            f"Final Test ROC-AUC: "
            f"{test_metrics['roc_auc_ovr_macro']:.4f}"
        )

    print("\nGenerated files:")

    for file in sorted(
        OUTPUT_DIR.iterdir()
    ):
        print(
            f"  {file}"
        )


if __name__ == "__main__":
    main()