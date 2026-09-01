# Wavelet + Random Forest Analysis Plots

Phase 1 baseline experiment `wavelet_rf_v1` (28 discrete-wavelet-transform features,
Random Forest classifier, patient-level train/val/test split). All plots are derived
from the already-audited artifacts in `reports/experiments/wavelet_rf_v1/` and
`data/splits/patient_split.csv`; no retraining or re-evaluation was performed.

## 1. Confusion Matrix
Filename: `01_confusion_matrix.png`
What it shows: 3x3 heatmap of actual vs. predicted class on the held-out test set
(n=498), with raw counts in each cell.
Key interpretation: Glioma is classified cleanly (200/222 correct). Meningioma is the
weakest class (69/117 correct), most often confused with Pituitary (39 misclassified
as Pituitary). Pituitary is also confused with both other classes (22 as Meningioma,
32 as Glioma).

## 2. Per-Class Metrics
Filename: `02_per_class_metrics.png`
What it shows: Grouped bar chart of precision, recall, and F1 for Meningioma, Glioma,
and Pituitary, read directly from `metrics.json`.
Key interpretation: Glioma has the best precision/recall/F1 (0.83 / 0.90 / 0.86).
Meningioma has the lowest recall (0.59), consistent with the confusion-matrix pattern
of Meningioma samples being predicted as Pituitary.

## 3. Top 10 Feature Importance
Filename: `03_top10_feature_importance.png`
What it shows: Horizontal bar chart of the 10 highest impurity-based (Gini) feature
importances from the trained Random Forest.
Key interpretation: The top 3 features (`dwt_l2_ll_std`, `dwt_l2_ll_energy`,
`dwt_l2_ll_mean`) are all statistics of the level-2 LL (approximation) subband and
together account for roughly 38% of total importance, far ahead of any other feature.

## 4. All Feature Importance
Filename: `04_all_feature_importance.png`
What it shows: Horizontal bar chart of all 28 DWT features ranked by importance.
Key interpretation: Importance drops off sharply after the top 3 L2-LL features; the
remaining 25 features are dominated by entropy statistics across various subbands,
each contributing a small, fairly even share.

## 5. Class Distribution
Filename: `05_class_distribution.png`
What it shows: Sample counts per class (Meningioma / Glioma / Pituitary) for each of
the Train, Validation, and Test splits, computed directly from
`data/splits/patient_split.csv`.
Key interpretation: The dataset is imbalanced (overall Meningioma 708 / Glioma 1426 /
Pituitary 930), and this imbalance is preserved consistently across all three splits
(patient-level split), which the model's `class_weight="balanced"` setting is intended
to address.

## 6. Validation-Based Hyperparameter Search
Filename: `06_validation_hyperparameter_search.png`
What it shows: All 30 candidate hyperparameter configurations ranked by
validation-set Macro F1 (x = rank, y = validation Macro F1); the selected candidate is
highlighted with its hyperparameters annotated. Only validation metrics are used —
no test-set data appears in this plot.
Key interpretation: The selected configuration
(`n_estimators=100, max_depth=None, min_samples_split=5, min_samples_leaf=2,
max_features=log2`, val macro F1=0.6354) ranks 1st among all 30 candidates searched,
confirming hyperparameter selection was legitimately validation-driven and the test
set was untouched until final evaluation.

## 7. ROC Curves
Filename: `07_roc_curves.png`
What it shows: One-vs-rest ROC curves for the 3 classes on the test set, computed
independently from `predictions.csv` probability columns
(`prob_meningioma`, `prob_glioma`, `prob_pituitary`), with a diagonal chance baseline.
Key interpretation: Independently computed macro AUC = 0.9058, matching
`metrics.json`'s reported `roc_auc.macro_ovr` = 0.9058 to 4 decimal places. Glioma has
the highest AUC (0.950), Pituitary the lowest (0.868), consistent with the per-class
precision/recall pattern.

## 8. Overall Metrics
Filename: `08_overall_metrics.png`
What it shows: Bar chart of accuracy, balanced accuracy, macro/weighted
precision/recall/F1, and macro ROC-AUC, all read directly from `metrics.json`.
Key interpretation: Overall test accuracy is 0.751 (balanced accuracy 0.717), with
macro F1 of 0.722 reflecting the drag from the weaker Meningioma and Pituitary
classes relative to Glioma.

## Feature Importance Summary (`feature_importance_summary.csv`)
What it shows: The 28 feature importances aggregated two ways — by wavelet subband
(L1-LH/HL/HH, L2-LL/LH/HL/HH) and by statistic type (mean/std/energy/entropy) — with
total and mean importance per group. Both groupings sum to 1.0 (up to floating-point
rounding).

Top feature: `dwt_l2_ll_std` (importance 0.1506).
Top 3 features: `dwt_l2_ll_std` (0.1506), `dwt_l2_ll_energy` (0.1200),
`dwt_l2_ll_mean` (0.1143).
Dominant subband: L2-LL (level-2 approximation subband), total importance 0.4111 —
more than 4x any other subband.
Dominant statistic: `std` (standard deviation), total importance 0.2870, narrowly
ahead of `entropy` (0.2575) and `energy` (0.2529); `mean` is lowest (0.2026).

This is Phase 1 baseline analysis only; no clinical or diagnostic claims are implied.
