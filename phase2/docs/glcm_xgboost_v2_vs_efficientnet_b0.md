# GLCM + XGBoost vs EfficientNet-B0: Results Comparison

**Phase 1 model:** `glcm_xgboost_v2`: GLCM texture features (84) + XGBoost (the study's official baseline, `study.yaml`'s `baseline_reference`)
**Phase 2 model:** `efficientnet_b0`: ImageNet-pretrained EfficientNet-B0, full fine-tuning, 3-class head
**Task:** classify brain tumour MRI slices as meningioma, glioma or pituitary
**Source numbers:** `reports/experiments/glcm_xgboost_v2/validation_metrics.json`, `reports/experiments/glcm_xgboost_v2/test_metrics.json`, `phase2/artifacts/results/efficientnet_b0/seed{42,43,44}/eval/test/metrics.json`, `phase2/runs/efficientnet_b0/lr3e-04_seed{42,43,44}/eval/val/metrics.json`

> Both models use the same patient-level split (`data/splits/patient_split.csv`): train 2,120, val 446 images from 34 patients, test 498 images from 37 patients. EfficientNet-B0's official `final_test` has been run (all 3 final seeds, `phase2/artifacts/test_access_log.csv`), so the test-split comparison below is the real, official result.

---

## 1. Validation split (446 images, 34 patients)

| Metric | GLCM + XGBoost | EfficientNet-B0 (mean of 3 seeds) |
|---|---|---|
| Accuracy | 0.6435 | 0.9559 ± 0.0079 |
| Balanced accuracy | 0.5884 | 0.9508 ± 0.0058 |
| Macro F1 | 0.5893 | 0.9494 ± 0.0083 |
| Weighted F1 | 0.6425 | 0.9560 ± 0.0078 |
| ROC-AUC (macro, one-vs-rest) | 0.8071 | 0.9906 ± 0.0008 |

| Class (val F1) | GLCM + XGBoost | EfficientNet-B0 (mean of 3 seeds) |
|---|---|---|
| Meningioma | 0.411 | 0.928 ± 0.009 |
| Glioma | 0.735 | 0.969 ± 0.006 |
| Pituitary | 0.593 | 0.951 ± 0.010 |

### 1.1 EfficientNet-B0 validation, per seed (lr = 3e-4)

| Seed | Val macro-F1 | Best epoch |
|---|---|---|
| 42 | 0.9580 | 22 |
| 43 | 0.9415 | 2 |
| 44 | 0.9487 | 5 |
| **Mean ± std** | **0.9494 ± 0.0083** | |

---

## 2. Test split (498 images, 37 patients) -- OFFICIAL RESULT

| Metric | GLCM + XGBoost | EfficientNet-B0 (mean of 3 seeds) |
|---|---|---|
| Accuracy | 0.6486 | 0.9411 ± 0.0162 |
| Balanced accuracy | 0.5907 | 0.9316 ± 0.0220 |
| Macro F1 | 0.5891 | 0.9371 ± 0.0183 |
| Weighted F1 | 0.6290 | 0.9407 ± 0.0166 |
| ROC-AUC (macro, one-vs-rest) | n/a (not recorded) | 0.9910 ± 0.0056 |

EfficientNet-B0 is ahead by **+0.35 macro-F1** on the test split (0.937 vs 0.589). Its weakest seed (0.925) is still far above GLCM+XGBoost's test macro-F1 (0.589).

### 2.1 EfficientNet-B0 test, per seed (lr = 3e-4)

| Seed | Test macro-F1 | Accuracy | Balanced accuracy |
|---|---|---|---|
| 42 | 0.9582 | 0.9598 | 0.9570 |
| 43 | 0.9279 | 0.9317 | 0.9200 |
| 44 | 0.9252 | 0.9317 | 0.9179 |
| **Mean ± std** | **0.9371 ± 0.0183** | **0.9411 ± 0.0162** | **0.9316 ± 0.0220** |

| Class (test F1) | GLCM + XGBoost | EfficientNet-B0 (mean of 3 seeds) |
|---|---|---|
| Meningioma | 0.355 | 0.916 ± 0.022 |
| Glioma | 0.748 | 0.952 ± 0.008 |
| Pituitary | 0.665 | 0.943 ± 0.028 |

### 2.2 Confusion matrices (test split, rows = true class, columns = predicted)

**GLCM + XGBoost**

| True \ Predicted | Meningioma | Glioma | Pituitary |
|---|---|---|---|
| Meningioma | 33 | 54 | 30 |
| Glioma | 17 | 187 | 18 |
| Pituitary | 19 | 37 | 103 |

**EfficientNet-B0 (summed over 3 seeds, 498 × 3 = 1,494 predictions)**

| True \ Predicted | Meningioma | Glioma | Pituitary |
|---|---|---|---|
| Meningioma | 314 | 25 | 12 |
| Glioma | 10 | 654 | 2 |
| Pituitary | 10 | 29 | 438 |

---

## 3. Setup differences

| | GLCM + XGBoost | EfficientNet-B0 |
|---|---|---|
| Input | GLCM texture features (84) from the tumour crop | Unmasked tumour crop, 224×224, gray copied into 3 channels, ImageNet normalisation |
| Features | Hand-crafted: gray-level co-occurrence matrix statistics | Learned end-to-end from ImageNet-pretrained weights |
| Model size | Gradient-boosted trees (xgboost) | 4,011,391 parameters, all trained |
| Hyperparameter selection | 108 candidates on validation macro-F1 | Learning rate in {1e-4, 3e-4, 1e-3}, seed 42, on validation macro-F1 |
| Seeds | 1 (seed 42) | 3 (seeds 42, 43, 44) |

---

## 4. How to read these numbers

1. **Compare like with like.** Validation vs validation (Section 1), test vs test (Section 2) -- both already the official, final numbers for both models.
2. **The gap is large on both splits.** GLCM + XGBoost's macro-F1 sits around 0.59 on both val and test; EfficientNet-B0 clears 0.92 on every individual seed, on both splits.
3. **Meningioma is GLCM + XGBoost's hardest class** (F1 0.355-0.411), most often confused with pituitary. EfficientNet-B0's weakest class is also meningioma (F1 ~0.92-0.93), but the gap to its other classes is far smaller.
