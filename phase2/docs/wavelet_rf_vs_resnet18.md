# Wavelet + Random Forest vs ResNet18: Results Comparison

**Phase 1 model:** `wavelet_rf_v1`: 2D DWT wavelet features (28) + Random Forest
**Phase 2 model:** `resnet18`: ImageNet-pretrained, full fine-tuning, 3-class head
**Task:** classify brain tumour MRI slices as meningioma, glioma or pituitary
**Related notes:** `reports/resnet18_review.md`, `reports/resnet18_review_qa.md`, `reports/experiments/wavelet_rf_v1/plots/README.md`

> **Status of the numbers in this document**
> - **ResNet18 has validation results only.** The official test evaluation (`final_test`) has not been run yet. The trained checkpoints are not on this machine.
> - The like-for-like comparison is therefore on the **validation split** (Section 2). Section 4 lists the ResNet18 test cells still to fill in.
> - Both models use the **same patient-level split** (`data/splits/patient_split.csv`): train 2,120, val 446 images from 34 patients, test 498 images from 37 patients.

---

## 1. Sources

| Numbers | Source |
|---|---|
| Wavelet + RF, test | `reports/experiments/wavelet_rf_v1/metrics.json` |
| Wavelet + RF, validation | Recomputed by refitting the selected RF on train. The run is deterministic (`random_state=42`, `n_jobs=1`) and the refit reproduces the committed test metrics exactly. Its validation macro-F1 (0.6354) matches `hyperparameter_search.json`. |
| Wavelet + RF, 95% CIs | Patient-level bootstrap over 2,000 resamples (resample patients, keep all their slices). This is the same scheme Phase 2 uses. |
| ResNet18, validation | `phase2/artifacts/selection/resnet18.json`, `phase2/artifacts/frozen/resnet18.json` |

---

## 2. Head-to-head on the validation split (like-for-like)

| Metric (val, 446 images) | Wavelet + RF | ResNet18 (mean of 3 seeds) |
|---|---|---|
| **Macro F1** | **0.6354** | **0.9455 ± 0.0032** |
| Macro F1, 95% CI (patient bootstrap) | 0.487 – 0.736 | ⟨not computed⟩ |
| Accuracy | 0.6704 | ⟨fill in⟩ |
| Balanced accuracy | 0.6467 | ⟨fill in⟩ |
| Weighted F1 | 0.6762 | ⟨fill in⟩ |
| ROC-AUC (macro, one-vs-rest) | 0.8492 | ⟨fill in⟩ |

ResNet18 is ahead by **+0.31 macro-F1**. Its weakest seed (0.9423) is well above the top of the RF's 95% interval (0.736), so the gap is not noise.

The ResNet18 cells marked ⟨fill in⟩ are in the run folders `phase2/runs/resnet18/lr3e-04_seed{42,43,44}/` and in section 8 of the walkthrough notebook. They are not in the committed JSON.

### 2.1 ResNet18 validation, per seed (lr = 3e-4)

| Seed | Val macro-F1 | Best epoch |
|---|---|---|
| 42 | 0.9453 | 26 |
| 43 | 0.9488 | 20 |
| 44 | 0.9423 | 19 |
| **Mean ± std** | **0.9455 ± 0.0032** | |

### 2.2 Hyperparameter selection (both on validation macro-F1)

| | Wavelet + RF | ResNet18 |
|---|---|---|
| What was searched | 30 RF settings (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`) | Learning rate in {1e-4, 3e-4, 1e-3}, seed 42 |
| Range of val macro-F1 | 0.607 – 0.635 | 0.9085 – 0.9453 |
| Selected | `n_estimators=100, max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features="log2"` (val macro-F1 0.6354) | lr = 3e-4 (val macro-F1 0.9453) |

---

## 3. Wavelet + RF: detailed results

### 3.1 Overall

| Metric | Validation (446 images, 34 patients) | Test (498 images, 37 patients) |
|---|---|---|
| Macro F1 | 0.6354 | **0.7224** |
| Macro F1, 95% CI (patient bootstrap) | 0.487 – 0.736 | 0.641 – 0.785 |
| Accuracy | 0.6704 | 0.7510 |
| Balanced accuracy | 0.6467 | 0.7170 |
| Weighted F1 | 0.6762 | 0.7468 |
| ROC-AUC (macro, one-vs-rest) | 0.8492 | 0.9058 |

### 3.2 Per class

| Class | Val precision / recall / F1 (support) | Test precision / recall / F1 (support) |
|---|---|---|
| Meningioma | 0.473 / 0.595 / **0.527** (74) | 0.711 / 0.590 / **0.645** (117) |
| Glioma | 0.823 / 0.744 / **0.782** (219) | 0.830 / 0.901 / **0.864** (222) |
| Pituitary | 0.594 / 0.601 / **0.597** (153) | 0.656 / 0.660 / **0.658** (159) |

### 3.3 Confusion matrices

Rows are the true class; columns are the predicted class.

**Validation**

| True \ Predicted | Meningioma | Glioma | Pituitary |
|---|---|---|---|
| Meningioma | 44 | 1 | 29 |
| Glioma | 22 | 163 | 34 |
| Pituitary | 27 | 34 | 92 |

**Test**

| True \ Predicted | Meningioma | Glioma | Pituitary |
|---|---|---|---|
| Meningioma | 69 | 9 | 39 |
| Glioma | 6 | 200 | 16 |
| Pituitary | 22 | 32 | 105 |

### 3.4 What the RF relies on

From `reports/experiments/wavelet_rf_v1/feature_importance.csv` (Gini importance):

- The top 3 features are all level-2 approximation (L2-LL) statistics: `dwt_l2_ll_std` (0.151), `dwt_l2_ll_energy` (0.120) and `dwt_l2_ll_mean` (0.114). Together they carry about **38%** of the total importance.
- The L2-LL subband as a whole carries **0.411**, more than 4× any other subband.
- The RF therefore relies mostly on the coarse intensity and contrast of the tumour region, not on fine texture detail.

---

## 4. Test split (official result): ResNet18 pending

| Metric (test, 498 images) | Wavelet + RF | ResNet18 seed 42 | seed 43 | seed 44 | ResNet18 mean ± std |
|---|---|---|---|---|---|
| Macro F1 | 0.7224 | ⟨fill in⟩ | ⟨fill in⟩ | ⟨fill in⟩ | ⟨fill in⟩ |
| Macro F1, 95% CI | 0.641 – 0.785 | ⟨fill in⟩ | ⟨fill in⟩ | ⟨fill in⟩ | n/a |
| Accuracy | 0.7510 | ⟨fill in⟩ | ⟨fill in⟩ | ⟨fill in⟩ | ⟨fill in⟩ |
| Balanced accuracy | 0.7170 | ⟨fill in⟩ | ⟨fill in⟩ | ⟨fill in⟩ | ⟨fill in⟩ |
| ROC-AUC (macro, one-vs-rest) | 0.9058 | ⟨fill in⟩ | ⟨fill in⟩ | ⟨fill in⟩ | ⟨fill in⟩ |

| Class (test F1) | Wavelet + RF | ResNet18 (mean of 3 seeds) |
|---|---|---|
| Meningioma | 0.645 | ⟨fill in⟩ |
| Glioma | 0.864 | ⟨fill in⟩ |
| Pituitary | 0.658 | ⟨fill in⟩ |

To produce the ResNet18 test numbers, run this on the machine that has `phase2/runs/resnet18/` (see `reports/resnet18_review_qa.md`, Section 1.4):

```
.venv/bin/python -m btdl.cli.final_test --run-dir runs/resnet18/lr3e-04_seed$S --confirm-final-test
```

---

## 5. Setup differences

| | Wavelet + RF | ResNet18 |
|---|---|---|
| Input | Tumour crop with the background masked out, 128×128, grayscale | Unmasked tumour crop, 224×224, gray copied into 3 channels, ImageNet normalisation |
| Features | 28 hand-crafted: db2 wavelet, 2 levels, 7 subbands × mean / std / energy / entropy | Learned end-to-end from ImageNet-pretrained weights |
| Model size | 100 trees | 11,178,051 parameters, all trained |
| Class imbalance handling | `class_weight="balanced"` | Class-weighted cross-entropy loss |
| Augmentation | None | Yes, train split only |
| Final model trained on | Train only | Train only |
| Seeds | 1 (seed 42) | 3 (seeds 42, 43, 44) |

---

## 6. How to read these numbers

1. **Compare like with like.** Compare validation with validation now, and test with test once ResNet18's `final_test` has run. Do not put the RF's test macro-F1 (0.722) next to ResNet18's validation macro-F1 (0.945).
2. **The RF scores lower on validation than on test** (0.635 vs 0.722). The validation patients seem harder for it: meningioma F1 is 0.527 on validation against 0.645 on test. If anything, comparing on validation makes the RF look slightly worse than it would on test.
3. **Meningioma is the RF's hardest class.** It is most often confused with pituitary: 29 of 74 on validation and 39 of 117 on test.
4. **The gap is large and holds up under the uncertainty estimates.** The RF's 95% validation interval tops out at 0.736, while all three ResNet18 seeds score above 0.94. Learned features clearly beat hand-crafted wavelet statistics on this task.
5. **Correction to an existing caveat:** `reports/resnet18_review_qa.md` (Section 3.4) says "Phase 1 did retrain [on train+val]". That is not true for this model: `src/models/random_forest.py` trains the final RF on the train split only, the same as ResNet18.
