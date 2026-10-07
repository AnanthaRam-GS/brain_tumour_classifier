# Phase 2 Repository Archaeology Audit

Date: 2026-10-07
Scope: read-only audit of the Phase 1 classical-ML codebase ahead of Phase 2 (deep learning).
Repository root: `Computer_Vision/Project` (git remote `AnanthaRam-GS/brain_tumour_classifier`, branch `main`).

No files were created, modified, moved, or deleted by this audit other than this report.

---

## 1. Executive summary

**Readiness verdict: CONDITIONALLY READY.** The Phase 1 foundation (raw loading, patient-level
split, shared sample/ROI/feature-table/evaluation contracts, GLCM v2 + XGBoost pipeline) is
disciplined, contract-driven, well tested (234/234 tests pass, 19.5s), and the headline GLCM v2
metrics quoted in the project brief were verified byte-for-byte against
`reports/experiments/glcm_xgboost_v2/experiment_summary.json` and `test_confusion_matrix.csv`.
Patient-level leakage was independently verified to be zero across train/val/test, and the
2,120/446/498 sample split and 162/34/37 patient split were reproduced exactly from the persisted
split file. However, **none of the Phase 2 deep-learning infrastructure exists yet** — no
`torch`/`torchvision` (confirmed not installed in the active venv), no `configs/dl/`, no
`src/training/`, no `src/evaluation/dl_*`, no device/seed/checkpoint modules. Phase 2 must be
built from scratch, reusing the data/split/ROI/metrics layers with modification.

### Top 5 risks
1. **Unresolved git merge with a code regression staged.** `git status` shows the repo is mid-merge
   ("All conflicts fixed but you are still merging"). The staged versions of
   `src/features/gabor.py` and `src/models/svm.py` are *less* defensive than the current `HEAD`
   versions — they drop type hints, docstrings, and critical input-validation `raise` branches
   (e.g. 2D/finite-value checks in `extract_gabor_features`). Committing this merge as-is would
   silently reintroduce a weaker version of shipped Phase 1 code. This must be resolved/reviewed
   before any Phase 2 branch is cut from `main`.
2. **No deep-learning dependencies installed.** `torch`/`torchvision` are absent from the venv;
   device handling (CUDA→MPS→CPU), pretrained-weights API choice, and numpy-2.x/torch
   compatibility are all unverified and must be checked once installed.
3. **ROI spec mismatch with the Phase 2 plan.** The existing `prepare_tumor_roi()` produces a
   **128×128** standardized view (not 224×224) and exposes both a masked (background-zeroed) and
   unmasked crop; the Phase 2 plan's crop→224 pipeline does not specify masking out background.
   Padding is computed as `ceil(max(h,w) * 0.10)` per side on the *tight-bbox longest side*, not a
   simple "10% of the box." Phase 2 code must decide explicitly whether to reuse masking and must
   change the resize target.
4. **Stale README/onboarding claims.** The README states "Expected current baseline: 139 passed,"
   but the suite now has 234 tests, all passing — harmless but indicates documentation is not kept
   in lock-step with commits, raising the risk that other README claims (e.g. config contracts) are
   similarly stale.
5. **No reusable generic trainer/checkpointing/device code.** `src/models/training.py` is classical
   sklearn-pipeline-shaped (impute→scale, `.fit`) and has no notion of epochs, batches, device
   placement, or checkpoint metadata — Phase 2's trainer, early stopping, and checkpoint modules
   must be built new; only the *evaluation metrics contract* (`CLASS_ORDER`, per-class
   precision/recall/F1, confusion matrix, ROC-AUC shape) is directly reusable.

---

## 2. Repository architecture map

```
Project/
├── 1512427/                     # raw Figshare MATLAB dataset (885M, git-ignored, present locally)
├── data/
│   ├── processed/samples/       # 3064 canonical NPZ files (image_raw, image_normalized, tumor_mask,
│   │                             tumor_border, label, patient_id, sample_id) — git-ignored, present
│   ├── processed/{images,masks,overlays,audit_overlays}/  # PNG visual derivatives, git-ignored
│   ├── splits/patient_split.csv, split_metadata.json      # the ONLY split files tracked in git
│   └── features/{glcm,glcm_v2,gabor,wavelet}/              # feature CSVs, git-ignored
├── configs/
│   ├── paths.yaml               # DEAD — not referenced anywhere in src/ or tests/
│   ├── features/{glcm,glcm_v2,gabor,lbp,wavelet,hog}.yaml  # frozen per-algorithm parameter configs
│   └── models/{xgboost,xgboost_glcm_v2,random_forest}.yaml
├── src/
│   ├── data/            mat_loader.py, convert_dataset.py, create_patient_split.py,
│   │                     feature_dataset.py, audit_dataset.py, visualize_samples.py
│   ├── preprocessing/    normalization.py, roi.py
│   ├── features/         config.py, feature_table.py, glcm.py, glcm_v2.py, gabor.py, lbp.py, wavelet.py
│   │                     (no hog.py despite configs/features/hog.yaml existing — HOG unimplemented)
│   ├── models/           training.py, xgboost_model.py, run_xgboost.py, run_xgboost_glcm_v2.py,
│   │                     svm.py, svm_gabor.py, random_forest.py, random_forest_wavelet.py,
│   │                     lbp_logistic_regression.py
│   ├── evaluation/       metrics.py, results.py, visualize_results.py, visualize_experiment.py,
│   │                     visualize_glcm_v2.py, run_glcm_xgboost_visualizations.py,
│   │                     run_glcm_v2_visualizations.py
│   └── demo/             pipeline_registry.py, inference.py, sample_selection.py,
│                          visualize_demo.py, run_review_demo.py
├── tests/                23 test files, 234 tests total
├── reports/experiments/  per-experiment metrics.json/predictions.csv/confusion_matrix.csv/figures
├── reports/demo/         git-ignored demo run outputs
└── models/               git-ignored fitted model bundles (joblib/.json), not present in this checkout
```

No notebooks are present (`notebooks/` only ever held a `.gitkeep`, later removed). No Jupyter-only
logic exists; all Phase 1 logic lives in importable `src/` modules with CLI `main()` entry points.
`src` is **not** an installable package (no `pyproject.toml`/`setup.py`); it relies on being run as
`python -m src.xxx` from the repo root, i.e. implicit `sys.path` resolution via the current working
directory, not an editable install. There is no `pyproject.toml`, `setup.cfg`, or lockfile — only
`requirements.txt` with unpinned versions.

---

## 3. Existing Phase 1 pipeline map — GLCM v2 + XGBoost (raw file → metric)

1. **Raw load**: `src/data/mat_loader.py:load_mat_sample` reads one `cjdata` struct via `scipy.io.loadmat`
   with an `h5py` fallback for MATLAB v7.3 files (`_load_with_h5py`, `mat_loader.py:155-174`). HDF5
   numeric arrays are explicitly transposed (`array = array.transpose()`, line 151) to correct
   MATLAB's reversed HDF5 dimension order. PID is decoded from char/uint arrays via `_to_text`
   (lines 64-89), handling `char` MATLAB class and reconstructing from uint16 codes.
2. **Conversion**: `src/data/convert_dataset.py` (not fully re-read line-by-line in this audit, but
   its CLI/README contract was verified) writes one NPZ per sample with `image_raw` (float32,
   untouched) and `image_normalized` (robust percentile-normalized via
   `src/preprocessing/normalization.py:robust_foreground_percentile_normalize`, lines 10-70 — 1st/99th
   percentile of foreground-only pixels, background pixels forced to exactly 0). No resizing occurs
   at conversion time (native 512×512 / 256×256 preserved).
3. **Split**: `src/data/create_patient_split.py:assign_patient_splits` (lines 116-146) does a
   per-class `rng.permutation` (seed 42) over unique `patient_id`s with 70/15/15 ratios, writes
   `data/splits/patient_split.csv` + `split_metadata.json` atomically (`_atomic_csv`/`_atomic_json`).
   `validate_split` (lines 175-253) asserts zero patient overlap and full class coverage per split
   before the file is ever written.
4. **Sample access contract**: `src/data/feature_dataset.py:load_phase1_sample`/`iter_phase1_samples`
   (lines 196-277) join the split CSV with the NPZ file, re-validating label/patient_id/sample_id
   agreement between the split file and the NPZ contents on every load (lines 229-242) — this is the
   leakage/consistency guard all feature code goes through.
5. **ROI**: `src/preprocessing/roi.py:prepare_tumor_roi` (lines 194-250) → tight mask bbox
   (`find_mask_bbox`) → 10%-padded centered square (`make_square_bbox`) → zero-pad crop
   (`crop_and_pad`) → native masked/unmasked views + bilinear/nearest-resized 128×128 views.
6. **Feature extraction**: `src/features/glcm_v2.py:extract_glcm_v2_dataset` (lines 165-229) calls
   `prepare_tumor_roi` then `extract_glcm_v2_features_from_roi` → `src/features/glcm.py` primitives
   (`quantize_glcm_image`, `build_masked_glcm`, `compute_glcm_properties`) at gray_levels=32,
   distances {1,2,4}, angles {0,45,90,135}, 6 properties → 72 directional + 12 mean/std = **84
   columns**, written via `src/features/feature_table.py:write_feature_table` to
   `data/features/glcm_v2/glcm_v2_features.csv` (validated against the canonical split on write).
7. **Model search**: `src/models/run_xgboost_glcm_v2.py:run_experiment` (lines 405-533) loads
   `configs/models/xgboost_glcm_v2.yaml`, builds a 108-candidate grid (4 weighting modes ×
   `n_estimators` × `max_depth` × `learning_rate`, `candidate_parameter_grid` lines 111-129), fits
   each candidate on **TRAIN only** (`evaluate_candidate`, lines 185-237; preprocessor —
   median-imputation-only, no scaling — fit via `fit_preprocessor_for(..., include_validation=False)`)
   and scores on **VAL only**. `select_best_candidate` (lines 240-255) picks by
   `(macro_f1 desc, balanced_accuracy desc, meningioma_f1 desc, accuracy desc, log_loss asc, …)`.
8. **Promotion + refit**: `decide_promotion` (lines 272-290) compares against the frozen v1 validation
   baseline (`reports/experiments/glcm_xgboost_v1/validation_metrics.json`); v2 beat v1 by macro-F1
   +0.0512 → `PROMOTE`. `_retrain_and_test` (lines 325-402) refits the winning hyperparameters on
   **train+val** (preprocessor refit with `include_validation=True`) and evaluates **once** on TEST.
9. **Evaluation**: `src/evaluation/results.py:build_result_dict` → `src/evaluation/metrics.py:
   evaluate_predictions` (fixed `CLASS_ORDER=[1,2,3]`) plus an ad-hoc `log_loss` line added directly
   in `run_xgboost_glcm_v2.py:225,384` (metrics.py itself has **no** log-loss function — it was
   bolted on in the experiment script, not the shared contract).
10. **Verification against the project brief**: `reports/experiments/glcm_xgboost_v2/experiment_summary.json`
    `test_metrics` = `{accuracy: 0.64859, balanced_accuracy: 0.59073, macro_f1: 0.58912,
    weighted_f1: 0.62898, log_loss: 0.86964, roc_auc.macro_ovr: 0.80715}` and
    `test_confusion_matrix.csv` = `[[33,54,30],[17,187,18],[19,37,103]]` — **all match the quoted
    brief exactly**, confirming the reported numbers correspond to a real, re-inspectable run.

---

## 4. Dataset / dataflow map

Raw `.mat` (cjdata struct: label, PID, image, tumorBorder, tumorMask) →
`mat_loader.load_mat_sample` (validated `BrainTumourSample`) →
`convert_dataset` → `data/processed/samples/{id}.npz` (image_raw, image_normalized, tumor_mask,
tumor_border, label, patient_id, sample_id) →
`create_patient_split` → `data/splits/patient_split.csv` (sample_id, patient_id, label, split) →
`feature_dataset.iter_phase1_samples` (cross-validates split vs NPZ) →
`preprocessing/roi.prepare_tumor_roi` (native + 128×128 masked/unmasked ROI) →
per-algorithm feature CSV (`data/features/<algo>/<algo>_features.csv`, schema
`sample_id,patient_id,label,split,<feature columns>`) →
`models/training.prepare_feature_matrices` (split column is sole authority for train/val/test) →
model-specific run script → `evaluation/results.build_result_dict` →
`reports/experiments/<name>/{metrics.json,predictions.csv,confusion_matrix.csv,experiment_metadata.json}`.

Raw dataset is present locally (885M, 3,064 `.mat` files across 4 numbered subdirectories +
`cvind.mat` + README). `data/processed/samples/` contains exactly 3,064 NPZ files (verified by
`ls | wc -l`). `cvind.mat` is explicitly **not** used for the Phase 1 split (README states it was
inspected and found patient-disjoint, kept only as reference).

---

## 5. Split and data-leakage assessment (verified numbers)

Verified directly from `data/splits/patient_split.csv` and `split_metadata.json` (commands in
Appendix):

| | train | val | test | total |
|---|---|---|---|---|
| samples | 2120 | 446 | 498 | 3064 |
| patients | 162 | 34 | 37 | 233 |
| meningioma samples | 517 | 74 | 117 | 708 |
| glioma samples | 985 | 219 | 222 | 1426 |
| pituitary samples | 618 | 153 | 159 | 930 |

- **This reproduces the quoted 2,120/446/498 and 162/34/37 split exactly.**
- PID intersection sizes, computed directly from the CSV: `train∩val = 0`, `train∩test = 0`,
  `val∩test = 0`. **Zero patient leakage confirmed empirically.**
- The split is **only reproducible from the persisted CSV in the normal workflow** — README
  explicitly instructs teammates *not* to regenerate it ("Do not regenerate the Phase 1 split
  during normal teammate setup"). It *is* deterministically regenerable by maintainers via
  `python -m src.data.create_patient_split` (fixed seed 42, `np.random.default_rng(42)`), given an
  identical `data/processed/manifest.csv` — this was not independently re-run in this audit (would
  write files; out of scope for a read-only audit) but the algorithm (`assign_patient_splits`) is
  a pure deterministic function of (manifest, seed, ratios).
- `grep` across `src/` for `train_test_split`, `KFold`, uncontrolled `shuffle=True` found **no**
  slice-level random split anywhere — the only `np.random`/seeded calls are: patient split (seed
  42), demo sample selection (`sample_selection.py`, seed 42 default, explicit `random_selection`
  opt-in for non-deterministic demo runs), dataset audit visualization sampling, and model seeds
  (SVM/RandomForest/XGBoost/LogReg `random_state`/`random_seed`, all sourced from frozen YAML
  configs requiring `random_seed == 42` in `xgboost_model.py:82-83`).
- Preprocessing-fit leakage check: `fit_feature_preprocessor` / `fit_preprocessor_for` are always
  called with `X_train` only (candidate search) or `vstack([X_train, X_val])` only at the *final*
  refit stage (never including test) — consistent with the documented contract. Class weights in
  `compute_sample_weights` (`run_xgboost_glcm_v2.py:132-167`) are computed from the metadata/`y`
  passed in, which at call sites is always `matrices.train_metadata`/`y_train` (search) or the
  train+val concatenation (final refit) — never test.

---

## 6. Existing preprocessing assessment, incl. ROI vs Phase 2 spec

**Intensity normalization** (`src/preprocessing/normalization.py:robust_foreground_percentile_normalize`):
foreground-only (pixel>0) 1st/99th percentile clip-and-scale to `[0,1]`, background forced to 0,
computed **per image**, at conversion time, applied uniformly regardless of split membership (not
a leakage vector since it's a per-image, label-free transform, but it **is** computed independent
of any frozen "normalization" step the Phase 2 plan calls for — Phase 2 specifies "one frozen
normalization," which is ambiguous as to whether it means this existing per-image percentile step,
or an additional dataset-level / ImageNet-style step layered on top for the CNN input).

**ROI** (`src/preprocessing/roi.py`), verified against real samples (60-sample random draw, all
512×512; mask tight-bbox longest side min=22px, median=77px, max=192px pixels):

| Aspect | Existing implementation (`roi.py`) | Phase 2 plan | Match? |
|---|---|---|---|
| Source | `image_normalized` (percentile-normalized float32) + `tumor_mask` | "Normalized MRI + mask" | Yes |
| Tight bbox | `find_mask_bbox`: half-open `(row_start,row_end,col_start,col_end)` from `np.nonzero(mask)` | tight tumor bbox | Yes |
| Padding definition | `ceil(base_side * 0.10)` **per side**, where `base_side = max(height, width)` of the tight bbox; final side = `base_side + 2*pad` | "~10% contextual padding" (basis unspecified) | Partially — existing code pads relative to the *longest tight-bbox side*, not the final square side or image size. This must be made explicit in the Phase 2 spec/config. |
| Square-ification | Square side fixed before centering; box centered on tight-bbox center, floor-rounded | square crop | Yes |
| Boundary handling | `crop_and_pad`: crops valid region, **zero-pads** the rest (never clips/shifts the box to stay in-bounds) | "boundary-safe crop" (method unspecified — could mean clip-and-shift instead of pad) | Needs explicit decision — zero-padding changes the effective field of view differently than shifting the box, especially for the 192px-side outliers near image edges. |
| Mask application | Produces **both** `roi_image` (unmasked) and `roi_image_masked` (background-zeroed) native views, plus resized counterparts | not specified whether background should be zeroed | **Key open question** — Phase 1 Gabor/Wavelet feed the *masked* 128×128 crop into feature extraction (confirmed via `inference.py`/`wavelet.py` call sites); GLCM/LBP use the *native unmasked* `roi_image` + mask together. Phase 2 CNNs conventionally benefit from surrounding tissue context; masking out background would discard that. |
| Resize | **128×128**, bilinear (image, `order=1`) / nearest (mask, `order=0`), mask re-binarized at 0.5 threshold | **224×224**, method unspecified, "grayscale replicated to 3 channels" | **Mismatch** — target size differs; existing `resize_roi()` hardcodes 128×128 as the default `standard_size` (parameterizable, so reusable with a changed argument, not a rewrite). |
| Grayscale→3ch | N/A (features are grayscale scalars) | required for CNN backbones pretrained on ImageNet | Not present — must be added in Phase 2. |
| Channel-stats / normalization for pretrained models | N/A | "one frozen normalization" (presumably ImageNet mean/std or dataset-specific) | Not present — must be added in Phase 2 preprocessing. |

Count of samples whose padded square crop would exceed image bounds: `required_padding` is
computed per-sample in `TumorROI.required_padding` (`roi.py:220-225`) but was **not** aggregated
across the full dataset in this audit (would require loading all 3,064 samples through
`prepare_tumor_roi`, which is a non-trivial compute step beyond "a few samples" diagnostic scope).
**UNCERTAIN** — flagged as an open item; can be computed cheaply in a follow-up pass by iterating
`iter_phase1_samples("all")` and counting `roi.required_padding`.

**Augmentation**: no augmentation code of any kind exists anywhere in `src/` (`grep` found none);
Phase 2's train-only augmentation module is entirely new work.

---

## 7. Existing training / evaluation assessment

- `src/models/training.py` provides `prepare_feature_matrices` (split-column-driven X/y/metadata
  split) and `fit_feature_preprocessor`/`transform_feature_matrices` (sklearn `Pipeline` of
  `SimpleImputer`+`StandardScaler`, fit-on-train-only). This is a **classical ML, non-iterative**
  trainer (`estimator.fit(X, y)` once) — it has no concept of epochs, batches, GPU/MPS placement,
  schedulers, or checkpoints. **Not reusable for Phase 2's trainer**; only the split-matrix
  preparation pattern is a useful conceptual precedent (REBUILD CLEANLY).
- `src/evaluation/metrics.py:evaluate_predictions` and `src/evaluation/results.py:build_result_dict`
  implement exactly the metric set the Phase 2 plan wants (accuracy, balanced accuracy, macro/weighted
  P/R/F1, per-class P/R/F1+support, fixed `[1,2,3]` confusion matrix order, OvR macro/weighted
  ROC-AUC with `label_binarize`) **except log-loss**, which exists only as an inline
  `sklearn.metrics.log_loss` call duplicated in `run_xgboost.py` and `run_xgboost_glcm_v2.py`, not
  in the shared `metrics.py` contract. **REUSE WITH MODIFICATION**: add `log_loss` to
  `evaluate_predictions` as a first-class field before Phase 2 relies on it, to avoid re-duplicating
  the same ad-hoc line in five new CNN scripts.
- `build_prediction_table` already produces `sample_id, patient_id, true_label, predicted_label,
  prob_<classname>×3` — matches the Phase 2 plan's prediction-table spec minus the model-count/
  inference-time/training-curve fields, which are experiment-metadata concerns Phase 2 will need to
  add alongside (not inside) this table.
- No generic early-stopping, checkpoint-metadata, or device-selection code exists anywhere in the
  repository (confirmed by `grep` for `cuda`, `mps`, `torch`, `checkpoint` across `src/` returning
  no hits outside dependency names).

---

## 8. Test coverage assessment

- Framework: `pytest`, 23 files under `tests/`, discovered via `python -m pytest -q`.
- **Run result: 234 passed, 0 failed, 0 skipped, 19.54s**, with 5 non-failing warnings (an LBP
  skimage float-image convention warning, a PyWavelets boundary-effect warning on the 128×128 level-2
  DWT, and 3 numpy 2.5/joblib deprecation warnings inside a synthetic-data XGBoost test). This
  **contradicts the README's claim of "139 passed"** — the suite has grown since that note was
  written and was not updated; the actual contract is passing, but the README is stale evidence
  that documentation and code can drift apart here.
- Representative assertions sampled: `tests/test_patient_split.py` (leakage/coverage asserts),
  `tests/test_roi.py` (bbox/padding/boundary/mask-binarization asserts), `tests/test_glcm_v2.py`,
  `tests/test_xgboost_glcm_v2.py`, `tests/test_review_demo.py` — these use a mix of small synthetic
  arrays (unit-level, fast) and, for dataset/split/audit tests, the **real** tracked
  `data/splits/patient_split.csv` where applicable. No test in this repository trains on or reads
  the full 3,064-sample raw/NPZ dataset (that would be slow); heavier experiment scripts
  (`run_xgboost_glcm_v2.py` itself) are driven manually, not via pytest, and are therefore
  untested by the automated suite beyond their helper-function unit tests.
- **Gap for Phase 2**: there is no `tests/test_phase2_smoke.py` or any DL-shaped test; none of the
  plan's proposed `tests/test_*.py` for `dl_dataset`, `dl_preprocessing`, `trainer`,
  `checkpointing`, `early_stopping`, `dl_metrics`, `dl_results` exist yet.

---

## 9. Reusable components (bucket 1 — SAFE TO REUSE)

- `src/data/mat_loader.py` — raw `.mat`/HDF5 loading and validation; well-tested (`tests/test_mat_loader.py`), used by every downstream pipeline, no changes needed for Phase 2 (same raw inputs).
- `data/splits/patient_split.csv` + `split_metadata.json` and the split-loading contract in `src/data/feature_dataset.py:load_split_index` — verified leakage-free; Phase 2 should consume the **same frozen split** for direct Phase 1/2 comparability.
- `src/preprocessing/normalization.py:robust_foreground_percentile_normalize` — simple, tested, per-image, label-free; reusable as the MRI-intensity normalization stage feeding ROI extraction.
- `src/evaluation/metrics.py:evaluate_predictions` and `src/evaluation/results.py` (prediction-table/result-schema builders) — directly satisfy most of the Phase 2 evaluation spec (see §7 caveat on log-loss).
- `src/data/create_patient_split.py` — not re-run, but the *split file it already produced* is the thing to reuse; the generator itself is reusable only if the split is ever intentionally regenerated (maintainer-only path per README).

## 10. Components requiring modification (bucket 2)

- `src/preprocessing/roi.py:prepare_tumor_roi`/`resize_roi` — core bbox/pad/crop logic is sound and tested, but needs: (a) `standard_size` changed from `(128,128)` to `(224,224)` for Phase 2 call sites, (b) an explicit decision + possible new parameter for whether Phase 2 crops use the masked or unmasked image, (c) grayscale→3-channel replication added on top (does not belong inside `roi.py` itself, but immediately downstream of it).
- `src/evaluation/metrics.py` — add a first-class `log_loss` field so five new CNN result dicts do not each re-duplicate the inline `sklearn.metrics.log_loss` pattern from `run_xgboost_glcm_v2.py`.
- `src/evaluation/results.py:build_result_dict`/`build_prediction_table` — extend (non-breaking) to accept the additional Phase 2 fields the plan requires (parameter count, inference time, training curves, checkpoint metadata) — these are new fields, not contract violations, but the schema (`REQUIRED_RESULT_FIELDS`) will need a Phase 2 variant or extension.
- `src/demo/pipeline_registry.py`/`inference.py` — the registry/adapter pattern (`feature_adapter`, `model_adapter`, `PipelineSpec`) is a good integration point for plugging in a CNN pipeline, but `inference.py`'s adapters are hard-coded to classical feature extraction (`extract_demo_features` branches on `glcm`/`lbp`/`gabor`/`wavelet`/`hog` by name) and to joblib-bundle sklearn/XGBoost estimators (`_predict_with_xgboost`, `_predict_with_generic_bundle`). A `cnn`/`torch` branch and a torch-checkpoint loader must be added; the dispatch *pattern* is reusable, the implementation is not CNN-ready as-is.

## 11. Components to rebuild (bucket 3) + Do-not-reuse list (bucket 4)

**Rebuild cleanly (concept right, implementation unsuitable):**
- `src/models/training.py` — the "fit preprocessor on train, transform all splits" *concept* is right, but the implementation (`sklearn.Pipeline.fit` once, no epochs/batches/device) cannot become a DL trainer; Phase 2's `src/training/trainer.py` must be new.
- `configs/paths.yaml` — the *idea* of a central paths config is right, but the file is currently dead code (unreferenced) and already out of date (points at `reports/figures`, which the repo has since renamed to `reports/experiments/*/plots` per the uncommitted renames in `git status`). Phase 2's `configs/dl/*.yaml` should not inherit from this file as-is.

**Do not reuse (bucket 4 — wrong, obsolete, or irrelevant to Phase 2):**
- The **currently-staged (uncommitted) merge versions** of `src/features/gabor.py` and
  `src/models/svm.py` — confirmed via `git diff --cached` to have stripped type hints, docstrings,
  and input-validation `raise` branches relative to `HEAD`. Do not branch Phase 2 work from the
  current index state until this merge is resolved/reviewed; branch from a clean `HEAD` or a
  reviewed merge commit instead.
- `configs/features/hog.yaml` with no corresponding `src/features/hog.py` — dead config, HOG+KNN is
  registered as `not_implemented` in the demo registry; irrelevant to Phase 2 (handcrafted feature,
  not a CNN concern).
- Any classical sklearn pipeline code (`svm.py`, `random_forest.py`, `lbp_logistic_regression.py`,
  `xgboost_model.py`) is Phase-1-specific and out of scope for reuse in the CNN training path,
  though it remains the baseline Phase 1 *results* are compared against.

---

## 12. Technical risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Unresolved merge commits a code regression | High (merge is open right now) | Medium — silently weakens gabor/svm validation | Resolve the merge deliberately (likely keep `HEAD`'s version, re-apply only the intended rename/structural changes) before cutting a Phase 2 branch |
| torch/torchvision absent from the dev environment | Certain (confirmed) | Blocks all Phase 2 work until installed | Install pinned `torch`/`torchvision`; verify MPS availability on this Darwin/arm machine and the `weights=` enum API before writing model code |
| ROI 128→224 and masking-policy mismatch causes Phase 1/2 incomparability | Medium | Medium — if Phase 2 silently uses a different effective crop/context than intended, visual quality and comparability to Phase 1 ROI-derived features suffers | Make padding basis, square size, boundary policy, and masking choice *explicit config values* in `configs/dl/input.yaml` and document the deviation from Phase 1, rather than silently reusing/forking `roi.py` defaults |
| No log_loss in shared metrics contract | Low | Low — easy to fix, but risk of 5× duplicated inline computation across new CNN scripts if not centralized first | Add `log_loss` to `evaluate_predictions` before writing any Phase 2 evaluation code |
| Unpinned dependencies (`requirements.txt` has no version pins) | Medium | Medium — numpy 2.5.1/sklearn 1.9.0/xgboost 3.4.1 already very new; a fresh install months from now could silently drift and break reproducibility of even Phase 1 results | Pin exact versions (or add a lockfile) before Phase 2 work multiplies the dependency surface |
| `src` not an installable package | Low | Low/Medium — works today via `python -m`, but fragile for notebooks, IDE tooling, or CI matrices | Consider adding a minimal `pyproject.toml` with an editable install, decoupled from Phase 2 scope |
| README drift (139 vs 234 tests; paths.yaml pointing at old `reports/figures`) | Medium | Low — mostly a trust/maintenance signal, not a functional bug | Update README alongside the next commit that resolves the pending merge |

---

## 13. Phase 2 readiness table

| Plan component | Existing equivalent | Bucket | Gap |
|---|---|---|---|
| `configs/dl/input.yaml` | `configs/features/*.yaml` pattern (YAML + loader in `src/features/config.py`) | 2 | Pattern reusable; content is new (224px, masking policy, channel replication) |
| `configs/dl/augmentation.yaml` | none | 3 | Build new; no augmentation code exists at all |
| `configs/dl/training.yaml` | `configs/models/xgboost*.yaml` pattern | 2 | Pattern reusable (frozen YAML + validator); DL-specific fields (lr schedule, epochs, early-stop patience) are new |
| `src/data/dl_dataset.py` | `src/data/feature_dataset.py` (`Phase1Sample`, `iter_phase1_samples`) | 2 | Reuse the split/NPZ loading and validation; add a `torch.utils.data.Dataset` wrapper on top |
| `src/preprocessing/dl_preprocessing.py` | `src/preprocessing/roi.py`, `normalization.py` | 2 | Reuse bbox/crop/normalization primitives; add 224 resize, grayscale→3ch, frozen tensor normalization |
| `src/training/device.py` | none | 3 | New — no CUDA/MPS/CPU logic anywhere in repo |
| `src/training/seed.py` | scattered `np.random.default_rng(seed)`/`random_state=` patterns | 2 | Pattern reusable; needs a single function seeding numpy/random/torch together |
| `src/training/trainer.py` | `src/models/training.py` (sklearn `.fit` once) | 3 | Concept only; must be rebuilt for epochs/batches/AdamW/class-weighted CE |
| `src/training/checkpointing.py` | `save_xgboost_artifacts` (joblib bundle + metadata dict pattern) | 2 | Bundle+metadata *pattern* reusable; torch `state_dict`/metadata schema is new |
| `src/training/early_stopping.py` | none | 3 | New |
| `src/evaluation/dl_metrics.py` | `src/evaluation/metrics.py` | 1 (mostly) | Directly reusable except log_loss needs adding first |
| `src/evaluation/dl_results.py` | `src/evaluation/results.py` | 2 | Reusable schema; extend for param count/inference time/training curves/checkpoint metadata |
| `tests/test_phase2_smoke.py` and DL unit tests | `tests/test_*.py` classical-ML equivalents as a style reference | 3 | New tests entirely; existing tests do not touch torch |
| Demo plug-in for a CNN | `src/demo/pipeline_registry.py` + `inference.py` | 2 | Registry pattern is a good integration point; add a `model_adapter == "torch"` branch and checkpoint loader |

---

## 14. Recommended implementation sequence

The Phase 2 plan's implicit order (configs → dataset → preprocessing → training infra → model
wrappers → evaluation → demo integration) is sound given the evidence gathered; **two deviations**
are recommended based on this audit:

1. **Resolve the pending git merge first** (not part of the original 12-step plan, but blocking —
   `gabor.py`/`svm.py` must not be committed in their currently-staged degraded form).
2. **Add `log_loss` to `src/evaluation/metrics.py` and pin `requirements.txt` versions before step 1
   of the DL plan**, so the Phase 2 evaluation and environment stories start from a clean, reusable
   base rather than copy-pasting the same ad-hoc log_loss line into five new scripts and inheriting
   today's unpinned-dependency risk.

Otherwise, proceed: `configs/dl/*.yaml` → `src/data/dl_dataset.py` (wrapping
`feature_dataset.iter_phase1_samples`) → `src/preprocessing/dl_preprocessing.py` (wrapping
`roi.py` with the 224/masking decisions made explicit in config) → `src/training/{device,seed}.py`
→ `src/training/trainer.py` → `src/training/{checkpointing,early_stopping}.py` →
`src/evaluation/{dl_metrics,dl_results}.py` → five model wrapper scripts → demo registry
integration → smoke tests last-mile, matching the plan's own sequencing.

---

## 15. Proposed Phase 2 repository structure

The plan's proposed structure is appropriate as given; the key isolation boundary the evidence
supports is: **everything under `src/data/`, `src/preprocessing/`, `src/training/`, and
`src/evaluation/dl_*` is architecture-agnostic common foundation**, shared by all five CNNs, while
each of `src/models/dl_<alexnet|vgg16|googlenet|resnet18|efficientnet_b0>.py` (naming inferred from
the plan, not present in the repo) should contain only the backbone construction + its
pretrained-weights loading call, delegating everything else (data, ROI, training loop, metrics,
checkpoints) to the common modules — mirroring how Phase 1's five feature families already share
`feature_dataset.py`/`roi.py`/`feature_table.py`/`training.py`/`metrics.py`/`results.py` while
keeping only the algorithm-specific extraction code (`glcm.py`, `gabor.py`, etc.) separate.

---

## 16. Explicit files to read before coding Phase 2

- `src/data/feature_dataset.py` (sample/split contract to wrap)
- `src/preprocessing/roi.py` and `src/preprocessing/normalization.py` (ROI/intensity logic to extend)
- `data/splits/patient_split.csv` + `data/splits/split_metadata.json` (the frozen split to reuse)
- `src/evaluation/metrics.py` and `src/evaluation/results.py` (metric/result contract to extend)
- `src/demo/pipeline_registry.py` and `src/demo/inference.py` (integration point for a CNN demo adapter)
- `configs/features/glcm_v2.yaml` and `configs/models/xgboost_glcm_v2.yaml` (as the best examples of this repo's "frozen YAML config + validator function" idiom to replicate for `configs/dl/*`)
- `reports/experiments/glcm_xgboost_v2/experiment_summary.json` (the Phase 1 number Phase 2 must be compared against)
- README.md in full (documents the contracts, though confirmed partially stale — cross-check against code, not against prose)

---

## 17. Open questions / ambiguities requiring a human decision

1. **Pending merge resolution.** Current behavior: repo is mid-merge with `gabor.py`/`svm.py`
   staged in a regressed state. Plan expectation: none (not covered by the Phase 2 plan at all).
   Options: (a) abort/redo the merge keeping `HEAD`'s validated code and reapplying only the
   intended file moves, (b) manually patch the staged files to restore validation before
   committing. Impact: if left unresolved, a future `git commit` silently ships weaker Phase 1 code
   underneath whatever Phase 2 branch is built on top of it.
2. **ROI resize target and masking policy for Phase 2.** Current behavior: `prepare_tumor_roi`
   defaults to 128×128 and produces a masked (background-zeroed) view that Gabor/Wavelet already
   consume. Plan expectation: 224×224, masking unspecified. Options: (a) reuse the masked crop for
   CNN input (consistent with half of Phase 1, but discards surrounding tissue context a CNN could
   exploit), (b) use the unmasked crop (more context, but then Phase 1 and Phase 2 see different
   information inside the same nominal "ROI"). Impact on Phase 1/2 comparability: whichever is
   chosen should be stated explicitly in `configs/dl/input.yaml` and in any comparison write-up,
   since it is a real methodological difference, not just an implementation detail.
3. **Padding basis.** Current behavior: 10% is computed relative to the tight bbox's longest side,
   not the image size or the final square side. Plan text ("~10% contextual padding") is ambiguous
   about the basis. Impact: changes effective field-of-view, especially for the smallest
   (22px-side) and largest (192px-side) tumors observed in the sampled data; should be pinned down
   in config/docs rather than left implicit.
4. **Boundary handling for out-of-bounds squares.** Current behavior: zero-pad (never shift/clip
   the box). Plan says "boundary-safe crop" without specifying the method. Exact count of
   affected samples was not computed in this audit (flagged as UNCERTAIN in §6) — recommend running
   `iter_phase1_samples("all")` through `prepare_tumor_roi` once to tabulate `required_padding`
   before deciding whether zero-padding is acceptable at 224×224 or whether a shift-based boundary
   policy is preferable for CNN inputs.
5. **What "one frozen normalization" means on top of the existing per-image percentile
   normalization.** Current behavior: percentile normalization already maps to `[0,1]` per image.
   Plan expectation: a single frozen normalization step for all five CNNs (commonly ImageNet
   mean/std after 3-channel replication). These are not mutually exclusive but need to be composed
   explicitly and documented, since applying ImageNet stats on top of an already-normalized `[0,1]`
   image is a different numerical pipeline than applying them directly to raw intensities.
6. **Whether `data/features/*` CSVs should ever be tracked for Phase 1/2 cross-comparison.** README
   says "may be exchanged later" — not resolved; irrelevant to blocking Phase 2 start but worth a
   decision before any shared feature-level ablation is attempted later.

---

## Appendix: commands executed (read-only), in order

1. `find . -maxdepth 3 ...` — top-level tree (excluding .git/venv/caches)
2. `git log --oneline --stat -30` — recent commit history
3. `find src tests configs data reports -type f ...` + `du -sh 1512427 data reports results` — structure/size inventory
4. `git status`, `git remote -v`, `git branch -a`, `cat .gitignore` — repo/VCS state
5. `Read README.md` — documented contracts
6. `Read` of `src/data/mat_loader.py`, `src/data/create_patient_split.py`, `src/preprocessing/roi.py`
7. `Read` of `src/data/feature_dataset.py`, `src/preprocessing/normalization.py`, `src/evaluation/metrics.py`
8. `find reports/experiments/glcm_xgboost_v2 -maxdepth 2 -type f` + `cat .../experiment_summary.json` — verify reported GLCM v2 metrics
9. `Read` of `src/models/training.py`, `src/evaluation/results.py`
10. `Read` of `src/demo/pipeline_registry.py`, `src/demo/inference.py`
11. `cat reports/experiments/glcm_xgboost_v2/test_confusion_matrix.csv` — verify reported confusion matrix
12. `ls data/processed/samples | wc -l` + `cat data/splits/split_metadata.json` (parsed via `python3 -c`) — verify split counts
13. `python3 -c` with pandas — verify PID intersection sizes across train/val/test (leakage check)
14. `python3 -c` with numpy — 60-sample random draw of image shapes and mask tight-bbox side lengths
15. `python3 -m pytest -q` — run full test suite (234 passed, 19.54s)
16. `cat requirements.txt`, `python3 --version`, `python3 -c "import ..."` — dependency/version inventory (torch/torchvision confirmed absent)
17. `grep -rn "train_test_split|KFold|shuffle=True|np.random|random.seed|random_state" src/` — leakage/seed audit across source
18. `git diff --cached -- src/features/gabor.py src/models/svm.py` — inspect uncommitted merge-staged regression
19. `Read src/features/glcm.py` (partial) — v1 GLCM feature definitions
20. `git log --oneline -1 -- src/features/gabor.py` / `git log --all --oneline -- src/features/gabor.py` — history check on the modified file
21. `Read configs/paths.yaml` + `grep -rln "paths.yaml|configs/paths" src/ tests/` — confirm `paths.yaml` is dead/unreferenced
22. `ls src/features/` + `grep -n "roi_image..." src/features/lbp.py src/features/wavelet.py` — confirm native vs standardized ROI usage per feature family
23. `Read src/models/xgboost_model.py` (partial) — seed/label-mapping contract
