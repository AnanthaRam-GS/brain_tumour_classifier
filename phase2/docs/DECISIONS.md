# Phase 2 Decision Record — contract version 1.0.0

This document freezes the Phase 2 (deep learning) design decisions for the
brain tumor MRI classifier. Every decision below is binding for all five
Phase 2 architectures (AlexNet, VGG16, GoogLeNet, ResNet18,
EfficientNet-B0) unless explicitly stated otherwise.

## D1 — Location and isolation

**Decision:** Phase 2 lives in `phase2/` inside this repository as an
installable package named `btdl` (`phase2/src/btdl`, src layout). Phase 2
has its own virtual environment (`phase2/.venv`). Phase 2 runtime code never
imports Phase 1 code (the top-level `src` package). Parity/comparison tests
may import Phase 1 code, but only from within `phase2/tests`.

**Rationale:** Phase 1 and Phase 2 have different dependency graphs (Phase 2
needs torch/torchvision; Phase 1 does not) and different lifecycles. Keeping
them in separate packages and separate environments prevents dependency
collisions and keeps Phase 1's 234-test suite immune to Phase 2 changes.

## D2 — Data source

**Decision:** Phase 2 reads Phase 1's canonical NPZs in
`data/processed/samples/` (fields: `image_normalized`, `tumor_mask`, `label`,
`patient_id`, `sample_id`). These are cross-verified against the raw
Figshare `.mat` files by an independent reader written in Phase 2, with a
SHA-256 manifest recorded for the sample set actually used.

**Rationale:** Reuses Phase 1's already-validated normalization and loading
work instead of re-deriving it, while an independent re-read against the raw
`.mat` files guards against silently trusting a corrupted or stale cache.

## D3 — Split

**Decision:** `data/splits/patient_split.csv` is reused exactly as produced
by Phase 1 (train 2,120 / val 446 / test 498; patients 162 / 34 / 37; zero
patient overlap across splits). Pinned by SHA-256 hash. Phase 2 never
regenerates a split.

**Rationale:** The audit (`docs/audit/phase2_repository_audit.md`) verified
this split is leak-free and exactly reproduces the counts reported for
Phase 1. Reusing it, rather than regenerating one, is the only way Phase 1
and Phase 2 results are directly comparable.

## D4 — ROI geometry (identical to Phase 1)

**Decision:** Tight mask bounding box; padding = `ceil(0.10 x longest
tight-box side)` applied per side; centered square; out-of-bounds regions
zero-padded (background is already 0 after foreground percentile
normalization); the **unmasked** image is cropped — the mask only localizes
the crop, so the CNN sees peritumoral context.

**Rationale:** Matches Phase 1's `prepare_tumor_roi()` geometry so any
accuracy difference between Phase 1 and Phase 2 is attributable to the model
family, not to a different ROI definition. Using the unmasked image (vs.
Phase 1's masked GLCM input) is a deliberate, documented difference — see
D15 limitations.

## D5 — Input

**Decision:** ROI resized to 224x224 (bilinear, antialias); grayscale
replicated to 3 channels; ImageNet mean/std normalization.

**Rationale:** 224x224x3 with ImageNet statistics is the standard input
contract expected by all five torchvision architectures' pretrained weights.

## D6 — ROI cache

**Decision:** The deterministic, pre-augmentation 224x224 single-channel ROI
for every sample is built once (stored as float16) with a recorded hash. All
five models train from the same cache.

**Rationale:** Guarantees every architecture sees byte-identical ROI input
before augmentation, removing ROI computation as a confound between models,
and avoids recomputing the same deterministic crop/resize on every epoch.

## D7 — Augmentation (train only; val/test deterministic)

**Decision:** Horizontal flip p=0.5; affine rotation ±10°, translation ±5%,
scale 0.9–1.1, zero fill; brightness and contrast ±10%. No vertical flip
(the dataset mixes axial/coronal/sagittal views, so vertical flip is not a
label-preserving transform). Applied before channel replication and
normalization.

**Rationale:** Modest geometric/photometric augmentation appropriate for
MRI slices without introducing anatomically implausible transforms.

## D8 — Models

**Decision:** torchvision ImageNet-pretrained weights, full fine-tuning
(no frozen backbone), final classifier replaced with a 3-class linear layer.
GoogLeNet keeps its shipped `transform_input` and uses `aux_logits=False` so
every model trains against the same loss shape.

**Rationale:** Full fine-tuning gives every architecture a fair chance on a
small (~3,000-image) medical dataset; `aux_logits=False` removes GoogLeNet's
auxiliary-loss asymmetry relative to the other four models.

## D9 — Loss

**Decision:** `CrossEntropyLoss` with inverse-frequency class weights
`n / (K * n_c)` computed from the **train** split only.

**Rationale:** Matches Phase 1's practice of deriving class handling only
from train-split statistics; prevents val/test leakage into the loss.

## D10 — Optimization

**Decision:** AdamW (weight decay 1e-4), cosine annealing over max epochs,
batch size 32, max 40 epochs, early stopping patience 8 on validation
macro-F1 (tie-break: lower validation loss), fp32.

**Rationale:** A single, shared, reasonable optimization recipe keeps the
five-architecture comparison about architecture, not about per-model tuning.

## D11 — Hyperparameter fairness

**Decision:** Identical learning-rate grid `{1e-4, 3e-4, 1e-3}` for every
model at seed 42, selected by validation macro-F1. The selected LR is then
run with seeds 42, 43, 44.

**Rationale:** Same search budget and seed for every architecture avoids
giving any one model an unfair tuning advantage; the 3-seed re-run quantifies
run-to-run variance for the reported metric.

## D12 — Selection

**Decision:** Best epoch is chosen by validation macro-F1. No refit on
train+val.

**Rationale:** Documented, deliberate difference from Phase 1, which refit
its final GLCM v2 + XGBoost model on train+val before the single test
evaluation. Phase 2 reports this difference explicitly (see D15) rather than
silently diverging from the baseline's methodology.

## D13 — Test gate

**Decision:** Only the final-test CLI may evaluate the test split. It
requires a frozen checkpoint and appends to a test-access log.

**Rationale:** Enforces "touch the test set once" discipline mechanically
rather than relying on developer self-restraint, matching the spirit of
Phase 1's single-test-evaluation practice (see the leakage assessment in the
repository audit).

## D14 — Evaluation

**Decision:** Accuracy, balanced accuracy, macro/weighted precision/recall/
F1, per-class P/R/F1, OvR ROC-AUC (macro and weighted), log loss, 3x3
confusion matrix in label order `[meningioma, glioma, pituitary]`;
patient-level bootstrap 95% confidence intervals (2,000 resamples); per-sample
tumor bounding-box size recorded; parameter count; inference time (batch 1
and batch 32, after warm-up, with device recorded).

**Rationale:** Superset of Phase 1's evaluation contract (see
`src/evaluation/metrics.py`), extended with bootstrap CIs and inference-time
profiling that Phase 1 did not need but a deep-learning comparison does.

## D15 — Reporting

**Decision:** Report mean ± std over 3 seeds. Phase 1's GLCM v2 + XGBoost
result (test macro-F1 0.5891) is the classical-baseline row in every
comparison table. State these limitations explicitly wherever Phase 2 is
compared to Phase 1:
- Phase 2 uses oracle (ground-truth) mask localization for the ROI, which
  Phase 1's deployed pipeline also used, but which a real deployment would
  not have at inference time.
- Phase 2 does not refit on train+val (D12), while Phase 1 did.
- Phase 2 crops the **unmasked** image (peritumoral context included), while
  Phase 1's GLCM v2 features were computed on the **masked** ROI.

**Rationale:** These three differences are the ones most likely to be
mistaken for a Phase 2 "win" or "loss" that is actually a methodology
difference; naming them keeps the comparison honest.

## D16 — Reproducibility

**Decision:** Seeds are set for Python's `random`, NumPy, and torch; DataLoader
workers and the sampling generator are seeded; deterministic algorithms are
enabled where supported. MPS is not bitwise-deterministic, so
checkpoint-reload equivalence is tested within a small numerical tolerance,
and only on the same device.

**Rationale:** Full bitwise determinism is not achievable on MPS (Apple's
backend), so the reproducibility test is scoped to what MPS can actually
guarantee, instead of asserting something that would flake on this team's
development hardware.

**Implementation note (train-time augmentation):** augmentation randomness
(`src/btdl/preprocessing/augmentation.py`) is keyed by a stable sha256-based
hash of `(seed, epoch, sample_id)`, seeding a local `torch.Generator` per
call -- never Python's `hash()` (not stable across processes) and never the
global torch/numpy/python RNG. This makes a given sample's augmentation for
a given epoch identical regardless of `num_workers`, batch order, or
device. This clarifies D16 and does not change any decision; no
`CONTRACT_VERSION` bump.

## D17 — Test-set discipline during development

**Decision:** The smoke-test experiment exercises the test-evaluation code
path against the **validation** split. The real test split is used only for
the five final models.

**Rationale:** Lets the test-evaluation code itself be exercised and
debugged repeatedly during development without spending "looks at the test
set" budget on anything other than the five final, frozen models.

## D18 — Registration before final test

**Decision:** A model's final runs must be **registered**
(`cli/register_frozen.py`, with the registration file committed) before
`final_test` may run for that model. Once registered, a model may be
tested **independently** of every other model -- there is no requirement
to wait for the other four architectures. Efficiency figures used in the
cross-model comparison are still measured centrally, on one machine, for
all five architectures (CPU and the owner's accelerator); the per-run
`efficiency.json` files produced during each model's own development
remain as provenance but are not the comparison's source of truth.

**Rationale:** The training protocol (grid search, selection, final
seeds) is fixed before any model is trained, so a test result cannot feed
back into selection for that model or any other -- the thing D13/D17's
test gate exists to prevent. What matters is that the exact checkpoint
being tested is provably the one that was frozen: the committed
registration hash (`best_pt_sha256` per seed) proves no run was retrained
after its test, which is the real integrity property, not the relative
order in which different teammates happen to finish. Requiring every
architecture to be ready before any of them can be tested (the earlier
"simultaneous unblinding" policy) was removed at the foundation owner's
request to match the team's actual schedule -- teammates finish at
different times, and blocking a finished model on others still in
progress serves no integrity purpose once registration alone proves
no-retraining.

This is a **process decision**: no contract file changed, and
`CONTRACT_VERSION` stays `1.0.0`.

**Enforcement:** D18 is enforced in code, not only by convention:

- `cli/register_frozen.py` writes a TRACKED, provenance-only
  `artifacts/frozen/<model>.json` (model/version/weights_id/selected lr/
  contract hash/lock version/git commit, and per `final_seed`: run_dir,
  `best_pt_sha256`, `frozen_at_commit`, best_epoch, best val macro-F1 --
  deliberately NO test information) once every final seed's run is
  frozen, conformant, clean, and has a passing rehearsal.
- `cli/final_test.py` (real mode only -- `--rehearsal` is completely
  unaffected) refuses to run unless the model being tested has a frozen
  registration whose recorded sha256 for this seed matches the exact
  checkpoint in use. There is no cross-model readiness gate: a model may
  be tested as soon as it is registered, regardless of any other model's
  status.
- `cli/export_results.py` independently re-checks the same registration
  before copying anything, refusing on a sha mismatch. It has never
  required a clean working tree on its own.
- `phase2/artifacts/test_access_log.csv` is exempt from the dirty-tree
  check used by `final_test`/`train` when its only changes since HEAD are
  well-formed appended rows (an exact byte prefix of HEAD plus rows with
  the header's column count) -- so running `final_test` once per final
  seed needs no commit between calls; see `config.git_dirty_paths`. No
  other file gets this exemption, and any non-append change to the log
  (an edit, a deletion, a reordering, a malformed row) is still dirty.
- `cli/unblinding_status.py` (CLI name kept for compatibility) now gates
  only the full five-model comparison (`cli/compare.py --mode test`),
  which legitimately needs every study model present: it reports READY
  only when every model in `configs/study.yaml` has a registration
  covering every `final_seed` with a `contract_dir_sha256` matching the
  current contract.
- `cli/efficiency_central.py` performs the central, single-machine
  measurement for every study model that is registered in the catalog,
  writing TRACKED `artifacts/comparison/efficiency_<device>.json`.
- `configs/study.yaml` (schema-validated, part of the foundation lock but
  not of `configs/contract/`) names which architectures are in scope for
  the comparison.

This enforcement is tooling/process only: `configs/contract/` is
unchanged, `CONTRACT_VERSION` stays `1.0.0`.

## D19 — Supplementary calibration analysis

**Decision:** Calibration is measured and reported as a **descriptive,
supplementary** analysis only -- it never influences model selection
(lr/seed selection, the architecture comparison's headline ranking, or
anything else). The metric is the **expected calibration error (ECE)**:
top-label (the predicted class's own confidence vs. whether that
prediction was correct), **15 equal-width bins** over `[0, 1]`,
`ECE = sum_bins (|bin| / N) * |accuracy(bin) - confidence(bin)|`. Reported
per model as the mean ± sample std of each seed's ECE, alongside a
reliability diagram (confidence vs. empirical accuracy per bin). Added
before unblinding (the comparison pipeline computes it from exported
test-split predictions, same as every other reported metric).

**Rationale:** A model with a lower macro-F1 but better-calibrated
probabilities (or vice versa) is a materially different kind of result
than a difference in macro-F1 alone communicates, and is worth recording
for anyone who later wants probability estimates (not just hard labels)
from these models. It is explicitly supplementary -- not a selection
criterion -- because the whole protocol (D11/D12) and the training recipe
were designed and fixed around macro-F1-based selection; retrofitting
calibration into the selection criterion after the fact would change what
"selection" means without the team agreement D10-D12 required for the
original criterion.

This is a **process decision**: no contract file changed, and
`CONTRACT_VERSION` stays `1.0.0`.

---

## Changing a decision

Any change to a decision in this document requires:
1. Team agreement (not a unilateral code change).
2. A `CONTRACT_VERSION` bump in `src/btdl/contracts.py` if the change affects
   class semantics, or a version note here otherwise.
3. An entry in the changelog below recording what changed and why.

## Changelog

- 2026-10-08 — Initial decision record (D1–D17), contract version 1.0.0.
- 2026-10-10 — Added D18 (simultaneous unblinding, central efficiency
  measurement). Process decision only -- no contract file changed,
  `CONTRACT_VERSION` stays 1.0.0.
- 2026-10-10 — Added D18's "Enforcement" paragraph: D18 is now enforced in
  code (`configs/study.yaml`, `cli/register_frozen.py`,
  `cli/unblinding_status.py`, D18 gates in `cli/final_test.py` and
  `cli/export_results.py`, `cli/efficiency_central.py`), foundation lock
  bumped to 1.1.0. Process/tooling only -- `configs/contract/` unchanged,
  `CONTRACT_VERSION` stays 1.0.0.
- 2026-10-11 — Added D19 (supplementary, descriptive-only calibration
  analysis: top-label ECE, 15 equal-width bins; never influences
  selection). Added the cross-model comparison pipeline
  (`src/btdl/comparison/`, `cli/compare.py`) and reliable CPU inference
  timing in `evaluation/efficiency.py`/`cli/efficiency_central.py`.
  Foundation lock bumped to 1.2.0. Process/tooling only -- `configs/
  contract/` unchanged, `CONTRACT_VERSION` stays 1.0.0.
- 2026-10-12 — D18 retitled "Registration before final test" and revised
  at the foundation owner's request (team schedule): the simultaneous
  five-model unblinding gate is removed from `cli/final_test.py`; a model
  may be tested as soon as its own registration is committed, independent
  of every other model. The registration gate itself (frozen checkpoint
  sha must match what was registered) is retained -- test results still
  cannot affect selection, since the training protocol is fixed before
  training begins and the committed registration hash proves no run was
  retrained after its test. `phase2/artifacts/test_access_log.csv` is now
  exempt from the dirty-tree check when its only changes are well-formed
  appended rows, so `final_test` for each final seed needs no commit in
  between. `cli/unblinding_status.py` (name kept for compatibility) now
  gates only `cli/compare.py --mode test`. Process/tooling change only --
  `configs/contract/` unchanged, `CONTRACT_VERSION` stays 1.0.0.
