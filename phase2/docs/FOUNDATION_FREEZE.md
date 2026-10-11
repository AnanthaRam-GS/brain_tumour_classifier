# Phase 2 Foundation Freeze

This document records the state of the Phase 2 common foundation as of commit
`45465fb6` on branch `phase2/registry-smoke`, after the real-data smoke
experiment validated the full pipeline end-to-end on this machine's default
device (Apple MPS). Everything described here is frozen: later work (model
architectures, grid runs, final test-set evaluation) builds on top of it
without modifying it.

**Handoff update (branch `phase2/handoff`):** the foundation is now also
protected by an explicit, machine-checked lock (see §7 below).
`configs/contract/` itself did not change -- the `contract_dir_sha256`
below is still exact. §7 also records the updated full test counts as of
this handoff work.

## 1. What is frozen

### Contracts

| Contract file | `contract_dir_sha256` (whole `configs/contract/` dir) |
|---|---|
| `configs/contract/data.yaml` | |
| `configs/contract/input.yaml` | |
| `configs/contract/augmentation.yaml` | |
| `configs/contract/training.yaml` | |
| `configs/contract/evaluation.yaml` | |

```
contract_dir_sha256 = 311a58cdb40226a3f8744679098686e0497c271b9ff32203f15c1ca7c6d646c6
```

This hash is recorded in every run's `metadata.json` and `FROZEN.json`, and
is re-verified (exact match required) by `cli/final_test.py` before any real
test-set evaluation.

### Data provenance hashes

| Artifact | sha256 |
|---|---|
| `split_sha256` (patient-level train/val/test split) | `ab45f42e99b4dc10c56b928a3a3ab17d753fb4251a970c51633c37c4a17fe9c1` |
| ROI cache `.npy` (`npy_sha256`, from `roi_cache_meta.json`) | `d9860739dc6fe5cec0c806d30e2630c4d3df24fe3f0d33c639ff4b3f88665735` |
| Manifest (`manifest_sha256`) | `b511f8d325c845983dcceb4bbc4bb31edcdea9e84f2dee4c92c77f8f7ca91314` |
| `artifacts/contract/size_tertiles.json` | `0a7e54d4ee9226b48e608646713026aba2a2911cfecb9b155560544429444049` |

Size tertile edges (ROI long-side, pixels): `[75.0, 113.66666666666674]`.

All four are recorded in every run's `metadata.json` for provenance; the
first three are also checked by `build_run_metadata`'s
`contract_conformant`/`contract_deviations` machinery whenever an effective
config is compared against the committed contract.

### Model registry

`src/btdl/models/registry.py` exposes an explicit `MODEL_REGISTRY: dict`
(no auto-discovery) of `ModelSpec(name, version, builder, weights_id,
reference_only, description)` entries, plus `get_spec(name)`,
`list_models()`, and `build_model(name, *, pretrained=True)`. Each entry has
a matching strict-whitelist YAML at `configs/models/<name>.yaml` (no keys
beyond `name`, `version`, `weights_id`, `reference_only`, `description` are
accepted — `config.load_model_config` raises on anything else).

Currently registered:

| name | version | reference_only | params | description |
|---|---|---|---|---|
| `tiny_cnn` | 1.0.0 | `true` | 98,307 | tiny 4-conv-block reference CNN, never eligible for real `final_test` |

`tiny_cnn` exists purely to exercise the training/evaluation pipeline; it is
intentionally `reference_only=True` so it can never pass the real-mode gates
in `cli/final_test.py` (only `--rehearsal`, against the val split). The five
real architectures (AlexNet, VGG16, GoogLeNet, ResNet18, EfficientNet-B0)
are out of scope for this freeze and will be added as additional registry
entries without touching anything described here.

## 2. Full test count

Phase 2 suite, from `phase2/`, `python -m pytest -q`:

- **Full suite (including slow): 402 passed, 0 failed, 0 skipped — 235.86s**
- **Fast only (`-m "not slow"`): 377 passed, 25 deselected — 44.92s**
- 25 tests carry `@pytest.mark.slow` (real-data contract/parity checks,
  default-size bootstrap, trainer convergence runs, and every CLI test that
  actually calls `run_train`/`run_final_test` end-to-end rather than just
  checking a validation gate).
- `tests/test_model_contract.py`: 9 tests, parametrized over every entry in
  `list_models()` (currently just `tiny_cnn`); will scale automatically as
  more models are registered.

Phase 1 suite, from repo root, `python -m pytest -q`:

- **234 passed, 0 failed — 18.12s**

Both suites were run back-to-back on a clean tree immediately before this
document's commit.

## 3. Smoke experiment results (real MPS hardware)

Machine default device, confirmed by `describe_device()`:

```json
{"type": "mps", "name": "Apple MPS", "torch_version": "2.14.1",
 "cuda_available": false, "mps_available": true,
 "platform": "macOS-26.6.2-arm64-arm-64bit"}
```

### D1 — determinism probe

Two `--smoke` runs of `tiny_cnn` at `lr=1e-3, seed=42` (2 epochs each, into
separate `_smoke` run directories) produced **not bit-identical**
`history.csv` files. Per-column max absolute difference between the two
runs:

| column | max abs diff |
|---|---|
| epoch | 0 |
| lr | 0.0 |
| train_aug_loss | 0.00829 |
| train_aug_macro_f1 | 0.00216 |
| val_loss | 0.1286 |
| val_log_loss | 0.0725 |
| val_accuracy | 0.0314 |
| val_balanced_accuracy | 0.0229 |
| val_macro_f1 | 0.0474 |
| improved | 0 rows differ |
| epoch_seconds | 0.0251 (wall-clock; expected to differ) |

This confirms D16's documented decision: MPS is not bitwise-deterministic
even with an identical seed and `deterministic_algorithms_warn_only=True`.
Reload equivalence (same checkpoint reloaded) is still exact (see below) —
only independent *training* runs diverge, as expected.

### D2 — conformant reference run (`tiny_cnn`, lr=1e-3, seed=42, full contract)

```
run_dir: phase2/runs/tiny_cnn/lr1e-03_seed42/
device: mps
epochs_run: 11
stop_reason: early_stopping
best_epoch: 2
best_val_macro_f1: 0.7161824688998601
best_val_loss: 0.6626036167144775
```

Full `eval/val/metrics.json`:

| metric | value |
|---|---|
| accuracy | 0.7489 |
| balanced_accuracy | 0.7270 |
| macro_f1 | 0.7162 |
| macro_precision | 0.7099 |
| macro_recall | 0.7270 |
| weighted_f1 | 0.7531 |
| weighted_precision | 0.7613 |
| weighted_recall | 0.7489 |
| log_loss | 0.6725 |
| roc_auc.macro_ovr | 0.8993 |

Per-class:

| class | f1 | precision | recall | support |
|---|---|---|---|---|
| meningioma | 0.5963 | 0.5517 | 0.6486 | 74 |
| glioma | 0.8221 | 0.8680 | 0.7808 | 219 |
| pituitary | 0.7302 | 0.7099 | 0.7516 | 153 |

Confusion matrix (rows = true, cols = predicted, order
`[meningioma, glioma, pituitary]`):

```
[[ 48,   3,  23],
 [ 24, 171,  24],
 [ 15,  23, 115]]
```

Mean epoch time: ~6.7-7.2s/epoch on MPS (11 epochs, early-stopped at
patience=8 past best_epoch=2).

**Reload equivalence (C8):** `reload_max_abs_diff = 1.1102230246251565e-16`
— the reloaded `best.pt`'s val-split predictions match the saved
`val_predictions.csv` to float64 machine epsilon, well under the contract's
`reload_equivalence_atol = 1.0e-5`. Both `eval/val/` and
`eval/rehearsal_val/` pass `validate_evaluation_dir(require_contract=True)`.

**Efficiency** (`cli.efficiency`, batch sizes from `evaluation.yaml`):

| | parameters | inference |
|---|---|---|
| total | 98,307 | |
| trainable | 98,307 | |
| batch=1 | | median 0.900 ms/batch, p90 1.133 ms/batch, 0.900 ms/image |
| batch=32 | | median 15.967 ms/batch, p90 16.095 ms/batch, 0.499 ms/image |

Best val macro-F1 (0.7162) is well above the 0.45 flag threshold — no flag
raised.

### D3 — test-set isolation confirmed

- `cli.final_test --rehearsal` ran entirely against the **val** split; it
  never constructed a test-split `RoiDataset` (the only
  `allow_test=True` call site in the entire `src/btdl` tree is the literal
  call inside `cli/final_test.py`, enforced by
  `tests/test_no_allow_test_outside_final_test.py`'s AST-based guard test).
- `phase2/artifacts/test_access_log.csv` remained header-only (1 line)
  after the rehearsal run — rehearsal never appends a log row.
- `git status --short` was clean immediately after the rehearsal run,
  confirming nothing it touched was tracked/dirtying state outside the
  gitignored `runs/` tree.

## 4. Model workflow — exact commands

Once per candidate architecture `<model>` (registered in
`MODEL_REGISTRY` with `reference_only=False`):

```bash
cd phase2

# 1. Grid search at the fixed grid seed, one run per lr in training.yaml's lr_grid
python -m btdl.cli.train --model <model> --lr 1e-4 --seed 42
python -m btdl.cli.train --model <model> --lr 3e-4 --seed 42
python -m btdl.cli.train --model <model> --lr 1e-3 --seed 42

# 2. Select the best lr by macro-F1 (then val-loss, then lr) across the grid runs
python -m btdl.cli.select_lr --model <model>
# writes artifacts/selection/<model>.json (tracked; committed)

# 3. Re-run at the selected lr for each of the two remaining final_seeds
python -m btdl.cli.train --model <model> --lr <selected_lr> --seed 43
python -m btdl.cli.train --model <model> --lr <selected_lr> --seed 44

# 4. For each of the 3 final runs (seeds 42, 43, 44 at the selected lr):
python -m btdl.cli.freeze      --run-dir runs/<model>/lr<selected_lr>_seed<seed>
python -m btdl.cli.efficiency  --run-dir runs/<model>/lr<selected_lr>_seed<seed>
python -m btdl.cli.final_test  --run-dir runs/<model>/lr<selected_lr>_seed<seed> --confirm-final-test
```

`final_test` without `--rehearsal` requires: the run is frozen (`FROZEN.json`
present, `best_pt_sha256`/`contract_dir_sha256` match), `contract_conformant`
is `True`, the working tree is clean, the model is not `reference_only`, the
run's seed is in `final_seeds`, and the run's lr matches
`selection/<model>.json`'s `selected_lr` — and it may be run at most once per
run directory (checked against `test_access_log.csv`).

## 5. Smoke artifacts

The smoke run's evaluation plots, efficiency numbers, training history, and
rehearsal metrics are committed under `phase2/artifacts/smoke/` for
reference (the `runs/` directory itself stays gitignored):

- `training_curves.png`, `confusion_matrix.png`, `roc_curves.png` (from
  `eval/val/`)
- `efficiency.json`
- `history.csv`
- `rehearsal_val_metrics.json` (from `eval/rehearsal_val/metrics.json`)

## 6. Out of scope for this freeze

AlexNet, VGG16, GoogLeNet, ResNet18, and EfficientNet-B0 registry entries,
and any real (non-rehearsal) `final_test` run against the actual test split,
are explicitly out of scope here and were not touched.

## 7. Handoff hardening (branch `phase2/handoff`)

Before opening the foundation to parallel model development, the
following were added on top of everything above, without changing
`configs/contract/` (`contract_dir_sha256` is unchanged, §1):

- **Scoped dirty-tree definition (A1):** `train`/`final_test`/run metadata
  now only treat a tree as dirty for tracked changes anywhere in the repo
  or untracked files under `phase2/` -- an untracked file at the repo
  root no longer blocks training. Offending paths are recorded in
  `metadata.json`'s `git_dirty_paths`.
- **`BTDL_CACHE_DIR` (A2):** overrides the default `phase2/cache/`
  directory for both reading and building the ROI cache; the resolved
  path is recorded in `metadata.json`'s `roi_cache_path`.
- **Pretrained-weights provenance (A3):** `metadata.json` records
  `weights_id` and `pretrained_weights_loaded`; `cli/train.py` passes
  `pretrained=True` exactly when the model's registry spec has a
  `weights_id`.
- **Shared torchvision classifier helper (D8):**
  `models/torchvision_common.py`'s `build_torchvision_classifier()`
  implements "load a torchvision architecture, optionally with ImageNet
  weights, replace its final layer with `nn.Linear(in_features, 3)`" once,
  with a verified `head_path`/`builder_kwargs` table for all five
  remaining architectures (see `docs/TEAM_GUIDE.md`).
- **Foundation lock:** `cli/lock.py` (`--write`/`--check`) hashes every
  locked foundation file -- `configs/contract/`, `config.py`,
  `contracts.py`, `runs.py`, every `.py` file under
  `data/preprocessing/training/evaluation/cli/`, and the registry
  machinery (`models/registry.py`, `models/model_spec.py`,
  `models/torchvision_common.py`) -- into the tracked
  `artifacts/contract/foundation_lock.json` (`lock_version` **1.0.0**),
  alongside `contract_dir_sha256`. `models/catalog.py` and any
  `models/<model>.py` are deliberately NOT locked, so a teammate can
  register a new architecture without review from the foundation owner.
  `.github/CODEOWNERS` encodes the same split.
- **`cli/check_setup.py`:** a PASS/FAIL/WARN checklist (versions, btdl's
  import location, device, contracts, the foundation lock, manifest/split
  hashes, a full ROI cache sha256 verify, working-tree cleanliness as a
  WARN, no Phase 1 `src` import, and the registry) that exits non-zero on
  any FAIL.
- **`cli/export_results.py`:** copies a real-final-tested run's
  human-review artifacts (never a checkpoint) into tracked
  `artifacts/results/<model>/seed<seed>/`, validated with
  `validate_evaluation_dir(require_contract=True)`.
- **Fresh-clone, cache-only rehearsal:** a plain `git clone` of this repo
  (outside the working tree, at this branch's HEAD) with only
  `roi224_v1.npy` copied in and `BTDL_CACHE_DIR` pointed at it --
  `check_setup` passed every check from the clone (confirmed `btdl`
  resolved under the clone's own `phase2/src`), and
  `cli.train --model tiny_cnn --lr 1e-3 --seed 42 --smoke` completed
  successfully. No raw dataset or per-sample NPZs were needed. Both the
  temporary clone and the temporary cache copy were deleted afterward.

### Test counts as of the handoff hardening commits

- Phase 2 full suite (including slow): **453 passed**, 320.06s
- Phase 2 fast (`-m "not slow"`): **424 passed**, 29 deselected, 73.58s
- Phase 1 suite: **234 passed**, 17.91s

All green on a clean tree immediately before that document's own commit.

## 8. Pre-freeze fix: requirements.lock (branch `phase2/handoff`)

`phase2/requirements.lock` turned out to have been committed and tracked
all along (first in `3e1c0971`, updated in `8e8a527f`) -- it was never
missing, never gitignored, and matched the installed environment exactly.
The earlier claim in this document's §7 and in `cli/check_setup.py`'s
docstring that "this repo has no separate requirements.lock file" was
simply wrong, caused by checking `cat requirements.lock` from the repo
root in the investigating session instead of from `phase2/`. No
`.gitignore` pattern was involved.

Fixed:

- `requirements.lock` now carries a 3-line header comment (generation
  command, Python version, platform) and was regenerated via
  `phase2/.venv/bin/python -m pip freeze --exclude-editable` (content
  identical to what was already committed, modulo the new header) -- 42
  lines total, sha256 `90342a26323fbcc80713e213dfcc29c38ad44b22851accf1fb5f94cd23207fa8`.
- `check_setup`'s version check now enforces a two-tier policy: the CORE
  packages (torch, torchvision, numpy, scikit-learn, scipy, h5py, pyyaml,
  pandas, matplotlib) must exactly match `pyproject.toml`'s pins (FAIL
  otherwise); every other package in `requirements.lock`, and the lock's
  recorded Python minor version, only WARN on a mismatch (a different
  OS/Python -- e.g. Colab -- may legitimately need different transitive
  versions).
- `README.md` and `docs/TEAM_GUIDE.md` updated: the install commands were
  already accurate (the file always existed), but now explain the
  CORE-vs-everything-else policy, the Colab `pip install -e "phase2"`
  flow, and branching from the `phase2-foundation-v1` tag (created by the
  foundation owner after this work merges) instead of from `main` HEAD.
- Foundation lock regenerated (`cli/check_setup.py` is a locked file and
  changed) -- `lock_version` stays **1.0.0**; `contract_dir_sha256`
  unchanged (`311a58cdb40226a3f8744679098686e0497c271b9ff32203f15c1ca7c6d646c6`).

### Updated full test counts (as of this pre-freeze fix)

- Phase 2 full suite (including slow): **461 passed**, 315.92s
- Phase 1 suite: **234 passed**, 17.82s

All green on a clean tree immediately before this fix's commit.

## 9. D18 enforcement in code (branch `phase2/unblinding-tools`)

D18 (simultaneous unblinding, §7/§8) is now enforced in code rather than
only by convention, without changing `configs/contract/`
(`contract_dir_sha256` unchanged, §1):

- **`configs/study.yaml`** (schema-validated via `config.load_study()`,
  part of the foundation lock but NOT of `configs/contract/`): names the
  five architectures in scope and the Phase 1 baseline reference.
- **`cli/register_frozen.py`**: writes TRACKED, provenance-only
  `artifacts/frozen/<model>.json` once every `final_seed`'s run at the
  selected lr is frozen, conformant, clean, and has a passing rehearsal.
  Contains no test information of any kind.
- **`cli/unblinding_status.py`**: READY (exit 0) only when every study
  model has a registration covering every `final_seed` with a
  `contract_dir_sha256` matching the current contract; NOT READY (exit 1)
  otherwise.
- **`cli/final_test.py`** (real mode only -- `--rehearsal` unaffected):
  refuses unless this run's model has a frozen registration whose
  recorded sha256 for this seed matches the checkpoint being tested, AND
  `unblinding_status` is READY.
- **`cli/export_results.py`**: independently re-checks the same
  registration, refusing on a sha mismatch.
- **`cli/efficiency_central.py`**: measures every study model that is
  registered in the catalog (skipping and listing the rest), on one
  device in one process, writing TRACKED
  `artifacts/comparison/efficiency_<device>.json`; `--overwrite` required
  to replace an existing measurement.

**Foundation lock**: regenerated, `lock_version` bumped **1.0.0 → 1.1.0**
(now also covers `configs/study.yaml` and the four new/changed `cli/`
modules above). `contract_dir_sha256` is unchanged.

**efficientnet_b0's three final runs (seeds 42/43/44, lr=3e-4) are now
registered frozen** (`artifacts/frozen/efficientnet_b0.json`).
`unblinding_status` correctly reports **NOT READY**: AlexNet, VGG16,
GoogLeNet, and ResNet18 are not yet registered.

**Central efficiency smoke-tested** (CPU and MPS, on this machine, for
efficientnet_b0 -- the only study model currently in the catalog); the
measurement files were deliberately NOT committed (per D18, the real
comparison measurement happens once all five architectures are
registered) -- deleted after inspection.

### Updated full test counts (as of this enforcement work)

- Phase 2 full suite (including slow + network): **512 passed**, 550.06s
- Phase 2 fast (`-m "not slow"`): **460 passed**, 52 deselected, 81.66s
- Phase 1 suite: **234 passed**, 17.90s

All green on a clean tree immediately before this work's final commit.
`test_access_log.csv` remained header-only throughout.

## 10. CPU timing fix, comparison pipeline, D19 (branch `phase2/comparison`)

### CPU inference-timing fix

`efficiency_central`'s CPU numbers were implausible (~202ms/img at
batch=1, ~215ms/img at batch=32 -- batch=32 should never be slower per
image). Diagnosed with a temporary thread-count/determinism/inference-
mode/memory-format sweep (deleted after use): on this heterogeneous
performance/efficiency-core CPU (Apple M5: 4 performance + 6 efficiency
cores), PyTorch's intra-op thread pool synchronizes across every
requested thread each op -- for a tiny batch=1 input the barrier overhead
dominates and gets WORSE with more threads (measured ~30ms/img at 1
thread → ~86ms/img at 4 (performance-core count) → ~200ms/img at all 10
physical cores, while batch=32 stayed flat throughout).

Fixed: `measure_inference()` now runs under `torch.inference_mode()`,
contiguous memory format (channels_last measured consistently worse), and
`torch.use_deterministic_algorithms(True, warn_only=True)` (same posture
as training's `seed_everything`, D16 -- doesn't alter eval-mode
forward-pass numerics). `efficiency_central`'s CPU thread count now
defaults to the performance-core count (`sysctl hw.perflevel0.physicalcpu`)
when detectable, else physical cores; both the policy and core counts are
recorded. Re-measured for efficientnet_b0: CPU batch=1 202→95ms/img,
batch=32 215→75ms/img (now correctly faster than batch=1); MPS
essentially unchanged (4.4/3.1 → 4.6/3.4ms/img, within noise).

### Cross-model comparison pipeline (D14, D15, D18)

New `src/btdl/comparison/` package (`loaders`, `baseline`, `stats`,
`tables`, `plots`, `report`) and `cli/compare.py --mode {test,val-preview}`:

- **test mode**: built ONLY from exported
  `artifacts/results/<model>/seed<k>/` directories plus the Phase 1
  baseline (`glcm_xgboost_v2`)'s test predictions (sample-id set checked
  for EXACT equality against the Phase 2 test split). Requires
  `unblinding_status` READY and every study model exported; refuses
  (listing what's missing) otherwise. Each seed's checkpoint sha256 is
  re-verified against its D18 frozen registration -- a mismatch is a hard
  refusal (`ComparisonIntegrityError`), never silently treated as
  "missing".
- **val-preview mode**: built from registered models' LOCAL `eval/val/`
  directories only -- never `eval/test/`, never a test-split dataset.
  Every output is labelled **"VALIDATION PREVIEW -- NOT RESULTS"** and
  written to the gitignored `phase2/runs/_preview/comparison/`.
- **Statistics**: per-model seed mean±std (sample std) for every D14
  headline and per-class metric; a pooled patient bootstrap per model
  (seed-averaged metric inside each resample); a paired patient bootstrap
  for every model pair and each model vs. the Phase 1 baseline (CI of the
  seed-averaged-metric difference, on identical resamples -- a model
  paired against itself gives exactly 0.0); size-tertile × class recall,
  seed-averaged; D19 calibration (ECE, 15 equal-width bins, top-label) per
  model, mean±std over seeds, with reliability-diagram data.
- **Outputs** (test mode, tracked): `headline/per_class/paired/
  stratified/calibration/efficiency` tables (CSV+MD), 8 figures, `
  comparison.json` (with content-hashed inputs), and `docs/RESULTS.md`
  with D15's limitation statements and the paired-comparison multiplicity
  caveat baked in; narrative sections are explicit TODO placeholders, not
  auto-generated.
- **Deterministic**: bootstrap settings come from the committed
  `evaluation.yaml` contract; two runs over identical inputs produce
  byte-identical tables and `comparison.json`.

### D19 — supplementary calibration analysis

Added to `DECISIONS.md`: calibration (ECE) is descriptive/supplementary
only and never influences model selection. Process decision, no contract
change.

### Val-preview run (efficientnet_b0, the only registered model)

`python -m btdl.cli.compare --mode val-preview`: `present_models =
["efficientnet_b0"]`, `missing_models = ["alexnet", "vgg16", "googlenet",
"resnet18"]`. Headline (seed mean ± std, matching the earlier seed-table
report exactly): macro-F1 **0.9494 ± 0.0083**, balanced accuracy **0.9508
± 0.0058**, meningioma F1 **0.9284 ± 0.0090**, macro AUC **0.9906 ±
0.0008**, ECE **0.0280 ± 0.0019**. 8 figures generated. Output under the
gitignored `phase2/runs/_preview/comparison/` -- nothing committed (per
D18, the real comparison measurement/output happens once all five
architectures are registered).

### Foundation lock

Regenerated, `lock_version` bumped **1.1.0 → 1.2.0** (now also covers
`src/btdl/comparison/` and `cli/compare.py`; `_LOCKED_SUBDIRS` gained
`"comparison"`). `contract_dir_sha256` unchanged.

### Updated full test counts (as of this comparison-pipeline work)

- Phase 2 full suite (including slow + network): **540 passed**, 749.33s
- Phase 2 fast (`-m "not slow"`): **479 passed**, 61 deselected, 89.63s
- Phase 1 suite: **234 passed**, 19.77s

All green on a clean tree immediately before this work's final commit.
`test_access_log.csv` remained header-only; no tracked files exist under
`artifacts/comparison/` (test mode has not been run -- only val-preview,
which writes to the gitignored `runs/_preview/`).
