# Phase 2 Foundation Freeze

This document records the state of the Phase 2 common foundation as of commit
`45465fb6` on branch `phase2/registry-smoke`, after the real-data smoke
experiment validated the full pipeline end-to-end on this machine's default
device (Apple MPS). Everything described here is frozen: later work (model
architectures, grid runs, final test-set evaluation) builds on top of it
without modifying it.

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
