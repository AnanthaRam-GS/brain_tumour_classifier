# Phase 2 team guide

You're implementing one of the five remaining architectures (AlexNet,
VGG16, GoogLeNet, ResNet18, EfficientNet-B0) on top of a frozen Phase 2
foundation. This document is the whole onboarding path: what you may touch,
how to set up, how to add your model, and the exact command sequence from a
smoke run to exported results. Read [`DECISIONS.md`](DECISIONS.md) and
[`FOUNDATION_FREEZE.md`](FOUNDATION_FREEZE.md) first if anything here
assumes context you don't have yet.

## 1. Rules

The foundation is frozen. Do **not** change, even locally for
experimentation:

- `configs/contract/*.yaml` (data, input, augmentation, training,
  evaluation contracts)
- the frozen split (`data/splits/patient_split.csv`)
- ROI geometry/cache code (`src/btdl/preprocessing/`, `src/btdl/data/`)
- normalisation (`src/btdl/preprocessing/model_input.py`)
- the training protocol (`src/btdl/training/`, `cli/train.py`,
  `cli/select_lr.py`)
- evaluation (`src/btdl/evaluation/`, `cli/evaluate.py`,
  `cli/final_test.py`)
- test-set access policy (D13, D17 in `DECISIONS.md`)
- the registration-before-test gate (D18 in `DECISIONS.md`) — see §6:
  `final_test` (without `--rehearsal`) runs for a model only once that
  model's own frozen runs are registered (`cli/register_frozen.py`,
  committed); you do NOT need to wait for the other four architectures

`python -m btdl.cli.lock --check` fails loudly (and CI/your own test run
will fail) if any of this changes without a regenerated, reviewed lock —
see `CODEOWNERS`. If you genuinely believe one of these needs to change,
that's a decision-process change (`DECISIONS.md`), not a quiet edit: open
it with the foundation owner.

## 2. What you may add or edit

Exactly:

- `src/btdl/models/<your_model>.py` — one new module
- **one** entry in `src/btdl/models/catalog.py` (`MODEL_REGISTRY`)
- `configs/models/<your_model>.yaml` — the matching strict config
- optionally, `tests/models/test_<your_model>.py`

Nothing else. `models/catalog.py` and every `models/<name>.py` file are
explicitly *not* locked (see `CODEOWNERS`) so you can do this without
waiting on review from the foundation owner.

## 3. Setup

```bash
git clone <repo-url>
cd <repo>
python3.12 -m venv phase2/.venv
phase2/.venv/bin/pip install -e phase2 --no-deps  # pulls pyproject.toml's pinned deps

# Get the ROI cache:
curl -L -o roi224_v1.npy \
  https://github.com/AnanthaRam-GS/brain_tumour_classifier/releases/download/phase2-foundation-v1/roi224_v1.npy
# Then either:
#   a) place it at phase2/cache/roi224_v1.npy (the default), or
#   b) export BTDL_CACHE_DIR=/path/to/your/cache/dir
# Never rebuild the cache yourself (cli/build_cache.py is foundation-owner
# only) -- check_setup performs a full sha256 verification against the
# tracked metadata, so a corrupted or wrong-version download is caught
# automatically. Expected npy sha256:
#   d9860739dc6fe5cec0c806d30e2630c4d3df24fe3f0d33c639ff4b3f88665735
# You do NOT need the raw dataset or the per-sample NPZs -- the cache is
# the only large asset.

cd phase2
.venv/bin/python -m btdl.cli.check_setup
```

Every line of `check_setup`'s checklist should read `[PASS]`
(`working tree clean` may `[WARN]` on an unrelated dirty file, and
non-CORE package version mismatches may `[WARN]` — see §4 — that's fine,
those are warnings, not failures). Fix anything `[FAIL]` before you start
training.

## 4. Running in Colab / Kaggle

Same commands, as shell cells, with the cache on Drive so it survives a
runtime restart:

```python
!git clone <repo-url> /content/repo
%cd /content/repo
!pip install -e "phase2"  # installs with pyproject.toml's pinned CORE packages

from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["BTDL_CACHE_DIR"] = "/content/drive/MyDrive/btdl_cache"  # contains roi224_v1.npy

!cd phase2 && python -m btdl.cli.check_setup
```

`pip install -e "phase2"` (no `--no-deps`) resolves and installs
pyproject.toml's pinned **CORE** packages (torch, torchvision, numpy,
scikit-learn, scipy, h5py, pyyaml, pandas, matplotlib) itself -- those
must match exactly, so `check_setup` `[FAIL]`s if pip resolved a different
version for any of them. Everything else in `requirements.lock` (the
transitive dependencies pinned in the original dev environment) is
expected to drift on Colab's own base image -- `check_setup` only `[WARN]`s
on those, and on a Python minor-version difference from the lock's
recorded version. **Resolve every `[FAIL]`; a `[WARN]` is informational
and does not block training.**

To survive a disconnect mid-run: `runs/` has no env-var override, so copy
the run directory to Drive yourself after each session and copy it back
before resuming --

```python
!cp -r phase2/runs/<your_model>/lr<lr>_seed<seed> /content/drive/MyDrive/btdl_runs/
# ... later, after a fresh clone/runtime restart ...
!cp -r /content/drive/MyDrive/btdl_runs/lr<lr>_seed<seed> phase2/runs/<your_model>/
```

`cli/train.py --resume` only needs `last.pt` and `metadata.json` back in
place at the same `run_dir` -- it refuses to resume if `lr`/`seed`/the
effective config/the contract hash differ from what the checkpoint
recorded (see §8).

## 5. Implementing your model

Every architecture is a thin wrapper around the shared helper
(`src/btdl/models/torchvision_common.py`, implements D8 once: full
fine-tuning, no frozen backbone, final layer replaced with
`nn.Linear(in_features, 3)`):

```python
"""<ModelName>: torchvision ImageNet-pretrained, full fine-tuning, 3-class head (D8)."""

from torchvision.models import <Weights_Enum>, <builder_fn>

from btdl.models.torchvision_common import build_torchvision_classifier


def build_<your_model>(pretrained: bool = True):
    return build_torchvision_classifier(
        builder=<builder_fn>,
        weights=<Weights_Enum>.IMAGENET1K_V1,
        head_path="<head_path>",
        pretrained=pretrained,
        builder_kwargs=<builder_kwargs>,  # omit if {}
    )
```

Per-architecture values (verified against this torchvision version by
`tests/test_torchvision_common.py`):

| Architecture      | `builder_fn`                           | `head_path`      | `builder_kwargs`          | weights enum                    |
|--------------------|----------------------------------------|-------------------|----------------------------|----------------------------------|
| AlexNet            | `torchvision.models.alexnet`           | `"classifier.6"`  | `{}`                       | `AlexNet_Weights`                |
| VGG16              | `torchvision.models.vgg16`             | `"classifier.6"`  | `{}`                       | `VGG16_Weights`                  |
| GoogLeNet          | `torchvision.models.googlenet`         | `"fc"`            | `{"aux_logits": False}`    | `GoogLeNet_Weights`              |
| ResNet18           | `torchvision.models.resnet18`          | `"fc"`            | `{}`                       | `ResNet18_Weights`               |
| EfficientNet-B0    | `torchvision.models.efficientnet_b0`   | `"classifier.1"`  | `{}`                       | `EfficientNet_B0_Weights`        |

GoogLeNet note (D8): `aux_logits=False` is required so its `forward()`
returns a single `[B, 3]` tensor like every other model, instead of an
auxiliary-logits tuple — `cli/train.py`'s trainer rejects non-tensor
outputs. Its shipped `transform_input` is left exactly as torchvision
ships it.

Then register it in `catalog.py`:

```python
"<your_model>": ModelSpec(
    name="<your_model>",
    version="1.0.0",
    builder=build_<your_model>,
    weights_id="<Weights_Enum>.IMAGENET1K_V1",
    reference_only=False,
    description="<one line>",
),
```

and add the matching `configs/models/<your_model>.yaml` (same five keys
as `configs/models/tiny_cnn.yaml`: `name`, `version`, `weights_id`,
`reference_only`, `description` — exactly, no hyperparameters; the loader
rejects any other key).

Before you train anything, `tests/test_model_contract.py` must pass for
your model (it's parametrized over every registered model automatically —
no changes needed there):

```bash
.venv/bin/python -m pytest -q tests/test_model_contract.py
```

It checks: `[B, 3]` tensor output in both train/eval mode (no aux tuple),
all parameters fp32, deterministic eval-mode forward, at least one
trainable parameter, and that your YAML matches your registry entry
exactly.

## 6. Workflow — exact commands, in order

```bash
cd phase2

# 0. Sanity check before spending any compute.
.venv/bin/python -m btdl.cli.check_setup
.venv/bin/python -m pytest -q tests/test_model_contract.py

# 1. Smoke run (SMOKE_MAX_EPOCHS epochs, non-conformant by design) --
#    confirms the whole pipeline runs end to end for your model before a
#    real run.
.venv/bin/python -m btdl.cli.train --model <your_model> --lr 1e-3 --seed 42 --smoke

# 2. Grid search: one run per lr in training.yaml's lr_grid, at grid_seed.
.venv/bin/python -m btdl.cli.train --model <your_model> --lr 1e-4 --seed 42
.venv/bin/python -m btdl.cli.train --model <your_model> --lr 3e-4 --seed 42
.venv/bin/python -m btdl.cli.train --model <your_model> --lr 1e-3 --seed 42

# 3. Select the best lr (macro-F1, then val-loss, then lr, as tie-breaks).
.venv/bin/python -m btdl.cli.select_lr --model <your_model>
#    writes artifacts/selection/<your_model>.json -- commit this file.

# 4. Re-run at the selected lr for the two remaining final seeds.
.venv/bin/python -m btdl.cli.train --model <your_model> --lr <selected_lr> --seed 43
.venv/bin/python -m btdl.cli.train --model <your_model> --lr <selected_lr> --seed 44

# 5. For EACH of the three final runs (seeds 42, 43, 44 at the selected lr):
RUN=runs/<your_model>/lr<selected_lr>_seed<seed>
.venv/bin/python -m btdl.cli.freeze     --run-dir $RUN
.venv/bin/python -m btdl.cli.efficiency --run-dir $RUN
.venv/bin/python -m btdl.cli.final_test --run-dir $RUN --confirm-final-test --rehearsal

# 6. Once all three final runs are frozen + rehearsed, register them (D18).
.venv/bin/python -m btdl.cli.register_frozen --model <your_model>
git add artifacts/selection/<your_model>.json artifacts/frozen/<your_model>.json
git commit -m "results(<your_model>): register frozen final runs"
#    Open a PR with exactly these two files. You do NOT need to wait for
#    the other four architectures -- once this is merged to main, you may
#    test your model immediately.

# 7. For EACH of the three final runs: check readiness first (recommended --
#    it runs every real-mode gate and writes nothing, so it's always safe
#    to run before you've decided you're really ready), then run for real,
#    exactly once each:
.venv/bin/python -m btdl.cli.final_test --run-dir $RUN --check-only
.venv/bin/python -m btdl.cli.final_test --run-dir $RUN --confirm-final-test
.venv/bin/python -m btdl.cli.export_results --run-dir $RUN
#    No commit is needed between these calls -- see "D18 — registration
#    before final test" below.

# 8. Commit the log and your results together, and open a PR.
git add artifacts/test_access_log.csv artifacts/results/<your_model>/
git commit -m "results(<your_model>): final test results"
```

**The test split is used exactly once per final run, by `final_test`
(without `--rehearsal`).** Never re-run `final_test` on the same `run_dir`
(it refuses a repeat run anyway, via `artifacts/test_access_log.csv`), and
**never retrain or re-run training after seeing test results** — that
silently turns the test split into a validation set and invalidates the
whole point of the gate.

**D18 — registration before final test (enforced in code).** `final_test`
(without `--rehearsal`) refuses to run unless your model has a frozen
registration (`artifacts/frozen/<your_model>.json`, written and committed
in step 6) whose recorded `best_pt_sha256` matches the exact checkpoint
being tested. That is the ONLY cross-model condition -- there is no
"wait for everyone else" gate: you may run `final_test` for your model as
soon as your own registration is merged, regardless of whether the other
four architectures are registered yet. `--rehearsal` is never affected by
any of this (it never touches the test split, and works even before
you've registered anything). `export_results` independently re-checks
the same registration before copying anything.

**`--check-only` (readiness check, writes nothing).** `final_test
--run-dir $RUN --check-only` runs every real-mode gate above, in the same
order, prints PASS/FAIL for each one, and exits non-zero if any gate
fails -- but it never constructs the test dataset, loads the model, or
writes anything (no log row, no `eval/test/`, no file of any kind). Run
it as many times as you like before your real final-test call; it is
mutually exclusive with `--confirm-final-test` and incompatible with
`--rehearsal` (rehearsal has no real-mode gates to check).

**Ordering (step 7, no intermediate commits needed):** run `final_test`
for all three final seeds first, then `export_results` for each, then
commit the log and the results together (step 8) in one PR.
`phase2/artifacts/test_access_log.csv` is specifically exempt from the
clean-tree check while its only changes are well-formed appended rows (an
exact byte prefix of what's committed, plus rows with the right column
count) -- so three `final_test` calls in a row do not need a commit
between them. Any OTHER change to the log (an edit, a deletion, rows
reordered, a malformed row) is still treated as dirty, same as before.
`export_results` has never required a clean tree.

The efficiency numbers used in the final cross-model comparison are
re-measured centrally by the foundation owner, for all five architectures
together on one machine (`cli/efficiency_central.py`); your own
`efficiency.json` from step 5 is kept as provenance but is not what gets
compared. `python -m btdl.cli.unblinding_status` (name kept for backward
compatibility) still exists, but it no longer gates `final_test` -- it
only reports whether the full five-model comparison
(`cli/compare.py --mode test`) is ready to run, which legitimately does
need every architecture present.

**Getting these tools.** If your branch predates this foundation update,
fetch and merge the latest foundation tag to pick up `register_frozen`/
`final_test`'s registration gate/the append-only log exemption without
losing your work:

```bash
git fetch --tags
git merge phase2-foundation-v1.3
```

Your existing frozen runs stay valid -- `configs/contract/` did not
change (`contract_dir_sha256` is identical), so nothing needs retraining;
you only need to run `register_frozen` (step 6) once you have it.

## 7. Git workflow

Branch from `phase2-foundation-v1`, the tag the foundation owner creates
on `main` once this handoff work is merged -- not from whatever `main`
happens to be at the time, so your branch's foundation state stays pinned
even if `main` moves on:

```bash
git fetch --tags
git switch -c phase2/model-<your_model> phase2-foundation-v1
# ... implement, train, select, freeze, final_test, export_results ...
git add src/btdl/models/<your_model>.py \
        src/btdl/models/catalog.py \
        configs/models/<your_model>.yaml \
        artifacts/selection/<your_model>.json \
        artifacts/results/<your_model>/
git commit -m "feat(phase2): <your_model>"
```

- Your PR touches only the paths listed in §2, plus
  `artifacts/selection/<your_model>.json` and
  `artifacts/results/<your_model>/`.
- **Never commit `runs/` or any checkpoint** (`*.pt`) — both are
  gitignored; `export_results` already copied everything reviewable out
  of `runs/` for you.
- If `python -m btdl.cli.lock --check` fails on your branch, you touched
  a locked file by accident — revert that change, don't regenerate the
  lock yourself (that requires the foundation owner's review).

## 8. Troubleshooting

- **MPS gives slightly different numbers between identical runs.**
  Expected — MPS is not bitwise-deterministic even with a fixed seed (see
  D16 and `FOUNDATION_FREEZE.md`'s determinism probe). Small run-to-run
  noise in `history.csv` is normal; it is not a bug in your model code.
- **CUDA OOM.** Do not change `batch_size` (it's part of the frozen
  training contract) — report it to the foundation owner instead. Lower
  `--num-workers` or use a smaller/cheaper instance if that's an option
  for you.
- **Training was interrupted (Colab disconnect, laptop sleep, etc).**
  Resume with the identical arguments plus `--resume`:
  `python -m btdl.cli.train --model <your_model> --lr <lr> --seed <seed> --resume`.
  It refuses to resume if `lr`/`seed`/the effective config/the contract
  hash differ from what the checkpoint recorded — that's by design (see
  D11/D16), not something to work around.
- **`check_setup` fails.** Read the `[FAIL]` line's detail message first —
  it names exactly what's wrong (missing cache, stale lock, wrong
  versions, etc.) and usually what to run to fix it.
