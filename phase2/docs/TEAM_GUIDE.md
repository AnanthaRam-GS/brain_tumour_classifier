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

# Get the ROI cache -- download link provided by the foundation owner.
# Put roi224_v1.npy wherever you like, then either:
#   a) place it at phase2/cache/roi224_v1.npy (the default), or
#   b) export BTDL_CACHE_DIR=/path/to/your/cache/dir
# You do NOT need the raw dataset or the per-sample NPZs -- the cache is
# the only large asset.

cd phase2
.venv/bin/python -m btdl.cli.check_setup
```

Every line of `check_setup`'s checklist should read `[PASS]`
(`working tree clean` may `[WARN]` on an unrelated dirty file — that's
fine, it's a warning, not a failure). Fix anything `[FAIL]` before you
start training.

## 4. Running in Colab / Kaggle

Same commands, as shell cells, with the cache on Drive so it survives a
runtime restart:

```python
!git clone <repo-url> /content/repo
%cd /content/repo
!python3.12 -m venv phase2/.venv
!phase2/.venv/bin/pip install -e phase2 --no-deps

from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["BTDL_CACHE_DIR"] = "/content/drive/MyDrive/btdl_cache"  # contains roi224_v1.npy

!cd phase2 && .venv/bin/python -m btdl.cli.check_setup
```

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

# 1. Smoke run (2 epochs, non-conformant by design) -- confirms the whole
#    pipeline runs end to end for your model before a real run.
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
.venv/bin/python -m btdl.cli.final_test --run-dir $RUN --confirm-final-test
.venv/bin/python -m btdl.cli.export_results --run-dir $RUN
```

**The test split is used exactly once per final run, by `final_test`.**
Never re-run `final_test` on the same `run_dir` (it refuses a repeat run
anyway, via `artifacts/test_access_log.csv`), and never retrain or retune
after seeing test results — that silently turns the test split into a
validation set.

## 7. Git workflow

```bash
git switch main && git pull
git switch -c phase2/model-<your_model>
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
