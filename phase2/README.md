# btdl — Phase 2 (deep learning)

Phase 2 deep-learning package for the brain tumor MRI classifier. Isolated
from Phase 1 (`src/`): its own package (`btdl`), its own virtual environment
(`phase2/.venv`), and Phase 2 runtime code never imports Phase 1 code.

## Setup

```bash
python3.12 -m venv phase2/.venv
phase2/.venv/bin/pip install -r phase2/requirements.lock
phase2/.venv/bin/pip install -e phase2 --no-deps
```

`phase2/requirements.lock` is the full pinned dependency set (`pip freeze
--exclude-editable`, header comment records the Python version/platform it
was generated on); `pyproject.toml` pins only the CORE packages
(torch/torchvision/numpy/scikit-learn/scipy/h5py/pyyaml/pandas/matplotlib)
that training/evaluation correctness depends on. `python -m
btdl.cli.check_setup` verifies both after install -- see
[`docs/TEAM_GUIDE.md`](docs/TEAM_GUIDE.md) for its full checklist and the
CORE-vs-everything-else version policy.

## Running tests

```bash
cd phase2
.venv/bin/python -m pytest -q
```

(equivalently, from the repo root: `phase2/.venv/bin/python -m pytest -q phase2/tests`)

For fast iteration, skip the `slow` tests (real-data full passes, default-size
bootstrap, trainer convergence runs -- anything over ~5s):

```bash
.venv/bin/python -m pytest -q -m "not slow"
```

**The full suite (no `-m` filter) is required before every merge** -- `-m "not
slow"` is for iteration only.

## Decisions

All frozen Phase 2 design decisions (data source, split, ROI geometry,
input/augmentation spec, training recipe, evaluation contract) are recorded
in [`docs/DECISIONS.md`](docs/DECISIONS.md). Read it before writing any
preprocessing, training, or evaluation code.

Teammates: do not edit `configs/contract/` or
`src/btdl/{data,preprocessing,training,evaluation}` without going through
the decision-change process in `docs/DECISIONS.md`.

## Implementing a model

If you're adding one of the five architectures (AlexNet, VGG16,
GoogLeNet, ResNet18, EfficientNet-B0), start at
[`docs/TEAM_GUIDE.md`](docs/TEAM_GUIDE.md) -- it covers setup, what you
may edit, the model implementation template, and the exact command
sequence from a smoke run to exported results.
