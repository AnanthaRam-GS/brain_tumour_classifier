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

## Running tests

```bash
cd phase2
.venv/bin/python -m pytest -q
```

(equivalently, from the repo root: `phase2/.venv/bin/python -m pytest -q phase2/tests`)

## Decisions

All frozen Phase 2 design decisions (data source, split, ROI geometry,
input/augmentation spec, training recipe, evaluation contract) are recorded
in [`docs/DECISIONS.md`](docs/DECISIONS.md). Read it before writing any
preprocessing, training, or evaluation code.

Teammates: do not edit `configs/contract/` or
`src/btdl/{data,preprocessing,training,evaluation}` without going through
the decision-change process in `docs/DECISIONS.md`.
