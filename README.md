# Brain Tumour MRI Computer Vision Project

This project investigates computer-vision methods for analysing brain tumour
MRI images from the original Figshare brain tumour dataset. The planned
workflow covers image and mask extraction, preprocessing, classical feature
engineering, model development, and evaluation. This initialization step does
not process any dataset files.

## Dataset

The raw dataset is stored locally at `1512427/`. Its four numbered
subdirectories contain cases `1.mat` through `3064.mat`; `cvind.mat` and the
dataset README are also kept at the root of that directory.

All files under `1512427/` are immutable source data. They must not be moved,
renamed, modified, or deleted. Derived artifacts belong under `data/` and are
excluded from version control.

## Processed-data layout

- `data/processed/samples/`: normalized per-case sample records or bundles.
- `data/processed/images/`: extracted and preprocessed MRI images.
- `data/processed/masks/`: extracted and preprocessed tumour masks.
- `data/processed/overlays/`: image-and-mask overlays used for validation.
- `data/splits/`: reproducible train, validation, and test split definitions.
- `data/features/`: tabular classical feature datasets.

Generated report images belong in `reports/sample_overlays/` and
`reports/figures/`. Project-relative locations are defined in
`configs/paths.yaml`.

## Raw dataset audit

The audit validates every raw MATLAB sample independently and writes a raw
manifest, summary tables, JSON diagnostics, and deterministic sample figures:

```bash
python -m src.data.audit_dataset --dataset-root 1512427
```

The command reads but never modifies the raw dataset. A malformed sample is
recorded in the audit outputs without terminating the remaining audit.
