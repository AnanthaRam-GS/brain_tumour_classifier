# Brain Tumour MRI Computer Vision Project

This project investigates computer-vision methods for analysing brain tumour
MRI images from the original Figshare brain tumour dataset. The workflow covers
validated conversion, preprocessing, classical feature engineering, model
development, and evaluation.

## Dataset

The raw dataset is stored locally at `1512427/`. Its four numbered
subdirectories contain cases `1.mat` through `3064.mat`; `cvind.mat` and the
dataset README are also kept at the root of that directory.

All files under `1512427/` are immutable source data. Keeping this original
evidence unchanged makes every derived result reproducible and prevents an
accidental preprocessing operation from corrupting the only source copy. Raw
files must not be moved, renamed, modified, or deleted. Derived artifacts belong
under `data/` and are excluded from version control.

## Processed-data layout

- `data/processed/samples/`: canonical compressed NPZ sample records.
- `data/processed/images/`: normalized 8-bit PNG visual derivatives.
- `data/processed/masks/`: binary 8-bit PNG visual derivatives.
- `data/processed/audit_overlays/`: 12 deterministic conversion checks.
- `data/processed/manifest.csv`: validated metadata for converted samples.
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

## Canonical dataset conversion

NPZ is the canonical working format because it preserves NumPy dtypes, native
array shapes, binary masks, labels, identifiers, and tumour-border arrays in a
single compressed sample file. Each NPZ contains both the untouched-valued
`float32` MRI (`image_raw`) and its per-image robust foreground-percentile
normalization (`image_normalized`). No image or mask is resized during
conversion: the 512×512 and 256×256 source resolutions remain unchanged so
that conversion does not introduce interpolation artifacts or alter mask
geometry.

The PNG images and masks are lossless visual derivatives for inspection and
external viewers. Their 8-bit representation is not an analytical source;
downstream computation should load the NPZ files.

Run a small restart-safe conversion before the complete dataset:

```bash
python -m src.data.convert_dataset \
  --dataset-root 1512427 \
  --raw-manifest reports/sample_manifest_raw.csv \
  --output-root data/processed \
  --report-path reports/dataset_conversion.json \
  --limit 10
```

Specific samples can be selected with `--sample-ids 1 2 3`. Add `--overwrite`
only when valid existing outputs should deliberately be regenerated. Without
it, complete valid samples are skipped and incomplete or invalid samples are
repaired.

Run the complete conversion with:

```bash
python -m src.data.convert_dataset \
  --dataset-root 1512427 \
  --raw-manifest reports/sample_manifest_raw.csv \
  --output-root data/processed \
  --report-path reports/dataset_conversion.json
```

Expected generated layout:

```text
data/processed/
├── samples/          # {sample_id}.npz: canonical analytical records
├── images/           # {sample_id}.png: normalized visual derivatives
├── masks/            # {sample_id}.png: 0/255 mask visual derivatives
├── audit_overlays/   # exactly 12 deterministic class-balanced figures
└── manifest.csv      # numerically ordered conversion metadata
```

The run summary and validation findings are written to
`reports/dataset_conversion.json`.
