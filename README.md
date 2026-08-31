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

## Phase 1 patient split

Phase 1 uses one definitive patient-level train/validation/test split for all
classical feature extraction and machine-learning experiments. The split is
70/15/15 over unique `patient_id` values, uses random seed `42`, and is
stratified by the patient's tumor class. Slice-level splitting is forbidden:
multiple MRI slices can come from the same patient, so splitting slices
directly would leak patient-specific information into validation or test
results.

Generate the shared split with:

```bash
python -m src.data.create_patient_split
```

The generated files are:

- `data/splits/patient_split.csv`
- `data/splits/split_metadata.json`

`1512427/cvind.mat` was inspected and found to be patient-disjoint under the
dataset's numeric sample ordering, but it is retained only as historical and
reference metadata. It is not used to generate the Phase 1 split.

All feature extraction and ML code must consume `data/splits/patient_split.csv`.
Individual modules must never call their own `train_test_split` over slices.

## Phase 1 data contract

The canonical analytical input for handcrafted features is
`data/processed/samples/{sample_id}.npz`, joined with the shared split in
`data/splits/patient_split.csv`. Feature code must use the common loader so
that sample IDs, patient IDs, labels, split membership, image arrays, and masks
are validated consistently.

```python
from src.data.feature_dataset import iter_phase1_samples

for sample in iter_phase1_samples("train"):
    image = sample.image_normalized
    mask = sample.tumor_mask
```

Feature modules must not create their own splits, read raw MATLAB files, or
depend on PNG derivatives. ROI handling and any additional preprocessing belong
in the common preprocessing layer, not inside individual feature extractors.

## Phase 1 ROI contract

Tumor ROI preprocessing is deterministic, sample-local, and performed in
memory. It uses `sample.image_normalized` as the source image and
`sample.tumor_mask` as the source annotation. The tight tumor bounding box is
expanded to a centered square, padded with 10% contextual tissue on each side,
and zero-padded at MRI boundaries when the desired square extends outside the
image. ROI files are not persisted to disk.

The shared ROI object keeps native-size views and standardized 128x128 views.
Images are resized with bilinear interpolation, masks with nearest-neighbor
interpolation, and resized masks are forced back to binary values. Both
unmasked and masked ROI images are exposed; masked images are computed by
multiplying the image by the final binary mask.

```python
from src.data.feature_dataset import iter_phase1_samples
from src.preprocessing.roi import prepare_tumor_roi

for sample in iter_phase1_samples("train"):
    roi = prepare_tumor_roi(sample)
    native_image = roi.roi_image
    native_mask = roi.roi_mask
    fixed_image = roi.roi_image_masked_resized
```

Future Phase 1 feature modules should use the shared ROI contract consistently:
GLCM and LBP operate on native ROI data with tumor-mask-aware logic; DWT,
HOG, and Gabor operate on standardized 128x128 masked ROIs; geometry uses the
native tumor mask; intensity uses normalized image values from tumor-mask
pixels. These algorithms are implemented separately from the ROI layer.
