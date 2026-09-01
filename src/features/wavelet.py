"""2D Discrete Wavelet Transform feature extraction for Phase 1 tumor ROIs.

Fully driven by the frozen ``configs/features/wavelet.yaml`` specification via
``src.features.config``. No wavelet, level, subband, or statistic here may
drift from that file; if the frozen config changes, this module picks up the
change automatically instead of needing an edit.

pywt.wavedec2 convention (verified empirically against pywt's own naming):
``pywt.wavedec2(image, wavelet, level=2, mode="symmetric")`` returns
``[cA2, (cH2, cV2, cD2), (cH1, cV1, cD1)]``, where cA2 is the level-2
approximation and (cH, cV, cD) are the horizontal/vertical/diagonal detail
coefficients at each level. Empirically, cH responds to horizontal-stripe
images (row-wise intensity variation, i.e. horizontal edges) and cV responds
to vertical-stripe images (column-wise variation, i.e. vertical edges) --
this matches the standard LL/LH/HL/HH subband convention where the label is
(row-filter, col-filter): LH = low-pass rows / high-pass cols = horizontal
detail = cH, and HL = high-pass rows / low-pass cols = vertical detail = cV.
So the deterministic mapping used here is::

    cA2 -> L2_LL
    cH2 -> L2_LH,  cV2 -> L2_HL,  cD2 -> L2_HH
    cH1 -> L1_LH,  cV1 -> L1_HL,  cD1 -> L1_HH

L1_LL (the level-1 approximation) is intentionally not retained -- only the
level-1 detail subbands and the full level-2 decomposition (including L2_LL)
are kept, per the frozen config's ``retained_subbands`` list.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pywt

from src.data.feature_dataset import iter_phase1_samples
from src.features.config import load_feature_config
from src.features.feature_table import (
    build_feature_dataframe,
    build_feature_metadata,
    write_feature_metadata,
    write_feature_table,
)
from src.preprocessing.roi import prepare_tumor_roi

METADATA_COLUMNS = ("sample_id", "patient_id", "label", "split")
SUBBAND_ORDER = ("L1_LH", "L1_HL", "L1_HH", "L2_LL", "L2_LH", "L2_HL", "L2_HH")
STAT_ORDER = ("mean", "std", "energy", "entropy")
ENTROPY_BINS = 64
ENTROPY_EPSILON = 1e-12


def _subband_coefficients(image: np.ndarray, parameters: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run pywt.wavedec2 and return the 7 retained subbands keyed by name."""

    coeffs = pywt.wavedec2(
        image,
        wavelet=parameters["wavelet"],
        level=parameters["level"],
        mode=parameters["mode"],
    )
    if len(coeffs) != 3:
        raise ValueError(f"expected a 2-level wavedec2 decomposition, got {len(coeffs)} levels")
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    return {
        "L1_LH": cH1,
        "L1_HL": cV1,
        "L1_HH": cD1,
        "L2_LL": cA2,
        "L2_LH": cH2,
        "L2_HL": cV2,
        "L2_HH": cD2,
    }


def _energy(coefficients: np.ndarray) -> float:
    """Mean of squared coefficients -- scale-invariant across differently-sized subbands."""

    return float(np.mean(np.square(coefficients)))


def _entropy(coefficients: np.ndarray) -> float:
    """Shannon entropy (nats) over a fixed-bin histogram of coefficient values.

    A histogram over the raw (signed) coefficient values is used rather than
    magnitudes so that entropy reflects the full distribution shape, not just
    its spread. Zero-probability bins are dropped before the log so constant
    or all-zero subbands (a single populated bin) yield a finite entropy of 0
    without any NaN/log(0) risk; a small epsilon guards against residual
    floating-point noise.
    """

    values = np.asarray(coefficients, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    value_range = float(values.max()) - float(values.min())
    if value_range < 1e-9:
        return 0.0
    counts, _ = np.histogram(values, bins=ENTROPY_BINS)
    probabilities = counts.astype(np.float64) / counts.sum()
    nonzero = probabilities[probabilities > ENTROPY_EPSILON]
    if nonzero.size == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log(nonzero)))


def _subband_stats(coefficients: np.ndarray) -> dict[str, float]:
    array = np.asarray(coefficients, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "energy": _energy(array),
        "entropy": _entropy(array),
    }


def extract_wavelet_features(image: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    """Compute the frozen-spec 2D DWT summary statistics for one ROI image."""

    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image_array.shape}")
    if not np.isfinite(image_array).all():
        raise ValueError("image contains NaN or infinite values")

    parameters = config["parameters"]
    prefix = config["feature_prefix"]
    subband_names = parameters["retained_subbands"]
    stat_order = parameters["summary_statistics"]

    subbands = _subband_coefficients(image_array, parameters)
    missing = sorted(set(subband_names).difference(subbands))
    if missing:
        raise ValueError(f"decomposition is missing retained subband(s): {missing}")

    features: dict[str, float] = {}
    for subband_name in SUBBAND_ORDER:
        if subband_name not in subband_names:
            continue
        stats = _subband_stats(subbands[subband_name])
        for stat in STAT_ORDER:
            if stat not in stat_order:
                continue
            feature_name = f"{prefix}_{subband_name.lower()}_{stat}"
            features[feature_name] = stats[stat]

    expected_count = config["expected_feature_count"]
    if len(features) != expected_count:
        raise ValueError(
            f"produced {len(features)} wavelet features, expected {expected_count} per frozen config"
        )
    return features


def _git_commit() -> str:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        return "unknown"


def _split_version(split_metadata_json: Path | str) -> str:
    path = Path(split_metadata_json)
    if not path.is_file():
        return "unknown"
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return str(metadata.get("split_version", "unknown"))


def build_wavelet_feature_table(
    split: str = "all",
    *,
    split_csv: Path | str = "data/splits/patient_split.csv",
    samples_dir: Path | str = "data/processed/samples",
    sample_ids: Iterable[str | int] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract wavelet features for the requested samples via the shared Phase 1 loaders."""

    config = config if config is not None else load_feature_config("wavelet")
    requested_ids = None if sample_ids is None else {str(int(value)) for value in sample_ids}
    records: list[dict[str, Any]] = []
    roi_padding_fraction: float | None = None
    roi_standard_size: tuple[int, int] | None = None

    for sample in iter_phase1_samples(split, split_csv=split_csv, samples_dir=samples_dir):
        if requested_ids is not None and sample.sample_id not in requested_ids:
            continue
        roi = prepare_tumor_roi(sample)
        roi_padding_fraction = roi.padding_fraction
        roi_standard_size = roi.standard_size
        features = extract_wavelet_features(roi.roi_image_masked_resized, config)
        records.append(
            {
                "sample_id": sample.sample_id,
                "patient_id": sample.patient_id,
                "label": sample.label,
                "split": sample.split,
                **features,
            }
        )

    if not records:
        raise ValueError("no samples matched the requested split/sample_ids filters")

    frame = build_feature_dataframe(records)
    roi_info = {
        "padding_fraction": roi_padding_fraction,
        "standard_size": list(roi_standard_size) if roi_standard_size is not None else None,
    }
    return frame, roi_info


def extract_and_write_wavelet_features(
    split: str = "all",
    *,
    split_csv: Path | str = "data/splits/patient_split.csv",
    samples_dir: Path | str = "data/processed/samples",
    sample_ids: Iterable[str | int] | None = None,
    output_csv: Path | str = "data/features/wavelet/wavelet_features.csv",
    metadata_json: Path | str = "data/features/wavelet/wavelet_metadata.json",
    split_metadata_json: Path | str = "data/splits/split_metadata.json",
) -> dict[str, Any]:
    """Extract wavelet features per the frozen config and write the validated outputs."""

    config = load_feature_config("wavelet")
    sample_ids_list = None if sample_ids is None else list(sample_ids)
    run_scope = (
        "full_dataset"
        if sample_ids_list is None
        else f"partial_subset_{len(sample_ids_list)}_samples"
    )
    frame, roi_info = build_wavelet_feature_table(
        split,
        split_csv=split_csv,
        samples_dir=samples_dir,
        sample_ids=sample_ids_list,
        config=config,
    )
    validated = write_feature_table(frame, output_csv, canonical_split=split_csv)
    feature_columns = [
        column for column in validated.columns if column not in METADATA_COLUMNS
    ]
    validation_summary = {
        "sample_count": int(len(validated)),
        "feature_count": len(feature_columns),
        "expected_feature_count": config["expected_feature_count"],
        "samples_by_split": {
            str(key): int(value) for key, value in validated["split"].value_counts().items()
        },
        "samples_by_label": {
            str(key): int(value)
            for key, value in validated["label"].value_counts().sort_index().items()
        },
    }
    standard_size = tuple(roi_info["standard_size"]) if roi_info["standard_size"] else "unknown"
    metadata = build_feature_metadata(
        feature_set_name=config["feature_set_name"],
        feature_version=config["feature_version"],
        algorithm=f"{config['algorithm']} (PyWavelets pywt.wavedec2, per frozen configs/features/wavelet.yaml)",
        parameters={
            **config["parameters"],
            "energy_definition": "mean of squared coefficients per subband",
            "entropy_definition": (
                f"Shannon entropy (nats) over a {ENTROPY_BINS}-bin histogram of raw "
                "coefficient values"
            ),
            "status": f"compliant with frozen configs/features/wavelet.yaml (feature_version {config['feature_version']})",
            "run_scope": run_scope,
        },
        input_representation=(
            f"{config['input_representation']} (TumorROI.roi_image_masked_resized, "
            f"tumor-masked, [0,1]-normalized, {standard_size} pixels)"
        ),
        roi_policy=(
            f"{config['roi_policy']} (prepare_tumor_roi() actual: "
            f"padding_fraction={roi_info['padding_fraction']}, standard_size={standard_size}, "
            "zero-padding for out-of-bounds)"
        ),
        source_split_version=_split_version(split_metadata_json),
        sample_count=int(len(validated)),
        feature_columns=feature_columns,
        code_commit=_git_commit(),
        validation_summary=validation_summary,
    )
    write_feature_metadata(metadata, metadata_json)
    return {"table": validated, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--split-csv", type=Path, default=Path("data/splits/patient_split.csv"))
    parser.add_argument("--samples-dir", type=Path, default=Path("data/processed/samples"))
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument(
        "--output-csv", type=Path, default=Path("data/features/wavelet/wavelet_features.csv")
    )
    parser.add_argument(
        "--metadata-json", type=Path, default=Path("data/features/wavelet/wavelet_metadata.json")
    )
    args = parser.parse_args()
    result = extract_and_write_wavelet_features(
        args.split,
        split_csv=args.split_csv,
        samples_dir=args.samples_dir,
        sample_ids=args.sample_ids,
        output_csv=args.output_csv,
        metadata_json=args.metadata_json,
    )
    print(json.dumps(result["metadata"]["validation_summary"], indent=2))


if __name__ == "__main__":
    main()
