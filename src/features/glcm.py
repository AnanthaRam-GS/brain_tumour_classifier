"""Mask-aware GLCM feature extraction for the Phase 1 baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_dataset import Phase1Sample, iter_phase1_samples, load_phase1_sample
from src.features.config import load_feature_config
from src.features.feature_table import (
    build_feature_metadata,
    build_feature_dataframe,
    validate_feature_table,
    write_feature_metadata,
    write_feature_table,
)
from src.preprocessing.roi import TumorROI, prepare_tumor_roi

GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
GLCM_AGGREGATIONS = ("mean", "std")
GLCM_FEATURE_COLUMNS = [
    f"glcm_{prop.lower() if prop != 'ASM' else 'asm'}_{aggregation}"
    for prop in GLCM_PROPERTIES
    for aggregation in GLCM_AGGREGATIONS
]

ANGLE_OFFSETS = {
    0: lambda distance: (0, distance),
    45: lambda distance: (-distance, distance),
    90: lambda distance: (-distance, 0),
    135: lambda distance: (-distance, -distance),
}


class GLCMExtractionError(ValueError):
    """Raised when mask-aware GLCM extraction cannot produce valid features."""


def quantize_glcm_image(image: np.ndarray, *, gray_levels: int = 32) -> np.ndarray:
    """Uniformly quantize normalized image values in [0, 1] to integer gray levels."""

    if gray_levels <= 1:
        raise GLCMExtractionError("gray_levels must be greater than 1")
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 2:
        raise GLCMExtractionError(f"image must be 2D, got shape {image_array.shape}")
    if not np.isfinite(image_array).all():
        raise GLCMExtractionError("image contains NaN or infinite values")
    tolerance = float(np.finfo(np.float32).eps * 16)
    if float(image_array.min()) < -tolerance or float(image_array.max()) > 1.0 + tolerance:
        raise GLCMExtractionError("image values must be within [0, 1]")
    quantized = np.floor(np.clip(image_array, 0.0, 1.0) * gray_levels)
    quantized = np.clip(quantized, 0, gray_levels - 1)
    return quantized.astype(np.uint8 if gray_levels <= np.iinfo(np.uint8).max + 1 else np.uint16)


def offset_for_angle(distance: int, angle_degrees: int) -> tuple[int, int]:
    """Return row/column offset for the frozen image-coordinate angle convention."""

    if distance <= 0:
        raise GLCMExtractionError("distance must be positive")
    if angle_degrees not in ANGLE_OFFSETS:
        raise GLCMExtractionError(f"unsupported GLCM angle: {angle_degrees}")
    return ANGLE_OFFSETS[angle_degrees](int(distance))


def build_masked_glcm(
    quantized_image: np.ndarray,
    mask: np.ndarray,
    *,
    gray_levels: int,
    distance: int,
    angle_degrees: int,
    symmetric: bool = True,
    normed: bool = True,
) -> tuple[np.ndarray, int]:
    """Build one GLCM using only pairs where both endpoints are inside the tumor mask."""

    image = np.asarray(quantized_image)
    mask_array = np.asarray(mask).astype(bool)
    if image.ndim != 2 or mask_array.ndim != 2:
        raise GLCMExtractionError("quantized_image and mask must be 2D")
    if image.shape != mask_array.shape:
        raise GLCMExtractionError(f"image and mask shapes differ: {image.shape} != {mask_array.shape}")
    if image.size and (int(image.min()) < 0 or int(image.max()) >= gray_levels):
        raise GLCMExtractionError("quantized image contains values outside configured gray levels")

    row_offset, col_offset = offset_for_angle(distance, angle_degrees)
    rows, cols = image.shape
    matrix = np.zeros((gray_levels, gray_levels), dtype=np.float64)
    valid_pairs = 0

    row_start = max(0, -row_offset)
    row_end = min(rows, rows - row_offset)
    col_start = max(0, -col_offset)
    col_end = min(cols, cols - col_offset)
    if row_start >= row_end or col_start >= col_end:
        return matrix, 0

    src = np.s_[row_start:row_end, col_start:col_end]
    dst = np.s_[row_start + row_offset : row_end + row_offset, col_start + col_offset : col_end + col_offset]
    valid = mask_array[src] & mask_array[dst]
    if not valid.any():
        return matrix, 0

    source_values = image[src][valid].astype(np.int64, copy=False)
    neighbor_values = image[dst][valid].astype(np.int64, copy=False)
    np.add.at(matrix, (source_values, neighbor_values), 1.0)
    valid_pairs = int(source_values.size)
    if symmetric:
        np.add.at(matrix, (neighbor_values, source_values), 1.0)
    if normed:
        total = matrix.sum()
        if total > 0:
            matrix /= total
    return matrix, valid_pairs


def compute_glcm_properties(matrix: np.ndarray) -> dict[str, float]:
    """Compute GLCM properties with scikit-image-compatible definitions."""

    glcm = np.asarray(matrix, dtype=np.float64)
    if glcm.ndim != 2 or glcm.shape[0] != glcm.shape[1]:
        raise GLCMExtractionError("GLCM matrix must be square")
    if not np.isfinite(glcm).all():
        raise GLCMExtractionError("GLCM matrix contains NaN or infinite values")
    total = float(glcm.sum())
    if total <= 0:
        raise GLCMExtractionError("cannot compute properties for an empty GLCM")
    probability = glcm / total
    levels = np.arange(probability.shape[0], dtype=np.float64)
    i, j = np.meshgrid(levels, levels, indexing="ij")
    diff = i - j
    asm = float(np.sum(probability**2))
    energy = float(np.sqrt(asm))

    mean_i = float(np.sum(i * probability))
    mean_j = float(np.sum(j * probability))
    std_i = float(np.sqrt(np.sum(((i - mean_i) ** 2) * probability)))
    std_j = float(np.sqrt(np.sum(((j - mean_j) ** 2) * probability)))
    if std_i <= 1e-15 or std_j <= 1e-15:
        correlation = 1.0
    else:
        correlation = float(np.sum((i - mean_i) * (j - mean_j) * probability) / (std_i * std_j))

    return {
        "contrast": float(np.sum((diff**2) * probability)),
        "dissimilarity": float(np.sum(np.abs(diff) * probability)),
        "homogeneity": float(np.sum(probability / (1.0 + diff**2))),
        "energy": energy,
        "correlation": correlation,
        "ASM": asm,
    }


def extract_glcm_features_from_roi(roi: TumorROI, config: dict[str, Any]) -> dict[str, float]:
    """Extract the frozen 12-feature GLCM vector from a prepared tumor ROI."""

    parameters = config["parameters"]
    gray_levels = int(parameters["gray_levels"])
    quantized = quantize_glcm_image(roi.roi_image, gray_levels=gray_levels)
    property_values = {property_name: [] for property_name in GLCM_PROPERTIES}

    for distance in parameters["distances"]:
        for angle in parameters["angles_degrees"]:
            matrix, valid_pairs = build_masked_glcm(
                quantized,
                roi.roi_mask,
                gray_levels=gray_levels,
                distance=int(distance),
                angle_degrees=int(angle),
                symmetric=bool(parameters["symmetric"]),
                normed=bool(parameters["normed"]),
            )
            if valid_pairs == 0:
                continue
            values = compute_glcm_properties(matrix)
            for property_name in GLCM_PROPERTIES:
                property_values[property_name].append(values[property_name])

    if not any(property_values[property_name] for property_name in GLCM_PROPERTIES):
        raise GLCMExtractionError(
            f"sample {roi.sample_id} has no valid tumor pixel pairs for configured GLCM distances/angles"
        )

    features: dict[str, float] = {}
    for property_name in GLCM_PROPERTIES:
        values = np.asarray(property_values[property_name], dtype=np.float64)
        if values.size == 0:
            raise GLCMExtractionError(f"no valid values for GLCM property {property_name}")
        name = property_name.lower() if property_name != "ASM" else "asm"
        features[f"glcm_{name}_mean"] = float(np.mean(values))
        features[f"glcm_{name}_std"] = float(np.std(values, ddof=0))
    if not np.isfinite(list(features.values())).all():
        raise GLCMExtractionError("GLCM feature vector contains NaN or infinite values")
    return {column: features[column] for column in GLCM_FEATURE_COLUMNS}


def extract_glcm_features(sample: Phase1Sample, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract one sample row, including canonical metadata and GLCM features."""

    loaded_config = load_feature_config("glcm") if config is None else config
    roi = prepare_tumor_roi(sample)
    return {
        "sample_id": sample.sample_id,
        "patient_id": sample.patient_id,
        "label": sample.label,
        "split": sample.split,
        **extract_glcm_features_from_roi(roi, loaded_config),
    }


def extract_glcm_dataset(
    *,
    output: Path | str = "data/features/glcm/glcm_features.csv",
    metadata_output: Path | str = "data/features/glcm/glcm_metadata.json",
    split: str = "all",
    sample_ids: Iterable[str | int] | None = None,
    split_csv: Path | str = "data/splits/patient_split.csv",
    samples_dir: Path | str = "data/processed/samples",
    progress_every: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract GLCM features for a split or explicit sample IDs and write shared outputs."""

    start = time.perf_counter()
    config = load_feature_config("glcm")
    if sample_ids is None:
        samples = iter_phase1_samples(split, split_csv=split_csv, samples_dir=samples_dir)
    else:
        samples = (
            load_phase1_sample(sample_id, split_csv=split_csv, samples_dir=samples_dir)
            for sample_id in sample_ids
        )

    records = []
    for index, sample in enumerate(samples, start=1):
        records.append(extract_glcm_features(sample, config))
        if progress_every > 0 and index % progress_every == 0:
            print(f"processed {index} samples")

    frame = build_feature_dataframe(records)
    validated = write_feature_table(frame, output, canonical_split=split_csv)
    split_metadata = _load_split_version(split_csv)
    metadata = build_feature_metadata(
        feature_set_name=config["feature_set_name"],
        feature_version=config["feature_version"],
        algorithm=config["algorithm"],
        parameters=config["parameters"],
        input_representation=config["input_representation"],
        roi_policy=config["roi_policy"],
        source_split_version=split_metadata,
        sample_count=len(validated),
        feature_columns=GLCM_FEATURE_COLUMNS,
        code_commit=get_code_commit(),
        validation_summary={
            "row_count": int(len(validated)),
            "feature_count": len(GLCM_FEATURE_COLUMNS),
            "nan_count": int(validated[GLCM_FEATURE_COLUMNS].isna().sum().sum()),
            "inf_count": int(np.isinf(validated[GLCM_FEATURE_COLUMNS].to_numpy(dtype=float)).sum()),
            "split_counts": validated["split"].value_counts().sort_index().to_dict(),
        },
    )
    write_feature_metadata(metadata, metadata_output)
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "processed": int(len(validated)),
                "failed": 0,
                "elapsed_seconds": round(elapsed, 3),
                "output_csv": str(output),
                "metadata_path": str(metadata_output),
            },
            indent=2,
        )
    )
    return validated, metadata


def get_code_commit() -> str:
    """Return the current Git commit hash, falling back safely outside Git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _load_split_version(split_csv: Path | str) -> str:
    metadata_path = Path(split_csv).with_name("split_metadata.json")
    if not metadata_path.is_file():
        return "unknown"
    with metadata_path.open(encoding="utf-8") as handle:
        return str(json.load(handle).get("split_version", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/features/glcm/glcm_features.csv"))
    parser.add_argument(
        "--metadata-output", type=Path, default=Path("data/features/glcm/glcm_metadata.json")
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--split-csv", type=Path, default=Path("data/splits/patient_split.csv"))
    parser.add_argument("--samples-dir", type=Path, default=Path("data/processed/samples"))
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    extract_glcm_dataset(
        output=args.output,
        metadata_output=args.metadata_output,
        split=args.split,
        sample_ids=args.sample_ids,
        split_csv=args.split_csv,
        samples_dir=args.samples_dir,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
