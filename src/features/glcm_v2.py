"""Expanded mask-aware GLCM v2 feature extraction."""

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
    build_feature_dataframe,
    build_feature_metadata,
    validate_feature_table,
    write_feature_metadata,
    write_feature_table,
)
from src.features.glcm import (
    GLCM_AGGREGATIONS,
    GLCM_PROPERTIES,
    build_masked_glcm,
    compute_glcm_properties,
    quantize_glcm_image,
)
from src.preprocessing.roi import TumorROI, prepare_tumor_roi

PROPERTY_NAMES = tuple(prop.lower() if prop != "ASM" else "asm" for prop in GLCM_PROPERTIES)


class GLCMV2ExtractionError(ValueError):
    """Raised when GLCM v2 extraction cannot produce valid features."""


def glcm_v2_feature_columns(config: dict[str, Any] | None = None) -> list[str]:
    """Return the frozen 84-column GLCM v2 feature order."""

    loaded = load_feature_config("glcm_v2") if config is None else config
    params = loaded["parameters"]
    columns: list[str] = []
    for prop in PROPERTY_NAMES:
        for distance in params["distances"]:
            for angle in params["angles_degrees"]:
                columns.append(f"glcm_{prop}_d{int(distance)}_a{int(angle)}")
        for aggregation in GLCM_AGGREGATIONS:
            columns.append(f"glcm_{prop}_{aggregation}")
    return columns


def directional_glcm_property_values(
    roi: TumorROI,
    config: dict[str, Any],
) -> tuple[dict[tuple[str, int, int], float], dict[str, list[float]]]:
    """Compute per-property, per-distance, per-angle values from valid masked GLCMs."""

    params = config["parameters"]
    gray_levels = int(params["gray_levels"])
    quantized = quantize_glcm_image(roi.roi_image, gray_levels=gray_levels)
    directional: dict[tuple[str, int, int], float] = {}
    aggregate_values = {name: [] for name in PROPERTY_NAMES}

    for distance in params["distances"]:
        for angle in params["angles_degrees"]:
            matrix, valid_pairs = build_masked_glcm(
                quantized,
                roi.roi_mask,
                gray_levels=gray_levels,
                distance=int(distance),
                angle_degrees=int(angle),
                symmetric=bool(params["symmetric"]),
                normed=bool(params["normed"]),
            )
            if valid_pairs == 0:
                continue
            values = compute_glcm_properties(matrix)
            for original_name, output_name in zip(GLCM_PROPERTIES, PROPERTY_NAMES, strict=True):
                value = float(values[original_name])
                directional[(output_name, int(distance), int(angle))] = value
                aggregate_values[output_name].append(value)

    if not directional:
        raise GLCMV2ExtractionError(
            f"sample {roi.sample_id} has no valid tumor pixel pairs for configured GLCM v2 distances/angles"
        )
    return directional, aggregate_values


def extract_glcm_v2_features_from_roi(
    roi: TumorROI,
    config: dict[str, Any],
) -> dict[str, float]:
    """Extract 72 directional values plus 12 v1-compatible aggregate features."""

    params = config["parameters"]
    directional, aggregate_values = directional_glcm_property_values(roi, config)
    features: dict[str, float] = {}
    for prop in PROPERTY_NAMES:
        values_for_property = []
        for distance in params["distances"]:
            for angle in params["angles_degrees"]:
                key = (prop, int(distance), int(angle))
                if key in directional:
                    value = directional[key]
                else:
                    value = 0.0
                features[f"glcm_{prop}_d{int(distance)}_a{int(angle)}"] = float(value)
                if key in directional:
                    values_for_property.append(float(value))
        valid_values = np.asarray(values_for_property, dtype=np.float64)
        if valid_values.size == 0:
            raise GLCMV2ExtractionError(f"no valid values for GLCM v2 property {prop}")
        features[f"glcm_{prop}_mean"] = float(np.mean(valid_values))
        features[f"glcm_{prop}_std"] = float(np.std(valid_values, ddof=0))

    ordered = {column: features[column] for column in glcm_v2_feature_columns(config)}
    if not np.isfinite(list(ordered.values())).all():
        raise GLCMV2ExtractionError("GLCM v2 feature vector contains NaN or infinite values")
    return ordered


def extract_glcm_v2_features(
    sample: Phase1Sample,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract one sample row, including canonical metadata and GLCM v2 features."""

    loaded_config = load_feature_config("glcm_v2") if config is None else config
    roi = prepare_tumor_roi(sample)
    return {
        "sample_id": sample.sample_id,
        "patient_id": sample.patient_id,
        "label": sample.label,
        "split": sample.split,
        **extract_glcm_v2_features_from_roi(roi, loaded_config),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
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


def extract_glcm_v2_dataset(
    *,
    output: Path | str = "data/features/glcm_v2/glcm_v2_features.csv",
    metadata_output: Path | str = "data/features/glcm_v2/glcm_v2_metadata.json",
    split: str = "all",
    sample_ids: Iterable[str | int] | None = None,
    split_csv: Path | str = "data/splits/patient_split.csv",
    samples_dir: Path | str = "data/processed/samples",
    split_metadata_json: Path | str = "data/splits/split_metadata.json",
    progress_every: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract GLCM v2 features and write shared feature-table outputs."""

    start = time.perf_counter()
    config = load_feature_config("glcm_v2")
    if sample_ids is None:
        samples = iter_phase1_samples(split, split_csv=split_csv, samples_dir=samples_dir)
    else:
        samples = (
            load_phase1_sample(sample_id, split_csv=split_csv, samples_dir=samples_dir)
            for sample_id in sample_ids
        )

    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        records.append(extract_glcm_v2_features(sample, config))
        if progress_every > 0 and index % progress_every == 0:
            print(f"processed {index} samples")
    if not records:
        raise GLCMV2ExtractionError("no samples were extracted")

    frame = build_feature_dataframe(records)
    validated = write_feature_table(frame, output, canonical_split=split_csv)
    feature_columns = [column for column in validated.columns if column not in ("sample_id", "patient_id", "label", "split")]
    validation_summary = {
        "sample_count": int(len(validated)),
        "feature_count": int(len(feature_columns)),
        "samples_by_split": {
            str(key): int(value) for key, value in validated["split"].value_counts().items()
        },
        "samples_by_label": {
            str(key): int(value) for key, value in validated["label"].value_counts().sort_index().items()
        },
        "finite_features": bool(np.isfinite(validated[feature_columns].to_numpy(dtype=float)).all()),
    }
    metadata = build_feature_metadata(
        feature_set_name=config["feature_set_name"],
        feature_version=config["feature_version"],
        algorithm=config["algorithm"],
        parameters=config["parameters"],
        input_representation=config["input_representation"],
        roi_policy=config["roi_policy"],
        source_split_version=_split_version(split_metadata_json),
        sample_count=len(validated),
        feature_columns=feature_columns,
        code_commit=_git_commit(),
        validation_summary=validation_summary,
    )
    write_feature_metadata(metadata, metadata_output)
    elapsed = time.perf_counter() - start
    print(
        f"completed GLCM v2 extraction: processed={len(validated)} failed=0 "
        f"elapsed_seconds={elapsed:.2f} output={output} metadata={metadata_output}"
    )
    return validated, metadata


def validate_glcm_v2_feature_file(
    feature_path: Path | str = "data/features/glcm_v2/glcm_v2_features.csv",
    *,
    canonical_split: Path | str = "data/splits/patient_split.csv",
) -> pd.DataFrame:
    """Read and validate a GLCM v2 feature table."""

    frame = pd.read_csv(feature_path, dtype={"patient_id": str})
    validated = validate_feature_table(frame, canonical_split=canonical_split)
    feature_columns = [column for column in validated.columns if column not in ("sample_id", "patient_id", "label", "split")]
    if len(feature_columns) != 84:
        raise GLCMV2ExtractionError(f"expected 84 GLCM v2 features, got {len(feature_columns)}")
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/features/glcm_v2/glcm_v2_features.csv"))
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/features/glcm_v2/glcm_v2_metadata.json"),
    )
    parser.add_argument("--split", default="all")
    parser.add_argument("--split-csv", type=Path, default=Path("data/splits/patient_split.csv"))
    parser.add_argument("--samples-dir", type=Path, default=Path("data/processed/samples"))
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    extract_glcm_v2_dataset(
        output=args.output,
        metadata_output=args.metadata_output,
        split=args.split,
        split_csv=args.split_csv,
        samples_dir=args.samples_dir,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
