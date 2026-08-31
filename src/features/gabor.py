"""Gabor filter-bank feature extraction for Phase 1 tumor ROIs.

Fully driven by the frozen ``configs/features/gabor.yaml`` specification via
``src.features.config``. No orientation, frequency, or statistic here may
drift from that file; if the frozen config changes, this module picks up
the change automatically instead of needing an edit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from skimage.filters import gabor

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
RESPONSE_STATISTICS = (
    "real_mean",
    "real_std",
    "magnitude_mean",
    "magnitude_std",
    "magnitude_energy",
)


def _feature_name(prefix: str, frequency: float, orientation_deg: float, stat: str) -> str:
    freq_code = f"f{round(frequency * 100):03d}"
    orientation_code = f"o{int(round(orientation_deg)):03d}"
    return f"{prefix}_{freq_code}_{orientation_code}_{stat}"


def gabor_kernel_bank(parameters: dict[str, Any]) -> list[tuple[float, float, float]]:
    """Deterministic (frequency, orientation_deg, theta_rad) bank from frozen parameters."""

    bank = []
    for frequency in parameters["frequencies"]:
        for orientation_deg in parameters["orientations_degrees"]:
            theta_rad = float(np.deg2rad(orientation_deg))
            bank.append((float(frequency), float(orientation_deg), theta_rad))
    return bank


def extract_gabor_features(image: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    """Compute the frozen-spec Gabor response statistics for one ROI image."""

    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image_array.shape}")
    if not np.isfinite(image_array).all():
        raise ValueError("image contains NaN or infinite values")

    parameters = config["parameters"]
    prefix = config["feature_prefix"]
    stat_order = parameters["response_statistics"]
    sigma_x = parameters.get("sigma_x")
    sigma_y = parameters.get("sigma_y")
    n_stds = parameters["n_stds"]
    offset = parameters["offset"]

    features: dict[str, float] = {}
    for frequency, orientation_deg, theta_rad in gabor_kernel_bank(parameters):
        real, imag = gabor(
            image_array,
            frequency=frequency,
            theta=theta_rad,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            n_stds=n_stds,
            offset=offset,
        )
        magnitude = np.hypot(real, imag)
        available_stats = {
            "real_mean": float(real.mean()),
            "real_std": float(real.std()),
            "magnitude_mean": float(magnitude.mean()),
            "magnitude_std": float(magnitude.std()),
            "magnitude_energy": float(np.mean(np.square(magnitude))),
        }
        for stat in stat_order:
            features[_feature_name(prefix, frequency, orientation_deg, stat)] = available_stats[stat]

    expected_count = config["expected_feature_count"]
    if len(features) != expected_count:
        raise ValueError(
            f"produced {len(features)} Gabor features, expected {expected_count} per frozen config"
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


def build_gabor_feature_table(
    split: str = "all",
    *,
    split_csv: Path | str = "data/splits/patient_split.csv",
    samples_dir: Path | str = "data/processed/samples",
    sample_ids: Iterable[str | int] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract Gabor features for the requested samples via the shared Phase 1 loaders."""

    config = config if config is not None else load_feature_config("gabor")
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
        features = extract_gabor_features(roi.roi_image_masked_resized, config)
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


def extract_and_write_gabor_features(
    split: str = "all",
    *,
    split_csv: Path | str = "data/splits/patient_split.csv",
    samples_dir: Path | str = "data/processed/samples",
    sample_ids: Iterable[str | int] | None = None,
    output_csv: Path | str = "data/features/gabor/gabor_features.csv",
    metadata_json: Path | str = "data/features/gabor/gabor_metadata.json",
    split_metadata_json: Path | str = "data/splits/split_metadata.json",
) -> dict[str, Any]:
    """Extract Gabor features per the frozen config and write the validated outputs."""

    config = load_feature_config("gabor")
    sample_ids_list = None if sample_ids is None else list(sample_ids)
    run_scope = (
        "full_dataset"
        if sample_ids_list is None
        else f"partial_subset_{len(sample_ids_list)}_samples"
    )
    frame, roi_info = build_gabor_feature_table(
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
        algorithm=f"{config['algorithm']} (skimage.filters.gabor, per frozen configs/features/gabor.yaml)",
        parameters={
            **config["parameters"],
            "status": f"compliant with frozen configs/features/gabor.yaml (feature_version {config['feature_version']})",
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
        "--output-csv", type=Path, default=Path("data/features/gabor/gabor_features.csv")
    )
    parser.add_argument(
        "--metadata-json", type=Path, default=Path("data/features/gabor/gabor_metadata.json")
    )
    args = parser.parse_args()
    result = extract_and_write_gabor_features(
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
