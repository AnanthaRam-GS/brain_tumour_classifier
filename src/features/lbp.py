"""Mask-aware LBP feature extraction for the Phase 1 baseline."""

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
from skimage.feature import local_binary_pattern

from src.data.feature_dataset import Phase1Sample, iter_phase1_samples, load_phase1_sample
from src.features.config import load_feature_config
from src.features.feature_table import (
    build_feature_metadata,
    build_feature_dataframe,
    write_feature_metadata,
    write_feature_table,
)
from src.preprocessing.roi import TumorROI, prepare_tumor_roi


METADATA_COLUMNS = ("sample_id", "patient_id", "label", "split")


class LBPExtractionError(ValueError):
    """Raised when LBP extraction cannot produce valid features."""


def extract_lbp_histogram(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    points: int,
    radius: float,
    method: str,
    normalized: bool = True,
) -> np.ndarray:
    """Compute a mask-aware LBP histogram."""

    image_array = np.asarray(image, dtype=np.float64)
    mask_array = np.asarray(mask).astype(bool)

    if image_array.ndim != 2:
        raise LBPExtractionError(
            f"image must be 2D, got shape {image_array.shape}"
        )

    if mask_array.ndim != 2:
        raise LBPExtractionError(
            f"mask must be 2D, got shape {mask_array.shape}"
        )

    if image_array.shape != mask_array.shape:
        raise LBPExtractionError(
            f"image and mask shapes differ: "
            f"{image_array.shape} != {mask_array.shape}"
        )

    if not np.isfinite(image_array).all():
        raise LBPExtractionError(
            "image contains NaN or infinite values"
        )

    if points <= 0:
        raise LBPExtractionError(
            "points must be positive"
        )

    if radius <= 0:
        raise LBPExtractionError(
            "radius must be positive"
        )

    if not mask_array.any():
        raise LBPExtractionError(
            "tumor mask contains no pixels"
        )

    lbp = local_binary_pattern(
        image_array,
        P=points,
        R=radius,
        method=method,
    )

    # Only pixels belonging to the tumor are included in the histogram.
    tumor_lbp_values = lbp[mask_array]

    if tumor_lbp_values.size == 0:
        raise LBPExtractionError(
            "no LBP values found inside tumor mask"
        )

    # For the "uniform" method, scikit-image produces P + 2 bins.
    expected_bin_count = points + 2

    histogram = np.bincount(
        tumor_lbp_values.astype(np.int64),
        minlength=expected_bin_count,
    )

    # Guard against an unexpected value produced by a different method/config.
    if histogram.size != expected_bin_count:
        raise LBPExtractionError(
            f"produced {histogram.size} histogram bins, "
            f"expected {expected_bin_count}"
        )

    histogram = histogram.astype(np.float64)

    if normalized:
        total = histogram.sum()

        if total <= 0:
            raise LBPExtractionError(
                "cannot normalize an empty LBP histogram"
            )

        histogram /= total

    if not np.isfinite(histogram).all():
        raise LBPExtractionError(
            "LBP histogram contains NaN or infinite values"
        )

    return histogram


def extract_lbp_features_from_roi(
    roi: TumorROI,
    config: dict[str, Any],
) -> dict[str, float]:
    """Extract the frozen 10-feature LBP vector from a prepared tumor ROI."""

    parameters = config["parameters"]

    points = int(parameters["P"])
    radius = float(parameters["R"])
    method = str(parameters["method"])

    histogram_config = parameters["histogram"]
    use_pixels = str(histogram_config["use_pixels"])
    normalized = bool(histogram_config["normalized"])

    if use_pixels != "tumor_mask_only":
        raise LBPExtractionError(
            f"unsupported LBP histogram pixel selection: {use_pixels}"
        )

    histogram = extract_lbp_histogram(
        roi.roi_image,
        roi.roi_mask,
        points=points,
        radius=radius,
        method=method,
        normalized=normalized,
    )

    expected_count = int(config["expected_feature_count"])

    if histogram.size != expected_count:
        raise LBPExtractionError(
            f"produced {histogram.size} LBP features, "
            f"expected {expected_count} per frozen config"
        )

    features = {
        f"{config['feature_prefix']}_{index}": float(value)
        for index, value in enumerate(histogram)
    }

    if not np.isfinite(list(features.values())).all():
        raise LBPExtractionError(
            "LBP feature vector contains NaN or infinite values"
        )

    if normalized and not np.isclose(
        sum(features.values()),
        1.0,
        atol=1e-12,
    ):
        raise LBPExtractionError(
            "normalized LBP histogram does not sum to 1"
        )

    return features


def extract_lbp_features(
    sample: Phase1Sample,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract one sample row, including canonical metadata and LBP features."""

    loaded_config = (
        load_feature_config("lbp")
        if config is None
        else config
    )

    roi = prepare_tumor_roi(sample)

    return {
        "sample_id": sample.sample_id,
        "patient_id": sample.patient_id,
        "label": sample.label,
        "split": sample.split,
        **extract_lbp_features_from_roi(roi, loaded_config),
    }


def extract_lbp_dataset(
    *,
    output: Path | str = "data/features/lbp/lbp_features.csv",
    metadata_output: Path | str = "data/features/lbp/lbp_metadata.json",
    split: str = "all",
    sample_ids: Iterable[str | int] | None = None,
    split_csv: Path | str = "data/splits/patient_split.csv",
    samples_dir: Path | str = "data/processed/samples",
    progress_every: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract LBP features for a split or explicit sample IDs."""

    start = time.perf_counter()

    config = load_feature_config("lbp")

    if sample_ids is None:
        samples = iter_phase1_samples(
            split,
            split_csv=split_csv,
            samples_dir=samples_dir,
        )
    else:
        samples = (
            load_phase1_sample(
                sample_id,
                split_csv=split_csv,
                samples_dir=samples_dir,
            )
            for sample_id in sample_ids
        )

    records = []

    for index, sample in enumerate(samples, start=1):
        records.append(
            extract_lbp_features(
                sample,
                config,
            )
        )

        if progress_every > 0 and index % progress_every == 0:
            print(f"processed {index} samples")

    if not records:
        raise LBPExtractionError(
            "no samples matched the requested split/sample_ids filters"
        )

    frame = build_feature_dataframe(records)

    validated = write_feature_table(
        frame,
        output,
        canonical_split=split_csv,
    )

    feature_columns = [
        column
        for column in validated.columns
        if column not in METADATA_COLUMNS
    ]

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
        feature_columns=feature_columns,
        code_commit=get_code_commit(),
        validation_summary={
            "row_count": int(len(validated)),
            "feature_count": len(feature_columns),
            "expected_feature_count": int(
                config["expected_feature_count"]
            ),
            "nan_count": int(
                validated[feature_columns].isna().sum().sum()
            ),
            "inf_count": int(
                np.isinf(
                    validated[feature_columns].to_numpy(dtype=float)
                ).sum()
            ),
            "split_counts": (
                validated["split"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "label_counts": (
                validated["label"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
        },
    )

    write_feature_metadata(
        metadata,
        metadata_output,
    )

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


def _load_split_version(
    split_csv: Path | str,
) -> str:
    """Load the frozen split version from split_metadata.json."""

    metadata_path = Path(split_csv).with_name(
        "split_metadata.json"
    )

    if not metadata_path.is_file():
        return "unknown"

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    return str(
        metadata.get(
            "split_version",
            "unknown",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/features/lbp/lbp_features.csv"
        ),
    )

    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path(
            "data/features/lbp/lbp_metadata.json"
        ),
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
    )

    parser.add_argument(
        "--sample-ids",
        nargs="+",
    )

    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path(
            "data/splits/patient_split.csv"
        ),
    )

    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path(
            "data/processed/samples"
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    extract_lbp_dataset(
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