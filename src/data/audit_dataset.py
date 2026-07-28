"""Audit the raw Figshare brain tumour MATLAB dataset without converting it."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.mat_loader import BrainTumourSample, MatSampleError, find_mat_sample_files, load_mat_sample
from src.data.visualize_samples import CLASS_NAMES, create_sample_figures

MANIFEST_COLUMNS = [
    "sample_id", "source_path", "patient_id", "label", "class_name",
    "image_height", "image_width", "image_dtype_original", "image_min",
    "image_max", "image_mean", "image_std", "mask_height", "mask_width",
    "tumour_pixels", "tumour_fraction", "has_empty_mask", "validation_status",
    "validation_message",
]


def _relative_source(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(dataset_root.resolve().parent))
    except ValueError:
        return str(path)


def _valid_row(sample: BrainTumourSample, dataset_root: Path) -> dict[str, Any]:
    tumour_pixels = int(sample.tumour_mask.sum())
    return {
        "sample_id": sample.sample_id,
        "source_path": _relative_source(Path(sample.source_path), dataset_root),
        "patient_id": sample.patient_id,
        "label": sample.label,
        "class_name": CLASS_NAMES[sample.label],
        "image_height": int(sample.image.shape[0]),
        "image_width": int(sample.image.shape[1]),
        "image_dtype_original": sample.image_dtype_original,
        "image_min": float(sample.image.min()),
        "image_max": float(sample.image.max()),
        "image_mean": float(sample.image.mean()),
        "image_std": float(sample.image.std()),
        "mask_height": int(sample.tumour_mask.shape[0]),
        "mask_width": int(sample.tumour_mask.shape[1]),
        "tumour_pixels": tumour_pixels,
        "tumour_fraction": float(tumour_pixels / sample.tumour_mask.size),
        "has_empty_mask": tumour_pixels == 0,
        "validation_status": "valid",
        "validation_message": "",
    }


def _invalid_row(path: Path, dataset_root: Path, message: str) -> dict[str, Any]:
    row = {column: None for column in MANIFEST_COLUMNS}
    row.update(
        sample_id=str(path.stem),
        source_path=_relative_source(path, dataset_root),
        has_empty_mask=None,
        validation_status="invalid",
        validation_message=message,
    )
    return row


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def audit_dataset(
    dataset_root: Path | str,
    reports_root: Path | str = "reports",
    seed: int = 2024,
) -> dict[str, Any]:
    """Load every sample independently and write a complete audit report."""

    root = Path(dataset_root)
    report_dir = Path(reports_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    sample_files = find_mat_sample_files(root)

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    class_seen: defaultdict[int, int] = defaultdict(int)
    figure_samples: defaultdict[int, list[BrainTumourSample]] = defaultdict(list)
    pixel_count = 0
    pixel_sum = 0.0
    pixel_square_sum = 0.0
    global_min = float("inf")
    global_max = float("-inf")
    for path in sample_files:
        try:
            sample = load_mat_sample(path)
            rows.append(_valid_row(sample, root))
            pixel_count += sample.image.size
            pixel_sum += float(np.sum(sample.image, dtype=np.float64))
            pixel_square_sum += float(
                np.sum(np.square(sample.image, dtype=np.float64), dtype=np.float64)
            )
            global_min = min(global_min, float(sample.image.min()))
            global_max = max(global_max, float(sample.image.max()))

            class_seen[sample.label] += 1
            reservoir = figure_samples[sample.label]
            if len(reservoir) < 4:
                reservoir.append(sample)
            else:
                replacement = int(rng.integers(0, class_seen[sample.label]))
                if replacement < 4:
                    reservoir[replacement] = sample
        except Exception as exc:
            rows.append(_invalid_row(path, root, str(exc)))

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest.to_csv(report_dir / "sample_manifest_raw.csv", index=False)
    valid = manifest[manifest["validation_status"] == "valid"].copy()

    class_distribution = (
        valid.groupby(["label", "class_name"], dropna=False)
        .size()
        .rename("sample_count")
        .reset_index()
        .sort_values("label")
    )
    if not valid.empty:
        patient_counts = valid.groupby("label")["patient_id"].nunique()
        class_distribution["unique_patients"] = (
            class_distribution["label"].map(patient_counts).astype(int)
        )
    else:
        class_distribution["unique_patients"] = pd.Series(dtype=int)
    class_distribution.to_csv(report_dir / "class_distribution.csv", index=False)

    patient_rows = []
    for patient_id, group in valid.groupby("patient_id"):
        counts = Counter(int(label) for label in group["label"])
        labels = sorted(counts)
        patient_rows.append(
            {
                "patient_id": patient_id,
                "sample_count": int(len(group)),
                "meningioma_samples": counts[1],
                "glioma_samples": counts[2],
                "pituitary_samples": counts[3],
                "labels": "|".join(map(str, labels)),
                "has_multiple_labels": len(labels) > 1,
            }
        )
    patient_distribution = pd.DataFrame(
        patient_rows,
        columns=[
            "patient_id", "sample_count", "meningioma_samples", "glioma_samples",
            "pituitary_samples", "labels", "has_multiple_labels",
        ],
    ).sort_values("patient_id")
    patient_distribution.to_csv(report_dir / "patient_distribution.csv", index=False)

    shape_distribution = (
        valid.groupby(["image_height", "image_width"])
        .size()
        .rename("sample_count")
        .reset_index()
        .sort_values(["image_height", "image_width"])
    )
    shape_distribution.to_csv(report_dir / "image_shape_distribution.csv", index=False)

    id_counts = Counter(path.stem for path in sample_files)
    duplicate_ids = {
        sample_id: count for sample_id, count in sorted(id_counts.items(), key=lambda item: int(item[0]))
        if count > 1
    }
    multi_label_patients = patient_distribution[
        patient_distribution["has_multiple_labels"] == True  # noqa: E712
    ]
    invalid = manifest[manifest["validation_status"] == "invalid"]
    empty_masks = valid[valid["has_empty_mask"] == True]  # noqa: E712

    if pixel_count:
        pixel_mean = pixel_sum / pixel_count
        pixel_variance = max(0.0, pixel_square_sum / pixel_count - pixel_mean**2)
        intensity = {
            "global_min": global_min,
            "global_max": global_max,
            "global_mean": pixel_mean,
            "global_std": float(np.sqrt(pixel_variance)),
            "total_pixels": pixel_count,
            "sample_mean_min": float(valid["image_mean"].min()),
            "sample_mean_max": float(valid["image_mean"].max()),
        }
    else:
        intensity = {}

    retained_samples = [
        sample for label in sorted(figure_samples) for sample in figure_samples[label]
    ]
    figures = create_sample_figures(
        retained_samples, report_dir / "sample_overlays", per_class=4, seed=seed
    )
    audit = {
        "dataset_root": str(root),
        "random_seed": seed,
        "total_mat_sample_files": len(sample_files),
        "valid_files": len(valid),
        "invalid_files": len(invalid),
        "unique_patients": int(valid["patient_id"].nunique()),
        "samples_per_patient": {
            str(row.patient_id): int(row.sample_count)
            for row in patient_distribution.itertuples(index=False)
        },
        "class_distribution": _records(class_distribution),
        "class_distribution_by_patient": _records(
            patient_distribution[
                [
                    "patient_id", "meningioma_samples", "glioma_samples",
                    "pituitary_samples", "labels",
                ]
            ]
        ),
        "image_shape_distribution": _records(shape_distribution),
        "image_intensity_distribution": intensity,
        "duplicate_sample_ids": duplicate_ids,
        "patients_with_multiple_class_labels": _records(multi_label_patients),
        "empty_masks": _records(
            empty_masks[["sample_id", "source_path", "patient_id", "label"]]
        ),
        "malformed_files": _records(
            invalid[["sample_id", "source_path", "validation_message"]]
        ),
        "sample_figures": [str(path) for path in figures],
    }
    with (report_dir / "dataset_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    result = audit_dataset(args.dataset_root, args.reports_root, args.seed)
    print(json.dumps({key: result[key] for key in (
        "total_mat_sample_files", "valid_files", "invalid_files", "unique_patients"
    )}, indent=2))


if __name__ == "__main__":
    main()
