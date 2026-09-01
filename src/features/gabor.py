import argparse
import json
from pathlib import Path

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


def _feature_name(prefix, frequency, orientation_deg, stat):
    freq_code = f"f{round(frequency * 100):03d}"
    orientation_code = f"o{int(round(orientation_deg)):03d}"
    return f"{prefix}_{freq_code}_{orientation_code}_{stat}"


def gabor_kernel_bank(parameters):
    bank = []
    for frequency in parameters["frequencies"]:
        for orientation_deg in parameters["orientations_degrees"]:
            theta_rad = float(np.deg2rad(orientation_deg))
            bank.append((float(frequency), float(orientation_deg), theta_rad))
    return bank


def extract_gabor_features(image, config):
    image_array = np.asarray(image, dtype=np.float64)
    parameters = config["parameters"]
    prefix = config["feature_prefix"]
    stat_order = parameters["response_statistics"]
    sigma_x = parameters.get("sigma_x")
    sigma_y = parameters.get("sigma_y")
    n_stds = parameters["n_stds"]
    offset = parameters["offset"]

    features = {}
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


def _split_version(split_metadata_json):
    path = Path(split_metadata_json)
    if not path.is_file():
        return "unknown"
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return str(metadata.get("split_version", "unknown"))


def build_gabor_feature_table(
    split="all",
    *,
    split_csv="data/splits/patient_split.csv",
    samples_dir="data/processed/samples",
    sample_ids=None,
    config=None,
):
    config = config or load_feature_config("gabor")
    requested_ids = None if sample_ids is None else {str(int(value)) for value in sample_ids}
    records = []
    padding_fraction = None
    standard_size = None

    for sample in iter_phase1_samples(split, split_csv=split_csv, samples_dir=samples_dir):
        if requested_ids is not None and sample.sample_id not in requested_ids:
            continue
        roi = prepare_tumor_roi(sample)
        padding_fraction = roi.padding_fraction
        standard_size = roi.standard_size
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
        "padding_fraction": padding_fraction,
        "standard_size": list(standard_size) if standard_size is not None else None,
    }
    return frame, roi_info


def extract_and_write_gabor_features(
    split="all",
    *,
    split_csv="data/splits/patient_split.csv",
    samples_dir="data/processed/samples",
    sample_ids=None,
    output_csv="data/features/gabor/gabor_features.csv",
    metadata_json="data/features/gabor/gabor_metadata.json",
    split_metadata_json="data/splits/split_metadata.json",
):
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
    feature_columns = [column for column in validated.columns if column not in METADATA_COLUMNS]
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
        algorithm=config["algorithm"],
        parameters={
            **config["parameters"],
            "status": f"compliant with frozen configs/features/gabor.yaml (feature_version {config['feature_version']})",
            "run_scope": run_scope,
        },
        input_representation=config["input_representation"],
        roi_policy=(
            f"{config['roi_policy']} (actual padding_fraction={roi_info['padding_fraction']}, "
            f"standard_size={standard_size})"
        ),
        source_split_version=_split_version(split_metadata_json),
        sample_count=int(len(validated)),
        feature_columns=feature_columns,
        code_commit="unknown",
        validation_summary=validation_summary,
    )
    write_feature_metadata(metadata, metadata_json)
    return {"table": validated, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser()
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
