"""python -m btdl.cli.audit

Full pass over the Phase 1 NPZ sample cache, independently verified against
the raw Figshare .mat files. Writes:
    phase2/artifacts/contract/manifest.csv
    phase2/artifacts/contract/dataset_audit.json

Exits non-zero if any HARD check fails. Deterministic: two runs at the same
commit produce byte-identical outputs (no wall-clock timestamps).
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from btdl import config
from btdl.data.manifest import build_manifest, content_sha256, discover_raw_files
from btdl.data.raw_reader import RawSampleError, read_raw_sample

ARTIFACTS_DIR = "phase2/artifacts/contract"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.repo_root(), text=True
        ).strip()
    except Exception:
        return "unknown"


def _recompute_normalization(image_raw: np.ndarray, lower_percentile=1.0, upper_percentile=99.0):
    """Independent reimplementation of Phase 1's robust_foreground_percentile_normalize spec.

    foreground = pixels with raw intensity > 0; low/high are the given
    percentiles of the foreground; result = clip(image, low, high) mapped
    to [0, 1] via (x - low) / (high - low), with background forced to 0.
    Written from the documented spec, not from Phase 1's source.
    """

    image = np.asarray(image_raw, dtype=np.float64)
    foreground_mask = image > 0
    foreground = image[foreground_mask]
    if foreground.size == 0:
        return np.zeros(image.shape, dtype=np.float32)

    low, high = np.percentile(foreground, [lower_percentile, upper_percentile])
    scale = high - low
    tolerance = np.finfo(np.float32).eps * max(1.0, abs(low), abs(high))
    if not np.isfinite(scale) or scale <= tolerance:
        return np.zeros(image.shape, dtype=np.float32)

    clipped = np.clip(image, low, high)
    normalized = ((clipped - low) / scale).astype(np.float32)
    normalized[~foreground_mask] = 0.0
    np.clip(normalized, 0.0, 1.0, out=normalized)
    return normalized


def _percentile_stats(values):
    if len(values) == 0:
        return {"min": None, "p5": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p5": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def run_audit():
    contract = config.load_contract("data")
    data_root = config.data_root()
    raw_dir = config.resolve_data_path(contract["raw_dir"])
    samples_dir = config.resolve_data_path(contract["samples_dir"])
    split_csv = config.resolve_data_path(contract["split_csv"])
    expected = contract["expected"]

    manifest = build_manifest(
        samples_dir=samples_dir, raw_dir=raw_dir, split_csv=split_csv, data_root=data_root
    )
    raw_files = discover_raw_files(raw_dir)
    npz_files = sorted(samples_dir.glob("*.npz"), key=lambda p: int(p.stem))
    npz_ids = {p.stem for p in npz_files}
    raw_ids = set(raw_files.keys())

    hard = {}
    reported = {}

    # --- HARD: raw/npz coverage -------------------------------------------------
    missing_npz_for_raw = sorted(raw_ids - npz_ids, key=int)
    missing_raw_for_npz = sorted(npz_ids - raw_ids, key=int)
    hard["raw_npz_coverage"] = {
        "pass": not missing_npz_for_raw and not missing_raw_for_npz,
        "raw_count": len(raw_ids),
        "npz_count": len(npz_ids),
        "missing_npz_for_raw": missing_npz_for_raw,
        "missing_raw_for_npz": missing_raw_for_npz,
    }

    # --- HARD: every raw file readable ------------------------------------------
    unreadable_raw = []
    for sample_id in sorted(raw_ids, key=int):
        try:
            read_raw_sample(raw_files[sample_id])
        except RawSampleError as exc:
            unreadable_raw.append({"sample_id": sample_id, "error": str(exc)})
    hard["raw_files_readable"] = {
        "pass": len(unreadable_raw) == 0,
        "n_checked": len(raw_ids),
        "n_unreadable": len(unreadable_raw),
        "unreadable": unreadable_raw,
    }

    # --- Per-sample detailed pass -------------------------------------------------
    raw_vs_npz_mismatches = []
    background_violations = []
    normalization_max_abs_diff = 0.0
    normalization_n_checked = 0
    normalization_failures = []
    mask_invalid = []
    non_finite_normalized = []
    out_of_range_normalized = []

    raw_shapes = []
    raw_dtypes = set()
    mask_areas_overall = []
    mask_areas_by_class = {1: [], 2: [], 3: []}
    bbox_longest_overall = []
    bbox_longest_by_class = {1: [], 2: [], 3: []}
    image_content_hashes = {}  # sha256(image_raw bytes) -> [sample_id, ...]

    bbox_longest_map = dict(zip(manifest["sample_id"], manifest["bbox_longest"].astype(int)))
    split_map = dict(zip(manifest["sample_id"], manifest["split"]))

    common_ids = sorted(npz_ids & raw_ids, key=int)
    for sample_id in common_ids:
        npz_path = samples_dir / f"{sample_id}.npz"
        with np.load(npz_path) as data:
            npz_label = int(data["label"])
            npz_patient_id = str(data["patient_id"])
            npz_image_raw = np.asarray(data["image_raw"], dtype=np.float32)
            npz_image_normalized = np.asarray(data["image_normalized"], dtype=np.float32)
            npz_mask = np.asarray(data["tumor_mask"])

        try:
            raw_sample = read_raw_sample(raw_files[sample_id])
        except RawSampleError:
            continue  # already recorded under raw_files_readable

        raw_shapes.append(tuple(raw_sample.image.shape))
        raw_dtypes.add(str(raw_sample.image.dtype))

        raw_image_as_float32 = raw_sample.image.astype(np.float32)
        if not (
            npz_label == raw_sample.label
            and npz_patient_id == raw_sample.patient_id
            and np.array_equal(npz_mask.astype(bool), raw_sample.tumor_mask)
            and np.array_equal(npz_image_raw, raw_image_as_float32)
        ):
            raw_vs_npz_mismatches.append(sample_id)

        # Mask validity
        mask_values = set(np.unique(npz_mask).tolist())
        mask_area = int(np.asarray(npz_mask).astype(bool).sum())
        if not mask_values.issubset({0, 1}) or mask_area == 0 or npz_mask.shape != npz_image_raw.shape:
            mask_invalid.append(sample_id)

        # Normalized-image sanity
        if not np.isfinite(npz_image_normalized).all():
            non_finite_normalized.append(sample_id)
        if npz_image_normalized.min() < 0.0 or npz_image_normalized.max() > 1.0:
            out_of_range_normalized.append(sample_id)

        background_mask = raw_image_as_float32 <= 0
        if background_mask.any():
            if not np.array_equal(npz_image_normalized[background_mask], np.zeros(background_mask.sum(), dtype=np.float32)):
                background_violations.append(sample_id)

        # Independent normalization recompute
        recomputed = _recompute_normalization(raw_image_as_float32)
        abs_diff = float(np.max(np.abs(recomputed - npz_image_normalized)))
        normalization_n_checked += 1
        normalization_max_abs_diff = max(normalization_max_abs_diff, abs_diff)
        if abs_diff > 1e-5:
            normalization_failures.append({"sample_id": sample_id, "max_abs_diff": abs_diff})

        # Stats
        mask_areas_overall.append(mask_area)
        if npz_label in mask_areas_by_class:
            mask_areas_by_class[npz_label].append(mask_area)
        bbox_longest_overall.append(bbox_longest_map[sample_id])
        if npz_label in bbox_longest_by_class:
            bbox_longest_by_class[npz_label].append(bbox_longest_map[sample_id])

        image_hash = hashlib.sha256(np.ascontiguousarray(npz_image_raw).tobytes()).hexdigest()
        image_content_hashes.setdefault(image_hash, []).append(sample_id)

    hard["raw_vs_npz_equal"] = {
        "pass": len(raw_vs_npz_mismatches) == 0,
        "n_checked": len(common_ids),
        "n_mismatches": len(raw_vs_npz_mismatches),
        "mismatches": raw_vs_npz_mismatches,
    }
    hard["masks_valid"] = {
        "pass": len(mask_invalid) == 0,
        "n_checked": len(common_ids),
        "n_invalid": len(mask_invalid),
        "invalid": mask_invalid,
    }
    hard["normalized_finite_and_in_range"] = {
        "pass": len(non_finite_normalized) == 0 and len(out_of_range_normalized) == 0,
        "non_finite": non_finite_normalized,
        "out_of_range": out_of_range_normalized,
    }
    hard["background_exactly_zero"] = {
        "pass": len(background_violations) == 0,
        "n_checked": len(common_ids),
        "n_violations": len(background_violations),
        "violations": background_violations,
    }
    hard["normalization_recompute_matches"] = {
        "pass": len(normalization_failures) == 0,
        "n_checked": normalization_n_checked,
        "max_abs_diff": normalization_max_abs_diff,
        "tolerance": 1e-5,
        "failures": normalization_failures,
    }

    # --- HARD: labels / class counts --------------------------------------------
    label_counts = manifest["label"].value_counts().to_dict()
    label_counts = {int(k): int(v) for k, v in label_counts.items()}
    expected_class_counts = {int(k): int(v) for k, v in expected["class_counts"].items()}
    labels_valid = set(manifest["label"].unique()).issubset({1, 2, 3})
    counts_match = label_counts == expected_class_counts
    hard["labels_and_class_counts"] = {
        "pass": labels_valid and counts_match and len(manifest) == expected["n_samples"],
        "n_samples": int(len(manifest)),
        "expected_n_samples": expected["n_samples"],
        "observed_class_counts": label_counts,
        "expected_class_counts": expected_class_counts,
    }

    # --- HARD: patients ----------------------------------------------------------
    patient_label_counts = manifest.groupby("patient_id")["label"].nunique()
    multi_label_patients = sorted(patient_label_counts[patient_label_counts > 1].index.tolist())
    n_unique_patients = int(manifest["patient_id"].nunique())
    hard["patients_single_label"] = {
        "pass": len(multi_label_patients) == 0 and n_unique_patients == expected["n_patients"],
        "n_unique_patients": n_unique_patients,
        "expected_n_patients": expected["n_patients"],
        "multi_label_patients": multi_label_patients,
    }

    # --- HARD: duplicate image content -------------------------------------------
    duplicate_groups = []
    for image_hash, sample_ids in image_content_hashes.items():
        if len(sample_ids) > 1:
            group = [
                {"sample_id": sample_id, "split": split_map[sample_id]} for sample_id in sample_ids
            ]
            duplicate_groups.append(group)
    hard["no_duplicate_images"] = {
        "pass": len(duplicate_groups) == 0,
        "n_duplicate_groups": len(duplicate_groups),
        "groups": duplicate_groups,
    }

    # --- REPORTED statistics ------------------------------------------------------
    shape_counts = {}
    for shape in raw_shapes:
        key = f"{shape[0]}x{shape[1]}"
        shape_counts[key] = shape_counts.get(key, 0) + 1

    slices_per_patient = manifest.groupby("patient_id").size()
    reported["shape_distribution"] = dict(sorted(shape_counts.items()))
    reported["raw_dtype"] = sorted(raw_dtypes)
    reported["mask_area_distribution"] = {
        "overall": _percentile_stats(mask_areas_overall),
        "per_class": {str(k): _percentile_stats(v) for k, v in mask_areas_by_class.items()},
    }
    reported["bbox_longest_side_distribution"] = {
        "overall": _percentile_stats(bbox_longest_overall),
        "per_class": {str(k): _percentile_stats(v) for k, v in bbox_longest_by_class.items()},
    }
    reported["slices_per_patient_distribution"] = _percentile_stats(slices_per_patient.tolist())

    audit_report = {
        "contract_version": contract["contract_version"],
        "code_commit": _git_commit(),
        "hard_checks": hard,
        "reported": reported,
    }

    artifacts_dir = config.repo_root() / ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(artifacts_dir / "manifest.csv", index=False)
    with (artifacts_dir / "dataset_audit.json").open("w") as handle:
        json.dump(audit_report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    all_pass = all(check["pass"] for check in hard.values())
    return all_pass, audit_report


def main():
    all_pass, audit_report = run_audit()
    for name, check in sorted(audit_report["hard_checks"].items()):
        status = "PASS" if check["pass"] else "FAIL"
        print(f"[{status}] {name}")
    if not all_pass:
        print("AUDIT FAILED -- see phase2/artifacts/contract/dataset_audit.json", file=sys.stderr)
        sys.exit(1)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
