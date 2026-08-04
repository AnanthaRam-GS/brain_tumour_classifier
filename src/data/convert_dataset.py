"""Convert validated Figshare MATLAB samples to canonical NumPy artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data.mat_loader import BrainTumourSample, find_mat_sample_files, load_mat_sample
from src.preprocessing.normalization import robust_foreground_percentile_normalize

CLASS_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}
NPZ_KEYS = {
    "image_raw", "image_normalized", "tumor_mask", "tumor_border",
    "label", "patient_id", "sample_id",
}
MANIFEST_COLUMNS = [
    "sample_id", "patient_id", "label", "class_name", "source_mat_path",
    "npz_path", "image_png_path", "mask_png_path", "image_height",
    "image_width", "raw_dtype", "raw_min", "raw_max", "normalized_min",
    "normalized_max", "normalization_low", "normalization_high",
    "foreground_pixel_count", "tumour_pixels", "tumour_fraction",
    "border_point_count", "empty_foreground", "constant_foreground",
    "border_warning", "normalization_warning", "conversion_status",
    "conversion_message",
]


class ConversionValidationError(ValueError):
    """Raised when a saved conversion artifact fails validation."""


def _scalar_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ConversionValidationError("string value is not scalar")
    return str(array.reshape(-1)[0])


def _scalar_int(value: np.ndarray) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ConversionValidationError("integer value is not scalar")
    number = array.reshape(-1)[0]
    if not np.issubdtype(np.asarray(number).dtype, np.integer):
        raise ConversionValidationError("label is not stored as an integer scalar")
    return int(number)


def validate_npz(
    npz_path: Path | str,
    expected_sample: BrainTumourSample | None = None,
    *,
    validate_filename: bool = True,
) -> dict[str, Any]:
    """Reload and validate one canonical NPZ, returning compact statistics."""

    path = Path(npz_path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(NPZ_KEYS.difference(archive.files))
            if missing:
                raise ConversionValidationError(f"NPZ missing key(s): {', '.join(missing)}")
            raw = archive["image_raw"]
            normalized = archive["image_normalized"]
            mask = archive["tumor_mask"]
            border = archive["tumor_border"]
            label = _scalar_int(archive["label"])
            patient_id = _scalar_text(archive["patient_id"])
            sample_id = _scalar_text(archive["sample_id"])
    except ConversionValidationError:
        raise
    except Exception as exc:
        raise ConversionValidationError(f"cannot reload NPZ: {exc}") from exc

    if raw.dtype != np.float32:
        raise ConversionValidationError(f"image_raw dtype is {raw.dtype}, expected float32")
    if normalized.dtype != np.float32:
        raise ConversionValidationError(
            f"image_normalized dtype is {normalized.dtype}, expected float32"
        )
    if mask.dtype != np.uint8:
        raise ConversionValidationError(f"tumor_mask dtype is {mask.dtype}, expected uint8")
    if raw.ndim != 2 or normalized.ndim != 2 or mask.ndim != 2:
        raise ConversionValidationError("raw image, normalized image, and mask must be 2D")
    if raw.shape != normalized.shape or raw.shape != mask.shape:
        raise ConversionValidationError(
            f"image/mask shape mismatch: raw={raw.shape}, normalized={normalized.shape}, mask={mask.shape}"
        )
    if not np.isfinite(raw).all():
        raise ConversionValidationError("image_raw contains NaN or infinite values")
    if not np.isfinite(normalized).all():
        raise ConversionValidationError("image_normalized contains NaN or infinite values")
    normalized_min = float(normalized.min())
    normalized_max = float(normalized.max())
    if normalized_min < 0.0 or normalized_max > 1.0:
        raise ConversionValidationError(
            f"normalized values outside [0, 1]: [{normalized_min}, {normalized_max}]"
        )
    mask_values = np.unique(mask)
    if not set(mask_values.tolist()).issubset({0, 1}):
        raise ConversionValidationError(f"tumor_mask is not binary: {mask_values.tolist()}")
    tumour_pixels = int(mask.sum())
    if tumour_pixels == 0:
        raise ConversionValidationError("tumor_mask is empty")
    if label not in CLASS_NAMES:
        raise ConversionValidationError(f"invalid label: {label}")
    if validate_filename and path.stem != sample_id:
        raise ConversionValidationError(
            f"sample ID {sample_id!r} does not match filename {path.stem!r}"
        )

    if expected_sample is not None:
        if raw.shape != expected_sample.image.shape:
            raise ConversionValidationError(
                f"output dimensions {raw.shape} differ from raw input {expected_sample.image.shape}"
            )
        if not np.array_equal(raw, expected_sample.image.astype(np.float32, copy=False)):
            raise ConversionValidationError("image_raw values differ from loader output")
        if not np.array_equal(mask, expected_sample.tumour_mask.astype(np.uint8)):
            raise ConversionValidationError("tumor_mask values differ from loader output")
        if label != expected_sample.label:
            raise ConversionValidationError("label does not match loader output")
        if patient_id != expected_sample.patient_id:
            raise ConversionValidationError("patient ID does not match loader output")
        if sample_id != expected_sample.sample_id:
            raise ConversionValidationError("sample ID does not match loader output")

    return {
        "image_shape": tuple(int(value) for value in raw.shape),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "normalized_min": normalized_min,
        "normalized_max": normalized_max,
        "tumour_pixels": tumour_pixels,
        "label": label,
        "patient_id": patient_id,
        "sample_id": sample_id,
        "border": border,
    }


def _validate_png(path: Path, shape: tuple[int, int], is_mask: bool) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)
            if image.format != "PNG":
                raise ConversionValidationError(f"{path} is not a PNG")
    except ConversionValidationError:
        raise
    except Exception as exc:
        raise ConversionValidationError(f"cannot read PNG {path}: {exc}") from exc
    if array.shape != shape:
        raise ConversionValidationError(f"PNG shape {array.shape} differs from {shape}")
    if array.dtype != np.uint8:
        raise ConversionValidationError(f"PNG dtype is {array.dtype}, expected uint8")
    if is_mask and not set(np.unique(array).tolist()).issubset({0, 255}):
        raise ConversionValidationError("mask PNG contains values other than 0 and 255")


def validate_sample_outputs(
    npz_path: Path,
    image_png_path: Path,
    mask_png_path: Path,
    expected_sample: BrainTumourSample,
) -> dict[str, Any]:
    """Validate all expected analytical and visual outputs for a sample."""

    stats = validate_npz(npz_path, expected_sample)
    _validate_png(image_png_path, stats["image_shape"], is_mask=False)
    _validate_png(mask_png_path, stats["image_shape"], is_mask=True)
    return stats


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray], sample: BrainTumourSample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix=f".{path.stem}-", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        validate_npz(temporary, sample, validate_filename=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_png(path: Path, array: np.ndarray, shape: tuple[int, int], is_mask: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".png", prefix=f".{path.stem}-", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            Image.fromarray(array, mode="L").save(handle, format="PNG", optimize=True)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_png(temporary, shape, is_mask=is_mask)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _prepare_border(border: np.ndarray) -> tuple[np.ndarray, int, str]:
    source = np.asarray(border)
    if source.size == 0:
        return np.asarray(source, dtype=np.float64), 0, "empty tumour border"
    try:
        numeric = np.asarray(source) if np.issubdtype(source.dtype, np.number) else source.astype(np.float64)
    except (TypeError, ValueError):
        return np.empty((0,), dtype=np.float64), 0, "non-numeric tumour border"
    warning_parts = []
    if not np.isfinite(numeric).all():
        warning_parts.append("tumour border contains non-finite coordinates")
    if numeric.size % 2:
        warning_parts.append("tumour border has an odd coordinate count")
    return numeric, int(numeric.size // 2), "; ".join(warning_parts)


def _artifact_paths(output_root: Path, sample_id: str) -> tuple[Path, Path, Path]:
    return (
        output_root / "samples" / f"{sample_id}.npz",
        output_root / "images" / f"{sample_id}.png",
        output_root / "masks" / f"{sample_id}.png",
    )


def _path_for_manifest(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _manifest_row(
    sample: BrainTumourSample,
    normalized: np.ndarray,
    normalization: dict[str, Any],
    border_point_count: int,
    border_warning: str,
    paths: tuple[Path, Path, Path],
    status: str,
    message: str = "",
) -> dict[str, Any]:
    mask = sample.tumour_mask.astype(np.uint8, copy=False)
    tumour_pixels = int(mask.sum())
    return {
        "sample_id": sample.sample_id,
        "patient_id": sample.patient_id,
        "label": sample.label,
        "class_name": CLASS_NAMES[sample.label],
        "source_mat_path": _path_for_manifest(Path(sample.source_path)),
        "npz_path": _path_for_manifest(paths[0]),
        "image_png_path": _path_for_manifest(paths[1]),
        "mask_png_path": _path_for_manifest(paths[2]),
        "image_height": int(sample.image.shape[0]),
        "image_width": int(sample.image.shape[1]),
        "raw_dtype": sample.image_dtype_original,
        "raw_min": float(sample.image.min()),
        "raw_max": float(sample.image.max()),
        "normalized_min": float(normalized.min()),
        "normalized_max": float(normalized.max()),
        "normalization_low": normalization["normalization_low"],
        "normalization_high": normalization["normalization_high"],
        "foreground_pixel_count": normalization["foreground_pixel_count"],
        "tumour_pixels": tumour_pixels,
        "tumour_fraction": float(tumour_pixels / mask.size),
        "border_point_count": border_point_count,
        "empty_foreground": normalization["empty_foreground"],
        "constant_foreground": normalization["constant_foreground"],
        "border_warning": border_warning,
        "normalization_warning": normalization["normalization_warning"],
        "conversion_status": status,
        "conversion_message": message,
    }


def _failure_row(path: Path, sample_id: str, message: str, output_root: Path) -> dict[str, Any]:
    row = {column: None for column in MANIFEST_COLUMNS}
    outputs = _artifact_paths(output_root, sample_id)
    row.update(
        sample_id=sample_id,
        source_mat_path=_path_for_manifest(path),
        npz_path=_path_for_manifest(outputs[0]),
        image_png_path=_path_for_manifest(outputs[1]),
        mask_png_path=_path_for_manifest(outputs[2]),
        conversion_status="failed",
        conversion_message=message,
    )
    return row


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".csv", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _audit_overlay_ids(raw_manifest: pd.DataFrame, seed: int = 2024) -> list[str]:
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    valid = raw_manifest
    if "validation_status" in valid.columns:
        valid = valid[valid["validation_status"].eq("valid")]
    for label in sorted(CLASS_NAMES):
        candidates = sorted(
            (str(int(value)) for value in valid.loc[valid["label"].eq(label), "sample_id"]),
            key=int,
        )
        if candidates:
            indices = rng.choice(len(candidates), size=min(4, len(candidates)), replace=False)
            selected.extend(candidates[int(index)] for index in sorted(indices))
    return selected


def _atomic_overlay(sample: BrainTumourSample, normalized: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(sample.image, cmap="gray")
    axes[0].set_title("Raw MRI")
    axes[1].imshow(normalized, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Normalized MRI")
    axes[2].imshow(sample.tumour_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Tumour mask")
    axes[3].imshow(normalized, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(
        np.ma.masked_where(~sample.tumour_mask, sample.tumour_mask),
        cmap="autumn", alpha=0.55, vmin=0, vmax=1,
    )
    axes[3].set_title("Normalized MRI + tumour")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Sample {sample.sample_id} | {CLASS_NAMES[sample.label]}")
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".png", prefix=f".{path.stem}-", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=140, format="png")
        plt.close(figure)
        with Image.open(temporary) as image:
            image.verify()
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _create_audit_overlays(
    ids: Iterable[str], file_by_id: dict[str, Path], output_root: Path
) -> list[str]:
    overlay_root = output_root / "audit_overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{sample_id}.png" for sample_id in ids}
    for stale in overlay_root.glob("*.png"):
        if stale.name not in expected_names:
            stale.unlink()
    created = []
    for sample_id in ids:
        path = file_by_id.get(sample_id)
        if path is None:
            continue
        try:
            sample = load_mat_sample(path)
            normalized, _ = robust_foreground_percentile_normalize(sample.image)
            _atomic_overlay(sample, normalized, overlay_root / f"{sample_id}.png")
            created.append(sample_id)
        except Exception:
            # Overlay diagnostics must never terminate or invalidate conversion.
            continue
    return created


def convert_dataset(
    dataset_root: Path | str,
    raw_manifest: Path | str,
    output_root: Path | str,
    report_path: Path | str,
    *,
    limit: int | None = None,
    sample_ids: Iterable[str | int] | None = None,
    overwrite: bool = False,
    seed: int = 2024,
) -> dict[str, Any]:
    """Convert selected raw samples and write a validated manifest and report."""

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    report_path = Path(report_path)
    for directory in ("samples", "images", "masks", "audit_overlays"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    files = find_mat_sample_files(dataset_root)
    id_counts = Counter(path.stem for path in files)
    duplicates = {key: count for key, count in id_counts.items() if count > 1}
    file_by_id = {path.stem: path for path in files}
    numeric_ids = sorted((int(path.stem) for path in files))
    expected_ids = set(range(1, max(numeric_ids, default=0) + 1))
    missing_ids = sorted(expected_ids.difference(numeric_ids))

    raw_frame = pd.read_csv(raw_manifest, dtype={"patient_id": str})
    required_raw_columns = {"sample_id", "patient_id", "label"}
    missing_columns = required_raw_columns.difference(raw_frame.columns)
    if missing_columns:
        raise ValueError(f"raw manifest missing column(s): {', '.join(sorted(missing_columns))}")
    raw_frame["sample_id"] = raw_frame["sample_id"].astype(int)
    raw_by_id = raw_frame.set_index("sample_id", drop=False)

    if sample_ids is not None:
        requested = [str(int(value)) for value in sample_ids]
        selected_files = [file_by_id[value] for value in requested if value in file_by_id]
    else:
        selected_files = files[:limit] if limit is not None else files
    if limit is not None and sample_ids is not None:
        selected_files = selected_files[:limit]

    rows: list[dict[str, Any]] = []
    counters = Counter()
    invalid_or_incomplete: list[str] = []
    mask_failures: list[dict[str, str]] = []
    dimension_mismatches: list[str] = []
    for source_path in selected_files:
        sample_id = source_path.stem
        paths = _artifact_paths(output_root, sample_id)
        existed = [path.exists() for path in paths]
        try:
            sample = load_mat_sample(source_path)
            if int(sample_id) not in raw_by_id.index:
                raise ConversionValidationError("sample missing from raw audit manifest")
            raw_record = raw_by_id.loc[int(sample_id)]
            if isinstance(raw_record, pd.DataFrame):
                raise ConversionValidationError("duplicate sample ID in raw audit manifest")
            if str(raw_record["patient_id"]) != sample.patient_id:
                raise ConversionValidationError("patient ID differs from raw audit manifest")
            if int(raw_record["label"]) != sample.label:
                raise ConversionValidationError("label differs from raw audit manifest")

            normalized, normalization = robust_foreground_percentile_normalize(sample.image)
            border, border_count, border_warning = _prepare_border(sample.tumour_border)
            existing_valid = False
            existing_error = ""
            if all(existed) and not overwrite:
                try:
                    validate_sample_outputs(*paths, sample)
                    existing_valid = True
                except ConversionValidationError as exc:
                    existing_error = str(exc)

            if existing_valid:
                counters["skipped"] += 1
                status = "skipped_existing"
            else:
                arrays = {
                    "image_raw": sample.image.astype(np.float32, copy=False),
                    "image_normalized": normalized.astype(np.float32, copy=False),
                    "tumor_mask": sample.tumour_mask.astype(np.uint8),
                    "tumor_border": border,
                    "label": np.asarray(sample.label, dtype=np.int64),
                    "patient_id": np.asarray(sample.patient_id),
                    "sample_id": np.asarray(sample.sample_id),
                }
                _atomic_npz(paths[0], arrays, sample)
                image_png = np.rint(normalized * 255.0).clip(0, 255).astype(np.uint8)
                mask_png = sample.tumour_mask.astype(np.uint8) * 255
                _atomic_png(paths[1], image_png, sample.image.shape, is_mask=False)
                _atomic_png(paths[2], mask_png, sample.image.shape, is_mask=True)
                validate_sample_outputs(*paths, sample)
                if overwrite and any(existed):
                    counters["overwritten"] += 1
                    status = "overwritten"
                elif any(existed):
                    counters["regenerated"] += 1
                    invalid_or_incomplete.append(sample_id)
                    status = "regenerated"
                else:
                    counters["converted"] += 1
                    status = "converted"
                if existing_error:
                    invalid_or_incomplete.append(sample_id)

            rows.append(
                _manifest_row(
                    sample, normalized, normalization, border_count, border_warning,
                    paths, status,
                )
            )
        except Exception as exc:
            message = str(exc)
            counters["failed"] += 1
            rows.append(_failure_row(source_path, sample_id, message, output_root))
            if "mask" in message.lower():
                mask_failures.append({"sample_id": sample_id, "message": message})
            if "dimension" in message.lower() or "shape mismatch" in message.lower():
                dimension_mismatches.append(sample_id)

    rows.sort(key=lambda row: int(row["sample_id"]))
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    _atomic_csv(manifest, output_root / "manifest.csv")

    successful = manifest[manifest["conversion_status"].ne("failed")]
    shape_distribution = (
        successful.groupby(["image_height", "image_width"]).size().rename("count").reset_index()
    )
    overlay_ids = _audit_overlay_ids(raw_frame, seed=seed)
    created_overlay_ids = _create_audit_overlays(overlay_ids, file_by_id, output_root)
    report = {
        "total_expected_samples": len(files),
        "selected_samples": len(selected_files),
        "successfully_converted_samples": int(len(successful)),
        "newly_converted_samples": counters["converted"],
        "skipped_existing_valid_samples": counters["skipped"],
        "overwritten_samples": counters["overwritten"],
        "regenerated_invalid_or_incomplete_samples": counters["regenerated"],
        "regenerated_sample_ids": sorted(set(invalid_or_incomplete), key=int),
        "failed_samples": counters["failed"],
        "failures": manifest.loc[
            manifest["conversion_status"].eq("failed"),
            ["sample_id", "source_mat_path", "conversion_message"],
        ].to_dict(orient="records"),
        "output_image_shape_distribution": [
            {
                "image_height": int(row.image_height),
                "image_width": int(row.image_width),
                "count": int(row.count),
            }
            for row in shape_distribution.itertuples(index=False)
        ],
        "raw_intensity_range": (
            [float(successful["raw_min"].min()), float(successful["raw_max"].max())]
            if not successful.empty else []
        ),
        "normalized_intensity_range": (
            [float(successful["normalized_min"].min()), float(successful["normalized_max"].max())]
            if not successful.empty else []
        ),
        "samples_with_empty_foreground": successful.loc[
            successful["empty_foreground"].eq(True), "sample_id"  # noqa: E712
        ].astype(str).tolist(),
        "samples_with_constant_foreground": successful.loc[
            successful["constant_foreground"].eq(True), "sample_id"  # noqa: E712
        ].astype(str).tolist(),
        "samples_with_tumour_border_warnings": successful.loc[
            successful["border_warning"].fillna("").ne(""), "sample_id"
        ].astype(str).tolist(),
        "samples_with_output_dimensions_different_from_raw": sorted(
            set(dimension_mismatches), key=int
        ),
        "missing_sample_ids": missing_ids,
        "duplicate_sample_ids": duplicates,
        "mask_validation_failures": mask_failures,
        "deterministic_audit_overlay_sample_ids": created_overlay_ids,
    }
    _atomic_json(report, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    report = convert_dataset(
        args.dataset_root,
        args.raw_manifest,
        args.output_root,
        args.report_path,
        limit=args.limit,
        sample_ids=args.sample_ids,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: report[key] for key in (
        "total_expected_samples", "selected_samples", "successfully_converted_samples",
        "skipped_existing_valid_samples", "regenerated_invalid_or_incomplete_samples",
        "failed_samples",
    )}, indent=2))


if __name__ == "__main__":
    main()
