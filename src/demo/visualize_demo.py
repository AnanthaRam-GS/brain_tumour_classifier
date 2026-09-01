"""Presentation figure and JSON output for Review 1 single-image demos."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_MPL_CONFIG_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "cv-review-demo-matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np

from src.data.feature_dataset import Phase1Sample
from src.demo.inference import CLASS_NAMES, DemoPrediction
from src.preprocessing.roi import TumorROI


def prediction_metadata(prediction: DemoPrediction) -> dict[str, object]:
    """Return strict JSON-serializable demo metadata."""

    payload = asdict(prediction)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["split"] = "test"
    return payload


def write_demo_metadata(prediction: DemoPrediction, path: Path | str) -> Path:
    """Write one demo metadata JSON file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(prediction_metadata(prediction), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return output


def save_demo_figure(
    *,
    sample: Phase1Sample,
    roi: TumorROI,
    prediction: DemoPrediction,
    output_path: Path | str,
    dpi: int = 300,
) -> Path:
    """Save a four-panel single-sample demo figure."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    axes[0, 0].imshow(sample.image_normalized, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Normalized MRI")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sample.image_normalized, cmap="gray", vmin=0, vmax=1)
    overlay = np.ma.masked_where(sample.tumor_mask.astype(bool) == 0, sample.tumor_mask)
    axes[0, 1].imshow(overlay, cmap="Reds", alpha=0.45)
    axes[0, 1].set_title("Tumor Mask Overlay")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(roi.roi_image, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Tumor ROI")
    axes[1, 0].axis("off")

    summary_axis = axes[1, 1]
    summary_axis.axis("off")
    confidence_text = (
        f"{prediction.confidence * 100:.2f}%" if prediction.confidence is not None else "Unavailable"
    )
    result_text = "CORRECT" if prediction.correct else "INCORRECT"
    summary = (
        f"Feature: {prediction.feature_name}\n"
        f"Model: {prediction.model_name}\n\n"
        f"Sample: {prediction.sample_id}\n"
        f"Patient: {prediction.patient_id}\n\n"
        f"Actual: {prediction.true_class_name}\n"
        f"Predicted: {prediction.predicted_class_name}\n"
        f"Confidence: {confidence_text}\n"
        f"Result: {result_text}"
    )
    summary_axis.text(0.02, 0.98, summary, va="top", ha="left", fontsize=12)
    if prediction.class_probabilities:
        names = list(prediction.class_probabilities)
        values = [prediction.class_probabilities[name] for name in names]
        inset = summary_axis.inset_axes([0.05, 0.05, 0.9, 0.34])
        y_positions = np.arange(len(names))
        inset.barh(y_positions, values)
        inset.set_yticks(y_positions, labels=names)
        inset.set_xlim(0, 1)
        inset.set_xlabel("Probability")
        inset.tick_params(axis="both", labelsize=8)
    fig.suptitle("Brain Tumor Classification - Review 1 Live Demo", fontsize=14)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def save_demo_outputs(
    *,
    sample: Phase1Sample,
    roi: TumorROI,
    prediction: DemoPrediction,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Write the demo PNG and JSON files for one sample."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"demo_sample_{prediction.sample_id}"
    figure_path = save_demo_figure(
        sample=sample,
        roi=roi,
        prediction=prediction,
        output_path=output_root / f"{stem}.png",
    )
    metadata_path = write_demo_metadata(prediction, output_root / f"{stem}.json")
    return {"figure": figure_path, "metadata": metadata_path}
