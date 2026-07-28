"""Deterministic diagnostic figures for audited MRI samples."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.data.mat_loader import BrainTumourSample

CLASS_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def select_samples(
    samples: Iterable[BrainTumourSample], per_class: int = 4, seed: int = 2024
) -> list[BrainTumourSample]:
    """Choose a reproducible set with up to ``per_class`` samples per class."""

    rng = np.random.default_rng(seed)
    selected: list[BrainTumourSample] = []
    materialized = list(samples)
    for label in sorted(CLASS_NAMES):
        candidates = [sample for sample in materialized if sample.label == label]
        candidates.sort(key=lambda sample: int(sample.sample_id))
        if candidates:
            indices = rng.choice(len(candidates), size=min(per_class, len(candidates)), replace=False)
            selected.extend(candidates[int(index)] for index in sorted(indices))
    return selected


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(mask)
    if not rows.size:
        return None
    left, right = int(columns.min()), int(columns.max())
    top, bottom = int(rows.min()), int(rows.max())
    return left, top, right - left + 1, bottom - top + 1


def save_sample_figure(sample: BrainTumourSample, output_path: Path | str) -> Path:
    """Save MRI, mask, overlay, and bounding-box panels for one sample."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)

    axes[0].imshow(sample.image, cmap="gray")
    axes[0].set_title("Original MRI")
    axes[1].imshow(sample.tumour_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Tumour mask")
    axes[2].imshow(sample.image, cmap="gray")
    axes[2].imshow(
        np.ma.masked_where(~sample.tumour_mask, sample.tumour_mask),
        cmap="autumn",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Mask overlay")
    axes[3].imshow(sample.image, cmap="gray")
    box = _bounding_box(sample.tumour_mask)
    if box is not None:
        axes[3].add_patch(Rectangle(box[:2], box[2], box[3], fill=False, color="red", lw=2))
    axes[3].set_title("Tumour bounding box")

    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"Sample {sample.sample_id} | {CLASS_NAMES[sample.label]} | PID {sample.patient_id}"
    )
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def create_sample_figures(
    samples: Iterable[BrainTumourSample],
    output_dir: Path | str,
    per_class: int = 4,
    seed: int = 2024,
) -> list[Path]:
    """Create deterministic per-class diagnostic figures."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for stale_figure in output_root.glob("sample_*.png"):
        stale_figure.unlink()
    paths = []
    for sample in select_samples(samples, per_class=per_class, seed=seed):
        filename = f"sample_{int(sample.sample_id):04d}_{CLASS_NAMES[sample.label]}.png"
        paths.append(save_sample_figure(sample, output_root / filename))
    return paths
