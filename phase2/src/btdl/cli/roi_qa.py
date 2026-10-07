"""python -m btdl.cli.roi_qa

Writes a deterministic visual QA contact sheet (phase2/artifacts/qa/
roi_contact_sheet.png) sampling TRAIN-split tiles: per class, the 2
smallest/median/largest crop_side samples; every train sample with any
out-of-bounds padding (cap 6 overall); and one 256x256-source sample.
Each tile shows the full normalized image (tight bbox in green, crop box
in red) next to the cached 224x224 ROI.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from btdl import config
from btdl.contracts import CLASS_NAMES
from btdl.data.roi_cache import open_roi_cache

OUTPUT_RELPATH = "phase2/artifacts/qa/roi_contact_sheet.png"
OOB_CAP = 6


def _select_samples(geometry: pd.DataFrame, manifest: pd.DataFrame):
    train = geometry[geometry["split"] == "train"].copy()
    train["sample_id_int"] = train["sample_id"].astype(int)
    train = train.sort_values(["crop_side", "sample_id_int"]).reset_index(drop=True)

    selected = {}  # sample_id -> reason

    for label in sorted(train["label"].unique()):
        sub = train[train["label"] == label].reset_index(drop=True)
        n = len(sub)
        if n == 0:
            continue
        smallest = sub.iloc[: min(2, n)]
        largest = sub.iloc[max(0, n - 2) :]
        mid_lo = max(0, n // 2 - 1)
        median = sub.iloc[mid_lo : mid_lo + 2]
        for _, row in pd.concat([smallest, median, largest]).iterrows():
            selected.setdefault(row["sample_id"], f"class {label} crop_side extreme/median")

    train["has_oob"] = (
        train[["oob_top", "oob_bottom", "oob_left", "oob_right"]].sum(axis=1) > 0
    )
    oob_rows = train[train["has_oob"]].sort_values("sample_id_int").head(OOB_CAP)
    for _, row in oob_rows.iterrows():
        selected.setdefault(row["sample_id"], "out-of-bounds padding")

    manifest_train = manifest[manifest["split"] == "train"].copy()
    manifest_train["sample_id_int"] = manifest_train["sample_id"].astype(int)
    small_shape = manifest_train[manifest_train["height"] == 256].sort_values("sample_id_int")
    if len(small_shape) > 0:
        row = small_shape.iloc[0]
        selected.setdefault(row["sample_id"], "256x256 source image")

    ordered_ids = sorted(selected.keys(), key=int)
    return ordered_ids, selected


def build_contact_sheet():
    data_contract = config.load_contract("data")
    repo_root = config.repo_root()

    geometry = pd.read_csv(
        repo_root / "phase2" / "artifacts" / "contract" / "roi_geometry.csv",
        dtype={"sample_id": str},
    )
    manifest = pd.read_csv(repo_root / data_contract["manifest"], dtype={"sample_id": str})
    samples_dir = config.resolve_data_path(data_contract["samples_dir"])

    sample_ids, reasons = _select_samples(geometry, manifest)
    cache = open_roi_cache()
    geometry_by_id = {row.sample_id: row for row in geometry.itertuples(index=False)}

    n_tiles = len(sample_ids)
    fig, axes = plt.subplots(n_tiles, 2, figsize=(6, 3 * n_tiles))
    if n_tiles == 1:
        axes = axes.reshape(1, 2)

    for i, sample_id in enumerate(sample_ids):
        with np.load(samples_dir / f"{sample_id}.npz") as data:
            image = data["image_normalized"]
            label = int(data["label"])

        row = geometry_by_id[sample_id]
        tight_r0, tight_r1, tight_c0, tight_c1 = _recompute_tight_bbox(samples_dir, sample_id)
        crop_r0, crop_r1, crop_c0, crop_c1 = row.crop_r0, row.crop_r1, row.crop_c0, row.crop_c1

        ax_full = axes[i, 0]
        ax_full.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax_full.add_patch(
            Rectangle(
                (tight_c0, tight_r0),
                tight_c1 - tight_c0,
                tight_r1 - tight_r0,
                edgecolor="lime",
                facecolor="none",
                linewidth=1.5,
            )
        )
        ax_full.add_patch(
            Rectangle(
                (crop_c0, crop_r0),
                crop_c1 - crop_c0,
                crop_r1 - crop_r0,
                edgecolor="red",
                facecolor="none",
                linewidth=1.5,
            )
        )
        ax_full.set_title(
            f"id={sample_id} {CLASS_NAMES[label]}\ncrop_side={row.crop_side} "
            f"upscale={row.upscale_factor:.2f}\n({reasons[sample_id]})",
            fontsize=7,
        )
        ax_full.axis("off")

        ax_roi = axes[i, 1]
        ax_roi.imshow(np.asarray(cache[sample_id]), cmap="gray", vmin=0, vmax=1)
        ax_roi.set_title("cached 224 ROI", fontsize=7)
        ax_roi.axis("off")

    fig.tight_layout()
    output_path = repo_root / OUTPUT_RELPATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path, n_tiles


def _recompute_tight_bbox(samples_dir, sample_id):
    from btdl.preprocessing.roi import tight_bbox

    with np.load(samples_dir / f"{sample_id}.npz") as data:
        mask = data["tumor_mask"]
    return tight_bbox(mask)


def main():
    path, n_tiles = build_contact_sheet()
    print(f"wrote {path} ({n_tiles} tiles)")


if __name__ == "__main__":
    main()
