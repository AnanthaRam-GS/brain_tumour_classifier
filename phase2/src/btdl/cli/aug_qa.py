"""python -m btdl.cli.aug_qa

Writes a deterministic visual QA contact sheet
(phase2/artifacts/qa/augmentation_sheet.png): 2 TRAIN samples per class,
each shown as the cached (unaugmented) ROI plus 7 augmented views
(epochs 0..6, seed 42), all displayed in [0,1] before to_model_input's
channel replication/normalization.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from btdl import config
from btdl.contracts import CLASS_NAMES
from btdl.data.roi_cache import open_roi_cache
from btdl.preprocessing.augmentation import apply_augmentation, sample_params

OUTPUT_RELPATH = "phase2/artifacts/qa/augmentation_sheet.png"
SEED = 42
N_EPOCHS = 7


def _select_samples(manifest: pd.DataFrame):
    train = manifest[manifest["split"] == "train"].copy()
    train["sample_id_int"] = train["sample_id"].astype(int)
    train = train.sort_values("sample_id_int")

    selected = []
    for label in sorted(train["label"].unique()):
        sub = train[train["label"] == label].head(2)
        selected.extend(sub["sample_id"].tolist())
    return selected


def _format_params(params) -> str:
    flip = "flip" if params.hflip else "no-flip"
    return (
        f"{flip} rot={params.degrees:.1f} scale={params.scale:.2f}\n"
        f"tx,ty={params.translate_px} b={params.brightness_factor:.2f} c={params.contrast_factor:.2f}"
    )


def build_augmentation_sheet():
    repo_root = config.repo_root()
    data_contract = config.load_contract("data")
    augmentation_cfg = config.load_contract("augmentation")

    manifest = pd.read_csv(repo_root / data_contract["manifest"], dtype={"sample_id": str})
    sample_ids = _select_samples(manifest)
    labels_by_id = dict(zip(manifest["sample_id"], manifest["label"]))

    cache = open_roi_cache()

    n_rows = len(sample_ids)
    n_cols = 1 + N_EPOCHS
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.6 * n_rows))

    for row_idx, sample_id in enumerate(sample_ids):
        label = int(labels_by_id[sample_id])
        roi = np.asarray(cache[sample_id], dtype=np.float32)
        roi_tensor = torch.from_numpy(roi).unsqueeze(0)

        ax0 = axes[row_idx, 0]
        ax0.imshow(roi, cmap="gray", vmin=0, vmax=1)
        ax0.set_title(f"id={sample_id} {CLASS_NAMES[label]}\ncached ROI", fontsize=7)
        ax0.axis("off")

        for epoch in range(N_EPOCHS):
            params = sample_params(SEED, epoch, sample_id, augmentation_cfg)
            augmented = apply_augmentation(roi_tensor, params, augmentation_cfg)
            ax = axes[row_idx, 1 + epoch]
            ax.imshow(augmented.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"epoch={epoch}\n{_format_params(params)}", fontsize=5.5)
            ax.axis("off")

    fig.tight_layout()
    output_path = repo_root / OUTPUT_RELPATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path, n_rows


def main():
    path, n_rows = build_augmentation_sheet()
    print(f"wrote {path} ({n_rows} samples x {1 + N_EPOCHS} tiles)")


if __name__ == "__main__":
    main()
