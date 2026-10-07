"""python -m btdl.cli.size_tertiles

Computes tertile edges of crop_side from the TRAIN split only
(roi_geometry.csv) and writes the tracked
phase2/artifacts/contract/size_tertiles.json.
"""

import json

import numpy as np
import pandas as pd

from btdl import config

OUTPUT_RELPATH = "phase2/artifacts/contract/size_tertiles.json"


def compute_size_tertiles(geometry: pd.DataFrame) -> dict:
    train = geometry[geometry["split"] == "train"]
    crop_sides = train["crop_side"].to_numpy()
    edge_low = float(np.percentile(crop_sides, 100 / 3))
    edge_high = float(np.percentile(crop_sides, 200 / 3))
    return {
        "edges": [edge_low, edge_high],
        "basis": "train_crop_side_tertiles",
        "n_train": int(len(crop_sides)),
        "min": float(crop_sides.min()),
        "max": float(crop_sides.max()),
    }


def build_and_write(geometry: pd.DataFrame = None) -> dict:
    repo_root = config.repo_root()
    if geometry is None:
        geometry = pd.read_csv(repo_root / "phase2" / "artifacts" / "contract" / "roi_geometry.csv")
    tertiles = compute_size_tertiles(geometry)

    output_path = repo_root / OUTPUT_RELPATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(tertiles, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return tertiles


def main():
    tertiles = build_and_write()
    print(json.dumps(tertiles, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
