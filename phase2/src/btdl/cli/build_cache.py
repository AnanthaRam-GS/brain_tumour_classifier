"""python -m btdl.cli.build_cache [--verify]

Builds the deterministic 224x224 ROI cache (phase2/cache/roi224_v1.npy,
git-ignored) from the committed manifest, plus tracked provenance
artifacts (roi_geometry.csv, roi_cache_meta.json). --verify recomputes the
cache's hashes and compares them against the committed metadata.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from btdl import config
from btdl.data.manifest import content_sha256 as npz_content_sha256
from btdl.preprocessing.resize import resize_roi
from btdl.preprocessing.roi import compute_roi

CACHE_NAME = "roi224_v1"
CACHE_RELPATH = f"phase2/cache/{CACHE_NAME}.npy"
GEOMETRY_RELPATH = "phase2/artifacts/contract/roi_geometry.csv"
META_RELPATH = "phase2/artifacts/contract/roi_cache_meta.json"
INPUT_CONTRACT_RELPATH = "phase2/configs/contract/input.yaml"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.repo_root(), text=True
        ).strip()
    except Exception:
        return "unknown"


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _sample_order_sha256(sample_ids) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_cache(output_dir=None) -> dict:
    """Build the cache + provenance artifacts. Reads real data from the repo;
    writes outputs under `output_dir` if given, else under the repo root."""

    data_contract = config.load_contract("data")
    input_contract = config.load_contract("input")
    repo_root = config.repo_root()

    manifest_relpath = data_contract["manifest"]
    manifest_path = repo_root / manifest_relpath
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str, "patient_id": str})
    samples_dir = config.resolve_data_path(data_contract["samples_dir"])

    size = tuple(int(s) for s in input_contract["resize"]["size"])
    cache_array = np.zeros((len(manifest), size[0], size[1]), dtype=np.float16)

    geometry_rows = []
    for i, row in enumerate(manifest.itertuples(index=False)):
        npz_path = samples_dir / f"{row.sample_id}.npz"
        actual_hash = npz_content_sha256(npz_path)
        if actual_hash != row.content_sha256:
            raise ValueError(
                f"sample {row.sample_id} content_sha256 mismatch vs manifest: "
                f"expected {row.content_sha256}, got {actual_hash}"
            )

        with np.load(npz_path) as data:
            image = data["image_normalized"]
            mask = data["tumor_mask"]

        geometry, crop = compute_roi(image, mask, input_contract)
        resized = resize_roi(crop, input_contract)
        cache_array[i] = resized.astype(np.float16)

        geometry_rows.append(
            {
                "sample_id": row.sample_id,
                "split": row.split,
                "label": row.label,
                "crop_r0": geometry.crop_box[0],
                "crop_r1": geometry.crop_box[1],
                "crop_c0": geometry.crop_box[2],
                "crop_c1": geometry.crop_box[3],
                "crop_side": geometry.crop_side,
                "oob_top": geometry.oob_top,
                "oob_bottom": geometry.oob_bottom,
                "oob_left": geometry.oob_left,
                "oob_right": geometry.oob_right,
                "upscale_factor": geometry.upscale_factor,
            }
        )

    output_root = Path(output_dir) if output_dir is not None else repo_root

    cache_path = output_root / CACHE_RELPATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, cache_array)

    geometry_path = output_root / GEOMETRY_RELPATH
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(geometry_rows).to_csv(geometry_path, index=False)

    meta = {
        "cache_name": CACHE_NAME,
        "cache_relpath": CACHE_RELPATH,
        "manifest_relpath": str(manifest_relpath),
        "shape": list(cache_array.shape),
        "dtype": str(cache_array.dtype),
        "npy_sha256": _file_sha256(cache_path),
        "sample_order_sha256": _sample_order_sha256(list(manifest["sample_id"])),
        "input_yaml_sha256": _file_sha256(repo_root / INPUT_CONTRACT_RELPATH),
        "manifest_sha256": _file_sha256(manifest_path),
        "code_commit": _git_commit(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }
    meta_path = output_root / META_RELPATH
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return meta


def verify_cache() -> tuple:
    repo_root = config.repo_root()
    meta_path = repo_root / META_RELPATH
    with meta_path.open() as handle:
        meta = json.load(handle)

    cache_path = repo_root / meta["cache_relpath"]
    actual_npy_sha256 = _file_sha256(cache_path)

    manifest_path = repo_root / meta["manifest_relpath"]
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    actual_order_sha256 = _sample_order_sha256(list(manifest["sample_id"]))

    ok = actual_npy_sha256 == meta["npy_sha256"] and actual_order_sha256 == meta["sample_order_sha256"]
    details = {
        "npy_sha256_expected": meta["npy_sha256"],
        "npy_sha256_actual": actual_npy_sha256,
        "sample_order_sha256_expected": meta["sample_order_sha256"],
        "sample_order_sha256_actual": actual_order_sha256,
    }
    return ok, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        ok, details = verify_cache()
        print(json.dumps(details, indent=2, sort_keys=True))
        if not ok:
            print("CACHE VERIFY FAILED", file=sys.stderr)
            sys.exit(1)
        print("CACHE VERIFY PASSED")
        return

    meta = build_cache()
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
