import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from btdl import config
from btdl.cli.build_cache import build_cache
from btdl.data.manifest import build_manifest
from btdl.data.roi_cache import RoiCacheError, open_roi_cache


def _write_npz(path, *, sample_id, patient_id, label, image, mask):
    np.savez(
        path,
        image_raw=image.astype(np.float32),
        image_normalized=image.astype(np.float32),
        tumor_mask=mask.astype(np.uint8),
        tumor_border=np.array([1.0, 2.0]),
        label=np.array(label),
        patient_id=np.array(patient_id),
        sample_id=np.array(sample_id),
    )


def _make_fake_repo(tmp_path):
    data_root = tmp_path
    samples_dir = data_root / "data" / "processed" / "samples"
    samples_dir.mkdir(parents=True)
    raw_dir = data_root / "raw"
    raw_dir.mkdir()

    rng = np.random.RandomState(0)
    specs = [("1", "p1", 1, 30), ("2", "p1", 2, 24), ("3", "p2", 3, 18)]
    for sample_id, patient_id, label, size in specs:
        image = rng.rand(size, size).astype(np.float32)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[size // 4 : size // 2, size // 4 : size // 2] = 1
        _write_npz(
            samples_dir / f"{sample_id}.npz",
            sample_id=sample_id,
            patient_id=patient_id,
            label=label,
            image=image,
            mask=mask,
        )
        (raw_dir / f"{sample_id}.mat").write_bytes(b"fake")

    split_csv = data_root / "split.csv"
    pd.DataFrame(
        {
            "sample_id": [s[0] for s in specs],
            "patient_id": [s[1] for s in specs],
            "label": [s[2] for s in specs],
            "split": ["train", "val", "test"],
        }
    ).to_csv(split_csv, index=False)

    manifest = build_manifest(
        samples_dir=samples_dir, raw_dir=raw_dir, split_csv=split_csv, data_root=data_root
    )
    contract_dir = tmp_path / "phase2" / "configs" / "contract"
    contract_dir.mkdir(parents=True)
    artifacts_dir = tmp_path / "phase2" / "artifacts" / "contract"
    artifacts_dir.mkdir(parents=True)
    manifest_path = artifacts_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    data_yaml = {
        "contract_version": "1.0.0",
        "raw_dir": "raw",
        "samples_dir": "data/processed/samples",
        "split_csv": "split.csv",
        "split_sha256": "",
        "manifest": "phase2/artifacts/contract/manifest.csv",
        "expected": {
            "n_samples": 3,
            "n_patients": 2,
            "class_counts": {1: 1, 2: 1, 3: 1},
            "split_samples": {"train": 1, "val": 1, "test": 1},
            "split_patients": {"train": 1, "val": 1, "test": 1},
        },
    }
    with (contract_dir / "data.yaml").open("w") as handle:
        yaml.safe_dump(data_yaml, handle)

    real_input_yaml = config.repo_root() / "phase2" / "configs" / "contract" / "input.yaml"
    shutil.copy(real_input_yaml, contract_dir / "input.yaml")

    return manifest


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    manifest = _make_fake_repo(tmp_path)
    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)
    return tmp_path, manifest


def test_build_cache_row_order_follows_manifest(fake_repo):
    tmp_path, manifest = fake_repo
    meta = build_cache()
    assert meta["shape"][0] == len(manifest)

    cache = open_roi_cache()
    for row in manifest.itertuples(index=False):
        assert row.sample_id in cache.sample_id_to_row
    # Row 0 in the cache corresponds to the first manifest row (sample_id sorted numerically).
    ordered_ids = list(manifest["sample_id"])
    assert list(cache.sample_id_to_row.keys()) == ordered_ids
    assert [cache.sample_id_to_row[sid] for sid in ordered_ids] == list(range(len(ordered_ids)))


def test_build_cache_is_byte_identical_on_rebuild(fake_repo):
    first = build_cache()
    second = build_cache()
    assert first["npy_sha256"] == second["npy_sha256"]
    assert first["sample_order_sha256"] == second["sample_order_sha256"]


def test_open_roi_cache_detects_tampered_npy(fake_repo):
    tmp_path, _ = fake_repo
    build_cache()
    cache_path = tmp_path / "phase2" / "cache" / "roi224_v1.npy"
    with cache_path.open("r+b") as handle:
        handle.seek(200)
        original = handle.read(1)
        handle.seek(200)
        handle.write(bytes([(original[0] + 1) % 256]))

    with pytest.raises(RoiCacheError):
        open_roi_cache(verify=True)


def test_open_roi_cache_detects_tampered_order(fake_repo):
    tmp_path, _ = fake_repo
    build_cache()
    manifest_path = tmp_path / "phase2" / "artifacts" / "contract" / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    reordered = manifest.iloc[::-1].reset_index(drop=True)
    reordered.to_csv(manifest_path, index=False)

    with pytest.raises(RoiCacheError):
        open_roi_cache()


def test_open_roi_cache_refuses_different_input_contract(fake_repo):
    tmp_path, _ = fake_repo
    build_cache()
    input_yaml_path = tmp_path / "phase2" / "configs" / "contract" / "input.yaml"
    with input_yaml_path.open() as handle:
        payload = yaml.safe_load(handle)
    payload["roi"]["padding_fraction"] = 0.25
    with input_yaml_path.open("w") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(RoiCacheError):
        open_roi_cache()


def test_open_roi_cache_missing_metadata_raises(tmp_path, monkeypatch):
    (tmp_path / "phase2").mkdir()
    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)
    with pytest.raises(RoiCacheError):
        open_roi_cache()


@pytest.mark.data
def test_real_cache_opens_and_verifies():
    contract = config.load_contract("data")
    samples_dir = config.resolve_data_path(contract["samples_dir"])
    if not samples_dir.is_dir() or not any(samples_dir.glob("*.npz")):
        pytest.skip("real NPZ dataset not present on disk")

    meta_path = config.repo_root() / "phase2" / "artifacts" / "contract" / "roi_cache_meta.json"
    if not meta_path.is_file():
        pytest.skip("real ROI cache not built")

    cache = open_roi_cache(verify=True)
    assert len(cache) == contract["expected"]["n_samples"]
