import time

import pytest
import torch

from btdl import config
from btdl.data.class_weights import compute_class_weights
from btdl.data.dataset import RoiDataset
from btdl.data.loader import make_loader

EXPECTED_CLASS_WEIGHTS = torch.tensor([1.3669, 0.7174, 1.1435], dtype=torch.float32)


def _real_data_available():
    contract = config.load_contract("data")
    samples_dir = config.resolve_data_path(contract["samples_dir"])
    return samples_dir.is_dir() and any(samples_dir.glob("*.npz"))


def _real_cache_available():
    meta_path = config.repo_root() / "phase2" / "artifacts" / "contract" / "roi_cache_meta.json"
    return meta_path.is_file()


pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (_real_data_available() and _real_cache_available()),
        reason="real NPZ dataset / ROI cache not present on disk",
    ),
]


def test_real_class_weights_match_expected():
    ds = RoiDataset("train", augment=False, seed=42)
    weights = compute_class_weights(ds)
    assert torch.allclose(weights, EXPECTED_CLASS_WEIGHTS, atol=1e-4)


def test_train_loader_ids_disjoint_by_patient_from_val_and_test():
    train_ds = RoiDataset("train", augment=False, seed=42)
    val_ds = RoiDataset("val", augment=False, seed=42)
    test_ds = RoiDataset("test", augment=False, seed=42, allow_test=True)

    train_patients = set(train_ds.patient_ids)
    val_patients = set(val_ds.patient_ids)
    test_patients = set(test_ds.patient_ids)

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)


def test_one_full_train_epoch_iterates_without_error():
    ds = RoiDataset("train", augment=True, seed=42)
    loader, sampler = make_loader(ds, batch_size=32, shuffle=True, seed=42, num_workers=0, device_type="cpu")
    sampler.set_epoch(0)
    n_seen = 0
    for batch in loader:
        n_seen += batch["image"].shape[0]
    assert n_seen == len(ds)


@pytest.mark.parametrize("num_workers", [0, 2, 4])
def test_throughput(num_workers):
    ds = RoiDataset("train", augment=True, seed=42)
    loader, sampler = make_loader(
        ds, batch_size=32, shuffle=True, seed=42, num_workers=num_workers, device_type="cpu"
    )
    sampler.set_epoch(0)

    start = time.perf_counter()
    n_seen = 0
    for batch in loader:
        n_seen += batch["image"].shape[0]
    elapsed = time.perf_counter() - start

    assert n_seen == len(ds)
    samples_per_second = n_seen / elapsed
    print(f"\nthroughput num_workers={num_workers}: {samples_per_second:.1f} samples/s ({elapsed:.2f}s for {n_seen} samples)")
