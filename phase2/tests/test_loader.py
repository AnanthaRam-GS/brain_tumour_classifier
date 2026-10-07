import torch
from torch.utils.data import DataLoader

from btdl.data.dataset import RoiDataset
from btdl.data.loader import make_loader
from btdl.data.sampler import EpochSampler


def _collect_one_epoch(loader, sampler, epoch):
    sampler.set_epoch(epoch)
    batches = []
    for batch in loader:
        batches.append(
            {
                "images": batch["image"].clone(),
                "targets": batch["target"].clone(),
                "sample_ids": list(batch["sample_id"]),
            }
        )
    return batches


def test_make_loader_train_uses_shuffle_and_epoch_aware_sampler(fake_repo):
    ds = RoiDataset("train", augment=True, seed=42)
    loader, sampler = make_loader(ds, batch_size=2, shuffle=True, seed=42, num_workers=0, device_type="cpu")
    assert isinstance(sampler, EpochSampler)
    assert sampler._shuffle is True
    assert sampler._epoch_aware is True


def test_make_loader_val_is_unshuffled(fake_repo):
    ds = RoiDataset("val", augment=False, seed=42)
    loader, sampler = make_loader(ds, batch_size=2, shuffle=False, seed=42, num_workers=0, device_type="cpu")
    assert sampler._shuffle is False
    assert sampler._epoch_aware is False


def test_pin_memory_only_on_cuda(fake_repo):
    ds = RoiDataset("val", augment=False, seed=42)
    loader_cpu, _ = make_loader(ds, batch_size=2, shuffle=False, seed=42, num_workers=0, device_type="cpu")
    assert loader_cpu.pin_memory is False
    loader_cuda, _ = make_loader(ds, batch_size=2, shuffle=False, seed=42, num_workers=0, device_type="cuda")
    assert loader_cuda.pin_memory is True


def test_drop_last_is_false(fake_repo):
    ds = RoiDataset("train", augment=True, seed=42)
    loader, _ = make_loader(ds, batch_size=3, shuffle=True, seed=42, num_workers=0, device_type="cpu")
    assert loader.drop_last is False


def test_sample_id_and_patient_id_come_back_as_lists_of_str(fake_repo):
    ds = RoiDataset("val", augment=False, seed=42)
    loader, sampler = make_loader(ds, batch_size=2, shuffle=False, seed=42, num_workers=0, device_type="cpu")
    batch = next(iter(loader))
    assert isinstance(batch["sample_id"], list)
    assert all(isinstance(s, str) for s in batch["sample_id"])
    assert isinstance(batch["patient_id"], list)
    assert all(isinstance(s, str) for s in batch["patient_id"])


def test_every_sample_appears_once_per_epoch(fake_repo):
    ds = RoiDataset("train", augment=True, seed=42)
    loader, sampler = make_loader(ds, batch_size=2, shuffle=True, seed=42, num_workers=0, device_type="cpu")
    batches = _collect_one_epoch(loader, sampler, epoch=0)
    seen_ids = [sid for batch in batches for sid in batch["sample_ids"]]
    assert sorted(seen_ids, key=int) == sorted(ds.sample_ids, key=int)


def test_num_workers_0_and_2_yield_identical_batches(fake_repo):
    # multiprocessing_context="fork" so the forked worker processes inherit
    # this test's monkeypatched btdl.config.repo_root (-> tmp_path); the
    # default "spawn" context on macOS re-imports modules fresh in each
    # worker and would NOT see the monkeypatch, pointing workers at the real
    # repo instead of the synthetic fake_repo. make_loader() itself does not
    # force a context (spawn is the safer default for real CUDA/MPS
    # training), so this test builds the DataLoader directly.
    ds0 = RoiDataset("train", augment=True, seed=42)
    sampler0 = EpochSampler(len(ds0), shuffle=True, seed=42, epoch_aware=True)
    loader0 = DataLoader(ds0, batch_size=2, sampler=sampler0, num_workers=0)
    batches0 = _collect_one_epoch(loader0, sampler0, epoch=0)

    ds2 = RoiDataset("train", augment=True, seed=42)
    sampler2 = EpochSampler(len(ds2), shuffle=True, seed=42, epoch_aware=True)
    loader2 = DataLoader(
        ds2,
        batch_size=2,
        sampler=sampler2,
        num_workers=2,
        multiprocessing_context="fork",
    )
    batches2 = _collect_one_epoch(loader2, sampler2, epoch=0)

    assert len(batches0) == len(batches2)
    for b0, b2 in zip(batches0, batches2):
        assert b0["sample_ids"] == b2["sample_ids"]
        assert torch.equal(b0["targets"], b2["targets"])
        assert torch.allclose(b0["images"], b2["images"], atol=1e-6)
