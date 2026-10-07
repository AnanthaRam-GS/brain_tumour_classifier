"""DataLoader factory: ties a RoiDataset to an EpochSampler with the project's
fixed loader policy (pin_memory only on CUDA, no drop_last, persistent
workers when parallel, per-worker numpy/random seeding as a safety net)."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from btdl.data.sampler import EpochSampler


def _worker_init_fn(worker_id: int) -> None:
    # Safety net only: augmentation itself never depends on these seeds
    # (it is keyed by (seed, epoch, sample_id) via its own local generator).
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int, num_workers: int, device_type: str):
    """Returns (DataLoader, EpochSampler). Call sampler.set_epoch(epoch) before each epoch."""

    epoch_aware = bool(getattr(dataset, "augment", False))
    sampler = EpochSampler(len(dataset), shuffle=shuffle, seed=seed, epoch_aware=epoch_aware)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
    return loader, sampler
