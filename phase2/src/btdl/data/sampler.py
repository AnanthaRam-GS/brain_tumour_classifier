"""Epoch-aware sampler: a sample-keyed, reproducible iteration order."""

import hashlib

import torch
from torch.utils.data import Sampler


def _seed_from(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class EpochSampler(Sampler):
    """Yields each of range(n) exactly once per epoch.

    shuffle=True draws the permutation from a torch.Generator seeded by a
    stable hash of (seed, epoch) -- reproducible per epoch, different across
    epochs/seeds. shuffle=False always yields 0..n-1 in order.
    epoch_aware=True yields (index, epoch) tuples instead of plain indices,
    for datasets whose __getitem__ needs the epoch (e.g. augmentation).
    """

    def __init__(self, n: int, *, shuffle: bool, seed: int, epoch_aware: bool):
        self._n = n
        self._shuffle = shuffle
        self._seed = seed
        self._epoch_aware = epoch_aware
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        if self._shuffle:
            generator = torch.Generator()
            generator.manual_seed(_seed_from(self._seed, self._epoch))
            order = torch.randperm(self._n, generator=generator).tolist()
        else:
            order = list(range(self._n))

        if self._epoch_aware:
            return iter((index, self._epoch) for index in order)
        return iter(order)
