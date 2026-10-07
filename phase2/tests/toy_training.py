"""Shared tiny, genuinely learnable synthetic task for trainer tests.

Images are small (not hard-coded to 224) and the model pools to a 3-vector
before a linear layer, so the class signal (channel mean) is directly
accessible -- this exercises the training engine's mechanics without
fighting a hard optimization landscape.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

IMAGE_SIZE = 16


class ToyDataset(Dataset):
    def __init__(self, n: int, *, augment: bool, seed: int, image_size: int = IMAGE_SIZE):
        self.n = n
        self.augment = augment
        generator = torch.Generator().manual_seed(seed)
        self.images = []
        self.labels = []
        for i in range(n):
            cls = i % 3
            base = torch.full((3, image_size, image_size), (cls + 1) * 0.2)
            noise = torch.randn(3, image_size, image_size, generator=generator) * 0.02
            self.images.append(torch.clamp(base + noise, 0.0, 1.0))
            self.labels.append(cls + 1)  # project label 1..3
        self.sample_ids = [str(i) for i in range(n)]
        self.patient_ids = [f"p{i}" for i in range(n)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, key):
        index = key[0] if self.augment else key
        return {
            "image": self.images[index],
            "target": torch.tensor(self.labels[index] - 1, dtype=torch.int64),
            "label": self.labels[index],
            "sample_id": self.sample_ids[index],
            "patient_id": self.patient_ids[index],
        }


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, 3)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class TupleOutputModel(nn.Module):
    """Mimics a model with an auxiliary output head (e.g. GoogLeNet-style)."""

    def __init__(self):
        super().__init__()
        self.inner = ToyModel()

    def forward(self, x):
        output = self.inner(x)
        return output, output  # (main, aux) -- not a bare Tensor


DEFAULT_TRAINING_CFG = {
    "max_epochs": 8,
    "optimizer": {"weight_decay": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8},
    "scheduler": {"t_max": "max_epochs", "eta_min": 0.0},
    "early_stopping": {"patience": 100, "min_delta": 0.0, "mode": "max"},
}


def make_training_cfg(**overrides):
    cfg = {
        "max_epochs": DEFAULT_TRAINING_CFG["max_epochs"],
        "optimizer": dict(DEFAULT_TRAINING_CFG["optimizer"]),
        "scheduler": dict(DEFAULT_TRAINING_CFG["scheduler"]),
        "early_stopping": dict(DEFAULT_TRAINING_CFG["early_stopping"]),
    }
    cfg.update(overrides)
    return cfg
