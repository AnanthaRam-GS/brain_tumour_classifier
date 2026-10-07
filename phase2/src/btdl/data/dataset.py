"""Phase 2 ROI dataset: torch.utils.data.Dataset over the frozen split + ROI cache."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from btdl import config
from btdl.contracts import label_to_index
from btdl.data.roi_cache import open_roi_cache
from btdl.data.split import load_split
from btdl.preprocessing.augmentation import apply_augmentation, sample_params
from btdl.preprocessing.model_input import to_model_input

VALID_SPLITS = ("train", "val", "test")


class RoiDataset(Dataset):
    """ROI/label access for one split. Sample ids come only from btdl.data.split.

    Opens the ROI cache lazily, once per worker process: __init__ opens it
    once only to validate against the manifest, then discards that handle
    so the dataset pickles cleanly to DataLoader workers; each worker opens
    its own cache (a read-only memmap) on first __getitem__ call.
    """

    def __init__(self, split: str, *, augment: bool, seed: int, allow_test: bool = False):
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        if split == "test" and not allow_test:
            raise ValueError("split='test' requires allow_test=True (see docs/DECISIONS.md D13)")
        if augment and split != "train":
            raise ValueError("augment=True is only allowed when split == 'train'")

        self._split = split
        self._augment = augment
        self._seed = seed

        split_index = load_split()
        sample_ids = split_index.sample_ids(split)

        data_contract = config.load_contract("data")
        manifest_path = config.repo_root() / data_contract["manifest"]
        manifest = pd.read_csv(manifest_path, dtype={"sample_id": str, "patient_id": str})
        manifest_by_id = {row.sample_id: row for row in manifest.itertuples(index=False)}

        cache = open_roi_cache()
        missing_in_cache = [sid for sid in sample_ids if sid not in cache.sample_id_to_row]
        if missing_in_cache:
            raise ValueError(
                f"{len(missing_in_cache)} {split!r} sample_ids missing from the ROI cache "
                f"(rebuild it): {missing_in_cache[:10]}"
            )

        labels = []
        patient_ids = []
        for sample_id in sample_ids:
            split_row = split_index[sample_id]
            manifest_row = manifest_by_id.get(sample_id)
            if manifest_row is None:
                raise ValueError(f"sample_id {sample_id!r} is in the split but missing from the manifest")
            if int(manifest_row.label) != split_row["label"]:
                raise ValueError(
                    f"label mismatch for sample_id {sample_id!r}: "
                    f"manifest={manifest_row.label}, split={split_row['label']}"
                )
            if str(manifest_row.patient_id) != split_row["patient_id"]:
                raise ValueError(f"patient_id mismatch for sample_id {sample_id!r}")
            labels.append(int(manifest_row.label))
            patient_ids.append(str(manifest_row.patient_id))

        self._sample_ids = tuple(sample_ids)
        self._labels = tuple(labels)
        self._patient_ids = tuple(patient_ids)

        self._augmentation_cfg = config.load_contract("augmentation") if augment else None
        self._model_input_cfg = config.load_contract("input")
        self._cache = None  # opened lazily, per worker process

    def _get_cache(self):
        if self._cache is None:
            self._cache = open_roi_cache()
        return self._cache

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, key):
        if self._augment:
            if not (isinstance(key, tuple) and len(key) == 2):
                raise TypeError(f"augment=True requires key=(index, epoch), got {key!r}")
            index, epoch = key
        else:
            if isinstance(key, tuple):
                raise TypeError(f"augment=False requires an int index, got a tuple: {key!r}")
            index = key
            epoch = None

        index = int(index)
        sample_id = self._sample_ids[index]
        label = self._labels[index]
        patient_id = self._patient_ids[index]

        cache = self._get_cache()
        roi = np.asarray(cache[sample_id], dtype=np.float32)
        roi_tensor = torch.from_numpy(roi).unsqueeze(0)  # [1, H, W]

        if self._augment:
            params = sample_params(self._seed, epoch, sample_id, self._augmentation_cfg)
            roi_tensor = apply_augmentation(roi_tensor, params, self._augmentation_cfg)

        image = to_model_input(roi_tensor, self._model_input_cfg)

        return {
            "image": image,
            "target": torch.tensor(label_to_index(label), dtype=torch.int64),
            "label": label,
            "sample_id": sample_id,
            "patient_id": patient_id,
        }

    @property
    def split(self) -> str:
        return self._split

    @property
    def augment(self) -> bool:
        return self._augment

    @property
    def sample_ids(self) -> tuple:
        return self._sample_ids

    @property
    def labels(self) -> tuple:
        return self._labels

    @property
    def patient_ids(self) -> tuple:
        return self._patient_ids

    @property
    def class_counts(self) -> dict:
        counts = {1: 0, 2: 0, 3: 0}
        for label in self._labels:
            counts[label] += 1
        return counts
