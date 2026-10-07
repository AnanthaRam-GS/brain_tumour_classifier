import torch

import pytest

from btdl.contracts import label_to_index
from btdl.data.dataset import RoiDataset


def test_train_item_shape_dtype_keys(fake_repo):
    ds = RoiDataset("train", augment=True, seed=42)
    item = ds[(0, 0)]
    assert set(item.keys()) == {"image", "target", "label", "sample_id", "patient_id"}
    assert item["image"].shape == (3, 224, 224)
    assert item["image"].dtype == torch.float32
    assert item["target"].dtype == torch.int64
    assert isinstance(item["label"], int)
    assert isinstance(item["sample_id"], str)
    assert isinstance(item["patient_id"], str)


def test_val_item_shape_dtype_keys(fake_repo):
    ds = RoiDataset("val", augment=False, seed=42)
    item = ds[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["image"].dtype == torch.float32


def test_label_target_mapping(fake_repo):
    ds = RoiDataset("train", augment=False, seed=42)
    for index in range(len(ds)):
        item = ds[index]
        assert item["target"].item() == label_to_index(item["label"])


def test_test_split_refused_without_allow_test(fake_repo):
    with pytest.raises(ValueError):
        RoiDataset("test", augment=False, seed=42)


def test_test_split_allowed_with_allow_test(fake_repo):
    ds = RoiDataset("test", augment=False, seed=42, allow_test=True)
    assert len(ds) > 0


def test_augment_on_val_refused(fake_repo):
    with pytest.raises(ValueError):
        RoiDataset("val", augment=True, seed=42)


def test_augment_on_test_refused(fake_repo):
    with pytest.raises(ValueError):
        RoiDataset("test", augment=True, seed=42, allow_test=True)


def test_invalid_split_name_refused(fake_repo):
    with pytest.raises(ValueError):
        RoiDataset("bogus", augment=False, seed=42)


def test_wrong_key_type_refused_when_augment_true(fake_repo):
    ds = RoiDataset("train", augment=True, seed=42)
    with pytest.raises(TypeError):
        ds[0]


def test_wrong_key_type_refused_when_augment_false(fake_repo):
    ds = RoiDataset("val", augment=False, seed=42)
    with pytest.raises(TypeError):
        ds[(0, 0)]


def test_eval_items_are_deterministic(fake_repo):
    ds = RoiDataset("val", augment=False, seed=42)
    first = ds[0]
    second = ds[0]
    assert torch.equal(first["image"], second["image"])


def test_train_items_differ_across_epochs(fake_repo):
    ds = RoiDataset("train", augment=True, seed=42)
    epoch0 = ds[(0, 0)]
    epoch1 = ds[(0, 1)]
    assert not torch.equal(epoch0["image"], epoch1["image"])


def test_train_items_identical_for_same_epoch(fake_repo):
    ds = RoiDataset("train", augment=True, seed=42)
    a = ds[(0, 0)]
    b = ds[(0, 0)]
    assert torch.equal(a["image"], b["image"])


def test_class_counts_and_sample_ids_properties(fake_repo):
    ds = RoiDataset("train", augment=False, seed=42)
    assert sum(ds.class_counts.values()) == len(ds)
    assert len(ds.sample_ids) == len(ds)
    assert len(ds.labels) == len(ds)
