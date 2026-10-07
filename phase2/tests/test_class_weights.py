import pytest
import torch

from btdl.data.class_weights import compute_class_weights
from btdl.data.dataset import RoiDataset


class _FakeDataset:
    def __init__(self, split, class_counts):
        self.split = split
        self.class_counts = class_counts


def test_synthetic_class_weights_formula():
    # n=10, K=3, counts {1: 5, 2: 3, 3: 2}
    ds = _FakeDataset("train", {1: 5, 2: 3, 3: 2})
    weights = compute_class_weights(ds)
    expected = torch.tensor(
        [10 / (3 * 5), 10 / (3 * 3), 10 / (3 * 2)], dtype=torch.float32
    )
    assert torch.allclose(weights, expected, atol=1e-6)


def test_class_weights_ordered_by_internal_index():
    ds = _FakeDataset("train", {1: 1, 2: 2, 3: 3})
    weights = compute_class_weights(ds)
    # internal index 0 -> label 1, 1 -> label 2, 2 -> label 3
    n = 6
    assert weights[0].item() == pytest.approx(n / (3 * 1))
    assert weights[1].item() == pytest.approx(n / (3 * 2))
    assert weights[2].item() == pytest.approx(n / (3 * 3))


def test_non_train_split_refused():
    ds = _FakeDataset("val", {1: 1, 2: 1, 3: 1})
    with pytest.raises(ValueError):
        compute_class_weights(ds)


def test_zero_count_class_refused():
    ds = _FakeDataset("train", {1: 5, 2: 0, 3: 2})
    with pytest.raises(ValueError):
        compute_class_weights(ds)


def test_against_real_roi_dataset_train(fake_repo):
    ds = RoiDataset("train", augment=False, seed=42)
    weights = compute_class_weights(ds)
    assert weights.shape == (3,)
    assert weights.dtype == torch.float32
    n = sum(ds.class_counts.values())
    for label, count in ds.class_counts.items():
        from btdl.contracts import label_to_index

        assert weights[label_to_index(label)].item() == pytest.approx(n / (3 * count))
