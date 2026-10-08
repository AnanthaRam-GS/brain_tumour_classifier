import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from btdl.data.sampler import EpochSampler
from btdl.evaluation.predict import PredictError, Predictions, predict
from tests.toy_training import ToyDataset, ToyModel, TupleOutputModel


def _make_loader(n=12, seed=43):
    ds = ToyDataset(n, augment=False, seed=seed)
    sampler = EpochSampler(len(ds), shuffle=False, seed=seed, epoch_aware=False)
    return ds, DataLoader(ds, batch_size=5, sampler=sampler)


def test_predict_returns_predictions_dataclass():
    torch.manual_seed(0)
    model = ToyModel()
    ds, loader = _make_loader()
    result = predict(model, loader, torch.device("cpu"))
    assert isinstance(result, Predictions)


def test_predict_order_matches_dataset_order():
    torch.manual_seed(0)
    model = ToyModel()
    ds, loader = _make_loader(n=12)
    result = predict(model, loader, torch.device("cpu"))
    assert list(result.sample_ids) == ds.sample_ids
    assert list(result.patient_ids) == ds.patient_ids


def test_predict_probs_are_valid_distributions():
    torch.manual_seed(0)
    model = ToyModel()
    ds, loader = _make_loader()
    result = predict(model, loader, torch.device("cpu"))
    assert result.probs.shape == (len(ds), 3)
    assert np.all(result.probs >= 0.0)
    assert np.allclose(result.probs.sum(axis=1), 1.0, atol=1e-5)


def test_predict_y_true_idx_matches_dataset_targets():
    torch.manual_seed(0)
    model = ToyModel()
    ds, loader = _make_loader()
    result = predict(model, loader, torch.device("cpu"))
    expected = np.array([ds[i]["target"].item() for i in range(len(ds))])
    assert np.array_equal(result.y_true_idx, expected)


def test_predict_eval_mode_and_no_grad_are_used():
    torch.manual_seed(0)
    model = ToyModel()
    model.train()  # deliberately left in train mode before calling predict
    ds, loader = _make_loader()
    predict(model, loader, torch.device("cpu"))
    assert model.training is False  # predict() must switch to eval mode


def test_predict_rejects_tuple_output_model():
    torch.manual_seed(0)
    model = TupleOutputModel()
    ds, loader = _make_loader()
    with pytest.raises(PredictError):
        predict(model, loader, torch.device("cpu"))


def test_predict_deterministic_across_calls():
    torch.manual_seed(0)
    model = ToyModel()
    ds, loader = _make_loader()
    first = predict(model, loader, torch.device("cpu"))
    second = predict(model, loader, torch.device("cpu"))
    assert np.array_equal(first.probs, second.probs)
    assert first.sample_ids == second.sample_ids
