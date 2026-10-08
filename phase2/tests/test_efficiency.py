import torch
import torch.nn as nn

from btdl.evaluation.efficiency import count_parameters, measure_inference
from btdl.training.device import select_device

DEVICE = select_device("cpu")


def _toy_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 3),
    )


def test_count_parameters_known_toy_model():
    model = nn.Linear(10, 3)  # 10*3 weights + 3 bias = 33
    result = count_parameters(model)
    assert result["total"] == 33
    assert result["trainable"] == 33


def test_count_parameters_with_frozen_layer():
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))
    for param in model[0].parameters():
        param.requires_grad = False
    result = count_parameters(model)
    total_expected = (10 * 5 + 5) + (5 * 3 + 3)
    trainable_expected = 5 * 3 + 3
    assert result["total"] == total_expected
    assert result["trainable"] == trainable_expected


def test_measure_inference_returns_expected_batch_sizes():
    model = _toy_cnn()
    result = measure_inference(model, DEVICE, batch_sizes=(1, 4), warmup=2, iters=5)
    assert set(result.keys()) == {1, 4}


def test_measure_inference_timings_are_positive():
    model = _toy_cnn()
    result = measure_inference(model, DEVICE, batch_sizes=(1,), warmup=2, iters=5)
    stats = result[1]
    assert stats["median_ms_per_batch"] > 0
    assert stats["p90_ms_per_batch"] > 0
    assert stats["ms_per_image"] > 0
    assert stats["p90_ms_per_batch"] >= stats["median_ms_per_batch"] * 0.5  # sanity, not strict


def test_measure_inference_ms_per_image_scales_with_batch_size():
    model = _toy_cnn()
    result = measure_inference(model, DEVICE, batch_sizes=(1, 32), warmup=3, iters=10)
    # ms_per_image = median_ms_per_batch / batch_size, by construction.
    assert result[32]["ms_per_image"] == result[32]["median_ms_per_batch"] / 32
    assert result[1]["ms_per_image"] == result[1]["median_ms_per_batch"] / 1


def test_measure_inference_device_description_included():
    model = _toy_cnn()
    result = measure_inference(model, DEVICE, batch_sizes=(1,), warmup=1, iters=3)
    assert result[1]["device"]["type"] == "cpu"


def test_measure_inference_leaves_model_in_eval_mode():
    model = _toy_cnn()
    model.train()
    measure_inference(model, DEVICE, batch_sizes=(1,), warmup=1, iters=3)
    assert model.training is False
