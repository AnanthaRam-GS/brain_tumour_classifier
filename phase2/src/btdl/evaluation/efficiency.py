"""Parameter counting and inference timing."""

import time

import torch

from btdl import config
from btdl.training.device import describe_device


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_inference(model, device, *, batch_sizes=None, warmup: int = None, iters: int = None) -> dict:
    """Per batch size: median_ms_per_batch, p90_ms_per_batch, ms_per_image, device.

    batch_sizes/warmup/iters default to configs/contract/evaluation.yaml's
    efficiency section; tests may pass smaller values explicitly.

    Eval mode, no_grad, random 3x224x224 input, with explicit device
    synchronization around each timed call (required for CUDA/MPS, whose
    kernel launches are asynchronous -- without it the wall-clock timing
    would measure launch overhead, not actual execution).
    """

    evaluation_cfg = config.load_contract("evaluation")
    if batch_sizes is None:
        batch_sizes = evaluation_cfg["efficiency"]["batch_sizes"]
    if warmup is None:
        warmup = evaluation_cfg["efficiency"]["warmup"]
    if iters is None:
        iters = evaluation_cfg["efficiency"]["iters"]

    model = model.to(device)
    model.eval()

    results = {}
    with torch.no_grad():
        for batch_size in batch_sizes:
            inputs = torch.rand(batch_size, 3, 224, 224, device=device)

            for _ in range(warmup):
                model(inputs)
            _synchronize(device)

            timings_ms = []
            for _ in range(iters):
                _synchronize(device)
                start = time.perf_counter()
                model(inputs)
                _synchronize(device)
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            timings_sorted = sorted(timings_ms)
            n = len(timings_sorted)
            median_ms = (
                timings_sorted[n // 2]
                if n % 2 == 1
                else (timings_sorted[n // 2 - 1] + timings_sorted[n // 2]) / 2
            )
            p90_index = min(n - 1, int(round(0.9 * (n - 1))))
            p90_ms = timings_sorted[p90_index]

            results[batch_size] = {
                "median_ms_per_batch": median_ms,
                "p90_ms_per_batch": p90_ms,
                "ms_per_image": median_ms / batch_size,
                "device": describe_device(device),
            }

    return results
