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

    Eval mode, torch.inference_mode() (strictly faster than no_grad -- no
    autograd version-counter bookkeeping at all, appropriate here since
    nothing downstream ever needs gradients on these outputs), contiguous
    memory format (channels_last measured consistently WORSE on this CPU
    backend for a conv-heavy model -- see the Part A diagnosis), random
    3x224x224 input, with explicit device synchronization around each timed
    call (required for CUDA/MPS, whose kernel launches are asynchronous --
    without it the wall-clock timing would measure launch overhead, not
    actual execution). torch.use_deterministic_algorithms(True,
    warn_only=True) is enabled (same posture as training's seed_everything,
    D16) -- it does not alter eval-mode forward-pass numerics, only which
    kernel variant is used, so it does not affect what is being measured;
    both the determinism flag and the memory format actually in effect are
    recorded in each batch size's result so the measurement's conditions are
    never implicit.
    """

    evaluation_cfg = config.load_contract("evaluation")
    if batch_sizes is None:
        batch_sizes = evaluation_cfg["efficiency"]["batch_sizes"]
    if warmup is None:
        warmup = evaluation_cfg["efficiency"]["warmup"]
    if iters is None:
        iters = evaluation_cfg["efficiency"]["iters"]

    torch.use_deterministic_algorithms(True, warn_only=True)

    model = model.to(device, memory_format=torch.contiguous_format)
    model.eval()

    results = {}
    with torch.inference_mode():
        for batch_size in batch_sizes:
            inputs = torch.rand(batch_size, 3, 224, 224, device=device).contiguous()

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
                "memory_format": "contiguous",
                "deterministic_algorithms": True,
            }

    return results
