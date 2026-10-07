"""The ONLY place ImageNet normalization constants are applied (docs/DECISIONS.md D5)."""

import torch


def to_model_input(roi224, cfg) -> torch.Tensor:
    """[H, W] or [1, H, W] float32 -> [3, H, W] float32: replicate channel, then normalize."""

    tensor = roi224 if isinstance(roi224, torch.Tensor) else torch.as_tensor(roi224)
    tensor = tensor.to(dtype=torch.float32)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"roi224 must have shape [H, W] or [1, H, W], got {tuple(tensor.shape)}")

    channels = cfg["model_input"]["channels"]
    mean = torch.tensor(cfg["model_input"]["mean"], dtype=torch.float32).view(channels, 1, 1)
    std = torch.tensor(cfg["model_input"]["std"], dtype=torch.float32).view(channels, 1, 1)

    replicated = tensor.repeat(channels, 1, 1)
    return (replicated - mean) / std
