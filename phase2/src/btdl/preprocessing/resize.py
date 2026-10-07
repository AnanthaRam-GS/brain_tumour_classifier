"""Deterministic ROI resize to the Phase 2 model input size (docs/DECISIONS.md D5).

Deliberately different from Phase 1's 128x128 skimage resize (order=1,
anti_aliasing=False): Phase 2 resizes to 224x224 with torch's bilinear,
antialiased interpolation on CPU float32, per the frozen input contract.
No parity with Phase 1 is claimed or required for this step.
"""

import numpy as np
import torch
import torch.nn.functional as F


def resize_roi(roi, cfg) -> np.ndarray:
    """Resize a 2D float32 ROI crop to cfg["resize"]["size"], clamped to cfg["resize"]["clamp"]."""

    roi_array = np.asarray(roi, dtype=np.float32)
    if roi_array.ndim != 2:
        raise ValueError(f"roi must be 2D, got shape {roi_array.shape}")

    size = tuple(int(s) for s in cfg["resize"]["size"])
    mode = cfg["resize"]["mode"]
    antialias = cfg["resize"]["antialias"]
    align_corners = cfg["resize"]["align_corners"]
    low, high = cfg["resize"]["clamp"]

    tensor = torch.from_numpy(roi_array).to(dtype=torch.float32, device="cpu")
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    resized = F.interpolate(
        tensor,
        size=size,
        mode=mode,
        align_corners=align_corners,
        antialias=antialias,
    )
    resized = torch.clamp(resized, float(low), float(high))
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)
