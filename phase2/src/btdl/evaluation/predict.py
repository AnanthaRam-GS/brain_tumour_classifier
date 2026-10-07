"""Shared inference: eval mode, no_grad, softmax, batch order preserved.

Used by the trainer's validation loop and by the evaluation CLIs, so
forward-pass/softmax/bookkeeping logic exists in exactly one place.
"""

from dataclasses import dataclass

import numpy as np
import torch


class PredictError(ValueError):
    """Raised when a model's output does not have the expected [B, 3] shape."""


@dataclass(frozen=True)
class Predictions:
    sample_ids: tuple
    patient_ids: tuple
    y_true_idx: np.ndarray
    probs: np.ndarray


def predict(model, loader, device) -> Predictions:
    model.eval()
    sample_ids = []
    patient_ids = []
    target_parts = []
    probs_parts = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"]
            batch_size = images.shape[0]

            output = model(images)
            if not torch.is_tensor(output) or tuple(output.shape) != (batch_size, 3):
                got = type(output).__name__ if not torch.is_tensor(output) else tuple(output.shape)
                raise PredictError(
                    f"model output must be a torch.Tensor of shape [{batch_size}, 3], got {got}"
                )

            probs = torch.softmax(output, dim=1)
            target_parts.append(targets.detach().cpu())
            probs_parts.append(probs.detach().cpu())
            sample_ids.extend(batch["sample_id"])
            patient_ids.extend(batch["patient_id"])

    y_true_idx = torch.cat(target_parts).numpy()
    probs_cat = torch.cat(probs_parts).numpy()
    return Predictions(
        sample_ids=tuple(sample_ids),
        patient_ids=tuple(patient_ids),
        y_true_idx=y_true_idx,
        probs=probs_cat,
    )
