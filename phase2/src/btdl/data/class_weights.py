"""Train-only inverse-frequency class weights (docs/DECISIONS.md D9)."""

import torch

from btdl.contracts import NUM_CLASSES, label_to_index


def compute_class_weights(dataset) -> torch.Tensor:
    """w_c = n / (K * n_c), ordered by internal class index. TRAIN split only."""

    if dataset.split != "train":
        raise ValueError(f"compute_class_weights requires dataset.split == 'train', got {dataset.split!r}")

    counts_by_label = dataset.class_counts
    n = sum(counts_by_label.values())

    weights = [0.0] * NUM_CLASSES
    for label, count in counts_by_label.items():
        index = label_to_index(label)
        if count == 0:
            raise ValueError(f"class with label {label} has zero samples in the train split")
        weights[index] = n / (NUM_CLASSES * count)

    return torch.tensor(weights, dtype=torch.float32)
