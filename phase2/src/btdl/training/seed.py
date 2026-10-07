"""Global reproducibility seeding (docs/DECISIONS.md D16)."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> dict:
    """Seeds python/numpy/torch(+cuda), enables deterministic algorithms.

    Returns a dict of the determinism settings actually in effect, meant to
    be recorded verbatim in run metadata.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cublas_workspace_config = None
    if torch.cuda.is_available():
        cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cublas_workspace_config is None:
            cublas_workspace_config = ":4096:8"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config

    torch.use_deterministic_algorithms(True, warn_only=True)

    return {
        "seed": seed,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "use_deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
        "cublas_workspace_config": cublas_workspace_config,
        "cuda_available": torch.cuda.is_available(),
    }
