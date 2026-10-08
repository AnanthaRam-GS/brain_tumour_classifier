"""Explicit model registry -- no import-time auto-discovery magic.

Adding a model = one new module (with a build_<name>(pretrained) -> nn.Module
function) + one MODEL_REGISTRY entry below.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn

from btdl.models.tiny_cnn import build_tiny_cnn


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: str
    builder: Callable[[bool], nn.Module]
    weights_id: Optional[str]
    reference_only: bool
    description: str


MODEL_REGISTRY: dict = {
    "tiny_cnn": ModelSpec(
        name="tiny_cnn",
        version="1.0.0",
        builder=build_tiny_cnn,
        weights_id=None,
        reference_only=True,
        description="Small 4-conv-block reference CNN; no pretrained weights; "
        "for exercising the training/evaluation pipeline end-to-end.",
    ),
}


def get_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; valid models are {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


def list_models() -> tuple:
    return tuple(sorted(MODEL_REGISTRY.keys()))


def build_model(name: str, *, pretrained: bool = True) -> nn.Module:
    spec = get_spec(name)
    return spec.builder(pretrained)
