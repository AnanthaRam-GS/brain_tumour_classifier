"""Explicit model registry machinery -- no import-time auto-discovery magic.

MODEL_REGISTRY itself lives in models/catalog.py (NOT locked -- see
artifacts/contract/foundation_lock.json / cli/lock.py), so a teammate can
register a new architecture without touching this (locked) file. Adding a
model = one new module (with a build_<name>(pretrained) -> nn.Module
function) + one MODEL_REGISTRY entry in catalog.py.

Re-exports ModelSpec and MODEL_REGISTRY so `from btdl.models.registry
import ...` remains every caller's (and every test's) single entry point --
the catalog.py/model_spec.py split underneath it is an implementation
detail.
"""

from btdl.models.catalog import MODEL_REGISTRY
from btdl.models.model_spec import ModelSpec

__all__ = ["ModelSpec", "MODEL_REGISTRY", "get_spec", "list_models", "build_model"]


def get_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; valid models are {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


def list_models() -> tuple:
    return tuple(sorted(MODEL_REGISTRY.keys()))


def build_model(name: str, *, pretrained: bool = True):
    spec = get_spec(name)
    return spec.builder(pretrained)
