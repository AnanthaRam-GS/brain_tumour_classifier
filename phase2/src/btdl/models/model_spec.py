"""ModelSpec: the registry entry type.

Kept in its own module (rather than inside registry.py or catalog.py)
purely to avoid a registry.py <-> catalog.py import cycle: registry.py's
machinery pulls MODEL_REGISTRY from catalog.py, and catalog.py's entries
need ModelSpec -- both import it from here instead of from each other.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: str
    builder: Callable[[bool], nn.Module]
    weights_id: Optional[str]
    reference_only: bool
    description: str
