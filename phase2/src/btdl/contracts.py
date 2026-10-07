"""Single source of truth for Phase 2 class semantics.

Values are fixed to match Phase 1's src/evaluation/metrics.py
(CLASS_ORDER = [1, 2, 3], CLASS_NAMES = {1: "meningioma", 2: "glioma",
3: "pituitary"}) so Phase 1 and Phase 2 results are directly comparable.
Any change here requires a CONTRACT_VERSION bump (see docs/DECISIONS.md).
"""

from types import MappingProxyType

import numpy as np

CONTRACT_VERSION = "1.0.0"

PROJECT_LABELS = (1, 2, 3)

CLASS_NAMES = MappingProxyType({1: "meningioma", 2: "glioma", 3: "pituitary"})

LABEL_TO_INDEX = MappingProxyType({1: 0, 2: 1, 3: 2})
INDEX_TO_LABEL = MappingProxyType({index: label for label, index in LABEL_TO_INDEX.items()})

NUM_CLASSES = 3

CLASS_ORDER_INDEX = (0, 1, 2)
CLASS_NAMES_BY_INDEX = tuple(CLASS_NAMES[INDEX_TO_LABEL[index]] for index in CLASS_ORDER_INDEX)


def _as_plain_int(value, name):
    """Coerce a Python int or numpy integer scalar to a plain int.

    Rejects bool/np.bool_, floats (including integral floats like 1.0),
    strings, None, and any other non-integral type.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got bool: {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError(
        f"{name} must be a Python int or numpy integer scalar, "
        f"got {type(value).__name__}: {value!r}"
    )


def label_to_index(label):
    label = _as_plain_int(label, "label")
    if label not in LABEL_TO_INDEX:
        raise ValueError(f"label must be one of {PROJECT_LABELS}, got {label!r}")
    return LABEL_TO_INDEX[label]


def index_to_label(index):
    index = _as_plain_int(index, "index")
    if index not in INDEX_TO_LABEL:
        raise ValueError(f"index must be one of {CLASS_ORDER_INDEX}, got {index!r}")
    return INDEX_TO_LABEL[index]
