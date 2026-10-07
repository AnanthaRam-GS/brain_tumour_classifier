"""Single source of truth for Phase 2 class semantics.

Values are fixed to match Phase 1's src/evaluation/metrics.py
(CLASS_ORDER = [1, 2, 3], CLASS_NAMES = {1: "meningioma", 2: "glioma",
3: "pituitary"}) so Phase 1 and Phase 2 results are directly comparable.
Any change here requires a CONTRACT_VERSION bump (see docs/DECISIONS.md).
"""

from types import MappingProxyType

CONTRACT_VERSION = "1.0.0"

PROJECT_LABELS = (1, 2, 3)

CLASS_NAMES = MappingProxyType({1: "meningioma", 2: "glioma", 3: "pituitary"})

LABEL_TO_INDEX = MappingProxyType({1: 0, 2: 1, 3: 2})
INDEX_TO_LABEL = MappingProxyType({index: label for label, index in LABEL_TO_INDEX.items()})

NUM_CLASSES = 3

CLASS_ORDER_INDEX = (0, 1, 2)
CLASS_NAMES_BY_INDEX = tuple(CLASS_NAMES[INDEX_TO_LABEL[index]] for index in CLASS_ORDER_INDEX)


def label_to_index(label):
    if isinstance(label, bool) or not isinstance(label, int):
        raise ValueError(f"label must be a plain int, got {type(label).__name__}: {label!r}")
    if label not in LABEL_TO_INDEX:
        raise ValueError(f"label must be one of {PROJECT_LABELS}, got {label!r}")
    return LABEL_TO_INDEX[label]


def index_to_label(index):
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"index must be a plain int, got {type(index).__name__}: {index!r}")
    if index not in INDEX_TO_LABEL:
        raise ValueError(f"index must be one of {CLASS_ORDER_INDEX}, got {index!r}")
    return INDEX_TO_LABEL[index]
