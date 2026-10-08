"""Shared torchvision classifier builder (implements D8 once).

Every per-architecture model module (e.g. models/resnet18.py) is a thin
wrapper around build_torchvision_classifier(): it supplies the torchvision
constructor, the ImageNet weights enum member (or None), the dotted path to
the final classifier layer, and any constructor kwargs the architecture
needs -- nothing else. The helper does exactly two things: build the
torchvision model (with or without pretrained weights), and replace its
final nn.Linear layer with a fresh nn.Linear(in_features, 3), initialised
under whatever torch seed is current when it's called (default nn.Linear
init -- no custom init scheme). Nothing else about the model is touched:
full fine-tuning (D8), no frozen backbone, no layers removed or added
beyond the head swap.

head_path is a dotted attribute/index path resolved via getattr/setattr,
e.g. "classifier.6" (an nn.Sequential's 7th child -- torchvision's
Sequential stores children under string-integer attribute names, so
getattr/setattr on "6" works exactly like named attributes) or "fc" (a
plain top-level attribute). Verified empirically (not guessed) against
each architecture's module tree, per architecture:

  alexnet:         head_path="classifier.6"
  vgg16:            head_path="classifier.6"
  googlenet:        head_path="fc", builder_kwargs={"aux_logits": False}
  resnet18:         head_path="fc"
  efficientnet_b0:  head_path="classifier.1"

GoogLeNet note (D8): keeps its shipped transform_input exactly as
torchvision ships it (not touched here); aux_logits=False is passed via
builder_kwargs so GoogLeNet's forward() returns a single [B, 3] tensor like
every other architecture, instead of a GoogLeNetOutputs tuple with
auxiliary logits -- removing its loss-shape asymmetry relative to the other
four models.
"""

from typing import Callable, Optional

import torch.nn as nn

from btdl.contracts import NUM_CLASSES


class TorchvisionHeadError(ValueError):
    """Raised when head_path does not resolve to an nn.Linear layer."""


def _resolve_head(model: nn.Module, head_path: str):
    parts = head_path.split(".")
    parent = model
    for part in parts[:-1]:
        if not hasattr(parent, part):
            raise TorchvisionHeadError(
                f"head_path {head_path!r} does not resolve on {type(model).__name__}: "
                f"no attribute {part!r} on {type(parent).__name__}"
            )
        parent = getattr(parent, part)

    leaf_name = parts[-1]
    if not hasattr(parent, leaf_name):
        raise TorchvisionHeadError(
            f"head_path {head_path!r} does not resolve on {type(model).__name__}: "
            f"no attribute {leaf_name!r} on {type(parent).__name__}"
        )
    return parent, leaf_name, getattr(parent, leaf_name)


def build_torchvision_classifier(
    *,
    builder: Callable,
    weights: Optional[object],
    head_path: str,
    pretrained: bool,
    builder_kwargs: dict = None,
) -> nn.Module:
    """Builds `builder(weights=weights if pretrained else None, **builder_kwargs)`
    and replaces the nn.Linear at `head_path` with nn.Linear(in_features, 3).

    Raises TorchvisionHeadError if head_path does not resolve to an
    nn.Linear (a wrong path for a given architecture is a programming
    error, not a runtime condition to silently tolerate).
    """

    builder_kwargs = dict(builder_kwargs) if builder_kwargs else {}
    weights_arg = weights if pretrained else None
    model = builder(weights=weights_arg, **builder_kwargs)

    parent, leaf_name, leaf = _resolve_head(model, head_path)
    if not isinstance(leaf, nn.Linear):
        raise TorchvisionHeadError(
            f"head_path {head_path!r} resolved to a {type(leaf).__name__}, not nn.Linear "
            f"-- pass the dotted path to the architecture's final classifier layer"
        )

    new_head = nn.Linear(leaf.in_features, NUM_CLASSES)
    setattr(parent, leaf_name, new_head)
    return model
