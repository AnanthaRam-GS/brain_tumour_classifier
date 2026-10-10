"""Model catalog: MODEL_REGISTRY, the single dict of registered models.

NOT part of the locked foundation (see artifacts/contract/foundation_lock.json
and cli/lock.py) -- a teammate adds a model by adding one new
models/<name>.py module (with a build_<name>(pretrained) -> nn.Module
function) and one entry below. See docs/TEAM_GUIDE.md. Everything that
CONSUMES this dict (get_spec/list_models/build_model) lives in the locked
models/registry.py, so adding an entry here can never silently change that
machinery.
"""

from btdl.models.efficientnet_b0 import build_efficientnet_b0
from btdl.models.model_spec import ModelSpec
from btdl.models.tiny_cnn import build_tiny_cnn

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
    "efficientnet_b0": ModelSpec(
        name="efficientnet_b0",
        version="1.0.0",
        builder=build_efficientnet_b0,
        weights_id="EfficientNet_B0_Weights.IMAGENET1K_V1",
        reference_only=False,
        description="torchvision EfficientNet-B0, ImageNet-pretrained, full fine-tuning, "
        "3-class head (D8).",
    ),
}
