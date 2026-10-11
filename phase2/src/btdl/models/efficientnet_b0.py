"""EfficientNet-B0: torchvision ImageNet-pretrained, full fine-tuning, 3-class head (D8).

No modification beyond the head swap: torchvision's default dropout
(p=0.2) and stochastic depth are left exactly as shipped, and no layer is
frozen (D8 -- full fine-tuning on this small dataset)."""

from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from btdl.models.torchvision_common import build_torchvision_classifier


def build_efficientnet_b0(pretrained: bool = True):
    return build_torchvision_classifier(
        builder=efficientnet_b0,
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1,
        head_path="classifier.1",
        pretrained=pretrained,
    )
