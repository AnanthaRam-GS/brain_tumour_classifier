"""ResNet18: torchvision ImageNet-pretrained, full fine-tuning, 3-class head (D8).

No modification beyond the head swap: no layer is frozen (D8 -- full
fine-tuning), no dropout added."""

from torchvision.models import ResNet18_Weights, resnet18

from btdl.models.torchvision_common import build_torchvision_classifier


def build_resnet18(pretrained: bool = True):
    return build_torchvision_classifier(
        builder=resnet18,
        weights=ResNet18_Weights.IMAGENET1K_V1,
        head_path="fc",
        pretrained=pretrained,
    )
