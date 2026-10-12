"""VGG16: torchvision ImageNet-pretrained, full fine-tuning, 3-class head (D8)."""

from torchvision.models import VGG16_Weights, vgg16

from btdl.models.torchvision_common import build_torchvision_classifier


def build_vgg16(pretrained: bool = True):
    return build_torchvision_classifier(
        builder=vgg16,
        weights=VGG16_Weights.IMAGENET1K_V1,
        head_path="classifier.6",
        pretrained=pretrained,
    )
