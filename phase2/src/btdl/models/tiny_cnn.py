"""TinyCNN: a small reference model, not a competitive architecture.

4 conv blocks (Conv -> BatchNorm -> ReLU -> MaxPool), doubling channels
16->32->64->128, then adaptive average pool -> linear to 3 classes. No
pretrained weights. Used to exercise the training/evaluation pipeline
end-to-end before AlexNet/VGG16/GoogLeNet/ResNet18/EfficientNet-B0 land.
"""

import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_tiny_cnn(pretrained: bool = False) -> nn.Module:
    """pretrained is accepted for interface uniformity with build_model() but
    ignored -- TinyCNN has no pretrained weights (reference_only=True)."""

    return TinyCNN()
