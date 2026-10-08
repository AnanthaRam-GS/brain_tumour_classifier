"""Tests for the shared torchvision classifier helper (Part B, D8).

pretrained=False throughout -- no network access, no downloads. These
tests exercise ONLY build_torchvision_classifier(); none of these five
architectures are added to MODEL_REGISTRY here (that's teammate work,
out of scope for this prompt)."""

import pytest
import torch
import torch.nn as nn
from torchvision.models import alexnet, efficientnet_b0, googlenet, resnet18, vgg16

from btdl.contracts import NUM_CLASSES
from btdl.models.torchvision_common import TorchvisionHeadError, build_torchvision_classifier

ARCHITECTURES = [
    pytest.param("alexnet", alexnet, "classifier.6", {}, id="alexnet"),
    pytest.param("vgg16", vgg16, "classifier.6", {}, id="vgg16"),
    pytest.param("googlenet", googlenet, "fc", {"aux_logits": False}, id="googlenet"),
    pytest.param("resnet18", resnet18, "fc", {}, id="resnet18"),
    pytest.param("efficientnet_b0", efficientnet_b0, "classifier.1", {}, id="efficientnet_b0"),
]


@pytest.mark.parametrize("name, builder, head_path, builder_kwargs", ARCHITECTURES)
def test_head_replaced_and_output_shape(name, builder, head_path, builder_kwargs):
    model = build_torchvision_classifier(
        builder=builder, weights=None, head_path=head_path, pretrained=False, builder_kwargs=builder_kwargs
    )
    x = torch.rand(2, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        out = model(x)
    assert torch.is_tensor(out), f"{name}: output must be a single tensor, not {type(out)} (check aux_logits)"
    assert tuple(out.shape) == (2, NUM_CLASSES), f"{name}: output shape {tuple(out.shape)}"

    model.train()
    out_train = model(x)
    assert torch.is_tensor(out_train), f"{name} (train mode): output must be a single tensor, not {type(out_train)}"
    assert tuple(out_train.shape) == (2, NUM_CLASSES)


@pytest.mark.parametrize("name, builder, head_path, builder_kwargs", ARCHITECTURES)
def test_head_is_a_fresh_linear_layer(name, builder, head_path, builder_kwargs):
    model = build_torchvision_classifier(
        builder=builder, weights=None, head_path=head_path, pretrained=False, builder_kwargs=builder_kwargs
    )
    parent = model
    parts = head_path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    head = getattr(parent, parts[-1])
    assert isinstance(head, nn.Linear)
    assert head.out_features == NUM_CLASSES


def test_pretrained_false_means_weights_arg_is_none(monkeypatch):
    captured = {}

    def fake_builder(*, weights, **kwargs):
        captured["weights"] = weights
        return alexnet(weights=None)

    build_torchvision_classifier(
        builder=fake_builder, weights="sentinel-weights-enum", head_path="classifier.6", pretrained=False
    )
    assert captured["weights"] is None


def test_pretrained_true_passes_through_weights_member(monkeypatch):
    captured = {}

    def fake_builder(*, weights, **kwargs):
        captured["weights"] = weights
        return alexnet(weights=None)

    build_torchvision_classifier(
        builder=fake_builder, weights="sentinel-weights-enum", head_path="classifier.6", pretrained=True
    )
    assert captured["weights"] == "sentinel-weights-enum"


def test_builder_kwargs_forwarded():
    captured = {}

    def fake_builder(*, weights, **kwargs):
        captured.update(kwargs)
        return alexnet(weights=None)

    build_torchvision_classifier(
        builder=fake_builder,
        weights=None,
        head_path="classifier.6",
        pretrained=False,
        builder_kwargs={"some_kwarg": 42},
    )
    assert captured == {"some_kwarg": 42}


def test_wrong_head_path_not_linear_raises():
    with pytest.raises(TorchvisionHeadError, match="not nn.Linear"):
        build_torchvision_classifier(builder=alexnet, weights=None, head_path="features", pretrained=False)


def test_unresolvable_head_path_raises():
    with pytest.raises(TorchvisionHeadError, match="does not resolve"):
        build_torchvision_classifier(builder=alexnet, weights=None, head_path="not_a_real_attr", pretrained=False)


def test_only_the_head_layer_is_replaced():
    # Everything upstream of the head must be the stock torchvision module
    # (full fine-tuning, no frozen/removed layers -- D8).
    reference = alexnet(weights=None)
    model = build_torchvision_classifier(builder=alexnet, weights=None, head_path="classifier.6", pretrained=False)
    assert len(model.classifier) == len(reference.classifier)
    for i in range(len(reference.classifier) - 1):
        assert type(model.classifier[i]) is type(reference.classifier[i])
    assert type(model.features) is type(reference.features)
