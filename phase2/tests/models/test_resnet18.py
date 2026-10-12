"""ResNet18 model-specific verification, mirroring test_efficientnet_b0.py.
Assigned to Sanjay Anand M; implemented by the foundation owner on their
behalf (see TEAM_GUIDE.md section 2)."""

import pytest
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from btdl.models.registry import build_model, get_spec, list_models
from btdl.models.resnet18 import build_resnet18

# torchvision resnet18's stock (1000-class) total parameter count, with the
# 1000-class nn.Linear(512, 1000) fc head (513,000 params) replaced by
# nn.Linear(512, 3) (1,539 params): 11,689,512 - 513,000 + 1,539.
EXPECTED_TOTAL_PARAMETERS = 11_178_051


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_resnet18(pretrained=False)  # never downloads


def test_head_is_linear_512_to_3(model):
    head = model.fc
    assert isinstance(head, nn.Linear)
    assert head.in_features == 512
    assert head.out_features == 3


def test_every_parameter_is_trainable(model):
    # D8: full fine-tuning, no frozen backbone.
    assert all(p.requires_grad for p in model.parameters())
    assert sum(1 for _ in model.parameters()) > 0


def test_exact_total_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    assert total == EXPECTED_TOTAL_PARAMETERS


def test_output_shape_train_mode(model):
    model.train()
    x = torch.rand(2, 3, 224, 224)
    output = model(x)
    assert torch.is_tensor(output)
    assert tuple(output.shape) == (2, 3)


def test_output_shape_eval_mode(model):
    model.eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert torch.is_tensor(output)
    assert tuple(output.shape) == (2, 3)


def test_registered_in_catalog_with_correct_spec():
    assert "resnet18" in list_models()
    spec = get_spec("resnet18")
    assert spec.name == "resnet18"
    assert spec.version == "1.0.0"
    assert spec.weights_id == "ResNet18_Weights.IMAGENET1K_V1"
    assert spec.reference_only is False
    assert build_model("resnet18", pretrained=False) is not None


# ---- network: downloads real ImageNet weights ----------------------------------------------


@pytest.mark.network
@pytest.mark.slow
def test_pretrained_weights_load_exactly_and_only_the_head_changes():
    pretrained_model = build_resnet18(pretrained=True)
    reference = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    pretrained_state = pretrained_model.state_dict()
    reference_state = reference.state_dict()

    head_keys = {"fc.weight", "fc.bias"}
    non_head_keys = set(reference_state.keys()) - head_keys
    assert non_head_keys  # sanity: there are non-head keys to compare

    for key in non_head_keys:
        assert key in pretrained_state, f"missing key {key!r} in built model's state_dict"
        assert torch.equal(pretrained_state[key], reference_state[key]), (
            f"{key} differs from torchvision's reference pretrained state_dict -- "
            "pretrained weights were not loaded exactly, or something besides the head changed"
        )

    # Explicit check that BatchNorm running stats and num_batches_tracked
    # (easy to accidentally miss/reset) are included and match exactly.
    bn_keys = [k for k in non_head_keys if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k]
    assert bn_keys  # sanity: resnet18 has BatchNorm layers
    for key in bn_keys:
        assert torch.equal(pretrained_state[key], reference_state[key]), key

    head_weight = pretrained_state["fc.weight"]
    head_bias = pretrained_state["fc.bias"]
    assert tuple(head_weight.shape) == (3, 512)
    assert tuple(head_bias.shape) == (3,)
    # The new head must NOT equal the reference's stock 1000-class head
    # sliced/reused -- it's a freshly initialised nn.Linear(512, 3).
    assert head_weight.shape != reference_state["fc.weight"].shape
