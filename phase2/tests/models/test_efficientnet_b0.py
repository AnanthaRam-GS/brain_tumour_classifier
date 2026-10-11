"""EfficientNet-B0 model-specific verification (TEAM_GUIDE section 2's
"optional tests/models/test_<your_model>.py"). tests/test_model_contract.py
already covers every registered model generically; this file checks
EfficientNet-B0-specific facts: the exact parameter count, that the head
is nn.Linear(1280, 3), that every parameter is trainable (D8: full
fine-tuning, nothing frozen), and -- offline-unsafe, hence @pytest.mark.
network -- that pretrained=True loads torchvision's real ImageNet weights
into every non-head tensor exactly, unmodified."""

import pytest
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from btdl.models.efficientnet_b0 import build_efficientnet_b0
from btdl.models.registry import build_model, get_spec, list_models

# torchvision efficientnet_b0's stock (1000-class) total parameter count,
# with the 1000-class nn.Linear(1280, 1000) head (1,281,000 params)
# replaced by nn.Linear(1280, 3) (3,843 params): 5,288,548 - 1,281,000 + 3,843.
EXPECTED_TOTAL_PARAMETERS = 4_011_391


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_efficientnet_b0(pretrained=False)  # never downloads


def test_head_is_linear_1280_to_3(model):
    head = model.classifier[1]
    assert isinstance(head, nn.Linear)
    assert head.in_features == 1280
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
    assert "efficientnet_b0" in list_models()
    spec = get_spec("efficientnet_b0")
    assert spec.name == "efficientnet_b0"
    assert spec.version == "1.0.0"
    assert spec.weights_id == "EfficientNet_B0_Weights.IMAGENET1K_V1"
    assert spec.reference_only is False
    assert build_model("efficientnet_b0", pretrained=False) is not None


# ---- network: downloads real ImageNet weights ----------------------------------------------


@pytest.mark.network
@pytest.mark.slow
def test_pretrained_weights_load_exactly_and_only_the_head_changes():
    pretrained_model = build_efficientnet_b0(pretrained=True)
    reference = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    pretrained_state = pretrained_model.state_dict()
    reference_state = reference.state_dict()

    head_keys = {"classifier.1.weight", "classifier.1.bias"}
    non_head_keys = set(reference_state.keys()) - head_keys
    assert non_head_keys  # sanity: there are non-head keys to compare

    for key in non_head_keys:
        assert key in pretrained_state, f"missing key {key!r} in built model's state_dict"
        assert torch.equal(pretrained_state[key], reference_state[key]), (
            f"{key} differs from torchvision's reference pretrained state_dict -- "
            "pretrained weights were not loaded exactly, or something besides the head changed"
        )

    head_weight = pretrained_state["classifier.1.weight"]
    head_bias = pretrained_state["classifier.1.bias"]
    assert tuple(head_weight.shape) == (3, 1280)
    assert tuple(head_bias.shape) == (3,)
    # The new head must NOT equal the reference's stock 1000-class head
    # sliced/reused -- it's a freshly initialised nn.Linear(1280, 3).
    assert head_weight.shape != reference_state["classifier.1.weight"].shape
