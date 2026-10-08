"""Parametrized over EVERY registered model -- any model added later is
automatically covered by this file, no changes needed here."""

import pytest
import torch

from btdl import config
from btdl.models.registry import MODEL_REGISTRY, build_model, get_spec, list_models

ALL_MODEL_NAMES = list_models()


@pytest.fixture(params=ALL_MODEL_NAMES)
def model_name(request):
    return request.param


@pytest.fixture
def model(model_name):
    torch.manual_seed(0)
    return build_model(model_name, pretrained=False)  # never downloads


def test_output_shape_in_train_mode(model):
    model.train()
    x = torch.rand(2, 3, 224, 224)
    output = model(x)
    assert torch.is_tensor(output)
    assert tuple(output.shape) == (2, 3)


def test_output_shape_in_eval_mode(model):
    model.eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert torch.is_tensor(output)
    assert tuple(output.shape) == (2, 3)


def test_output_is_not_tuple_or_aux(model):
    model.eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert not isinstance(output, (tuple, list))


def test_all_parameters_fp32(model):
    for param in model.parameters():
        assert param.dtype == torch.float32


def test_eval_forward_is_deterministic(model):
    model.eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2)


def test_trainable_parameter_count_positive(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0


def test_yaml_matches_registry_spec(model_name):
    spec = get_spec(model_name)
    yaml_config = config.load_model_config(model_name)
    assert yaml_config["name"] == spec.name
    assert yaml_config["version"] == spec.version
    assert yaml_config["weights_id"] == spec.weights_id
    assert yaml_config["reference_only"] == spec.reference_only
    assert yaml_config["description"] == spec.description


def test_registry_is_an_explicit_dict_not_auto_discovered():
    assert isinstance(MODEL_REGISTRY, dict)
    assert len(MODEL_REGISTRY) >= 1


def test_unknown_model_raises_and_lists_valid_names():
    with pytest.raises(ValueError, match="tiny_cnn"):
        build_model("definitely_not_a_real_model")
