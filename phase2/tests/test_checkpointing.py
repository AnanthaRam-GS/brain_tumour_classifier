import torch
import torch.nn as nn
import pytest

from btdl import config
from btdl.training.checkpointing import (
    CheckpointError,
    build_run_metadata,
    load_checkpoint,
    save_best_checkpoint,
    save_last_checkpoint,
)
from btdl.training.device import describe_device, select_device
from btdl.training.early_stopping import EarlyStopping
from btdl.training.seed import seed_everything


def _toy_model():
    torch.manual_seed(0)
    return nn.Linear(10, 3)


def _toy_metadata(model, tmp_contract_hash=None, effective_cfg=None):
    device = select_device("cpu")
    settings = seed_everything(42)
    if effective_cfg is None:
        effective_cfg = config.load_contract("training")
    metadata = build_run_metadata(
        model=model,
        model_name="toy",
        model_version="v1",
        effective_cfg=effective_cfg,
        lr=1e-3,
        seed=42,
        best_epoch=0,
        best_val_macro_f1=0.5,
        best_val_loss=1.0,
        device_description=describe_device(device),
        determinism_settings=settings,
    )
    if tmp_contract_hash is not None:
        metadata = dict(metadata)
        metadata["contract_dir_sha256"] = tmp_contract_hash
    return metadata


def test_best_checkpoint_round_trip_identical_state_dict(tmp_path):
    model = _toy_model()
    metadata = _toy_metadata(model)
    save_best_checkpoint(tmp_path, model=model, metadata=metadata)

    model2 = nn.Linear(10, 3)
    load_checkpoint(tmp_path / "best.pt", model2, device="cpu")

    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2)


def test_reload_gives_identical_predictions(tmp_path):
    model = _toy_model()
    metadata = _toy_metadata(model)
    save_best_checkpoint(tmp_path, model=model, metadata=metadata)

    model2 = nn.Linear(10, 3)
    load_checkpoint(tmp_path / "best.pt", model2, device="cpu")

    x = torch.randn(5, 10)
    model.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    assert torch.equal(out1, out2)


def test_weights_only_load_works(tmp_path):
    model = _toy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    es = EarlyStopping(patience=3)
    es.step(0.5, 1.0, 0)
    metadata = _toy_metadata(model)

    save_last_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=es,
        epoch=0,
        history=[{"epoch": 0, "val_loss": 1.0}],
        metadata=metadata,
    )

    # torch.load(..., weights_only=True) is used internally by load_checkpoint;
    # a successful call here (no UnpicklingError) is the assertion.
    model2 = nn.Linear(10, 3)
    payload = load_checkpoint(tmp_path / "last.pt", model2, device="cpu")
    assert payload["epoch"] == 0
    assert payload["history"] == [{"epoch": 0, "val_loss": 1.0}]
    assert "optimizer_state_dict" in payload
    assert "scheduler_state_dict" in payload
    assert "early_stopping_state" in payload


def test_metadata_mirrored_to_json(tmp_path):
    model = _toy_model()
    metadata = _toy_metadata(model)
    save_best_checkpoint(tmp_path, model=model, metadata=metadata)
    assert (tmp_path / "metadata.json").is_file()

    import json

    with (tmp_path / "metadata.json").open() as handle:
        loaded = json.load(handle)
    assert loaded["model_name"] == "toy"
    assert loaded["seed"] == 42


def test_atomic_write_leaves_no_partial_checkpoint_on_failure(tmp_path, monkeypatch):
    model = _toy_model()
    metadata = _toy_metadata(model)

    # Simulate a failure mid-write: torch.save raises after the temp file
    # would have been partially written.
    original_save = torch.save

    def _failing_save(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(torch, "save", _failing_save)
    with pytest.raises(RuntimeError):
        save_best_checkpoint(tmp_path, model=model, metadata=metadata)
    monkeypatch.setattr(torch, "save", original_save)

    assert not (tmp_path / "best.pt").is_file()
    # No leftover temp file either.
    assert list(tmp_path.glob(".*best.pt.tmp")) == []


def test_contract_mismatch_refused_by_default(tmp_path):
    model = _toy_model()
    metadata = _toy_metadata(model, tmp_contract_hash="deadbeef")
    save_best_checkpoint(tmp_path, model=model, metadata=metadata)

    model2 = nn.Linear(10, 3)
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "best.pt", model2, device="cpu")


def test_contract_mismatch_allowed_with_strict_contract_false(tmp_path):
    model = _toy_model()
    metadata = _toy_metadata(model, tmp_contract_hash="deadbeef")
    save_best_checkpoint(tmp_path, model=model, metadata=metadata)

    model2 = nn.Linear(10, 3)
    payload = load_checkpoint(tmp_path / "best.pt", model2, device="cpu", strict_contract=False)
    assert payload["metadata"]["contract_dir_sha256"] == "deadbeef"


def test_build_run_metadata_contains_required_fields():
    model = _toy_model()
    metadata = _toy_metadata(model)
    required = {
        "model_name",
        "model_version",
        "total_parameters",
        "trainable_parameters",
        "contract_version",
        "contract_dir_sha256",
        "effective_cfg",
        "contract_conformant",
        "contract_deviations",
        "manifest_sha256",
        "split_sha256",
        "roi_cache_npy_sha256",
        "augmentation_yaml_sha256",
        "training_yaml_sha256",
        "lr",
        "weight_decay",
        "batch_size",
        "seed",
        "best_epoch",
        "best_val_macro_f1",
        "best_val_loss",
        "torch_version",
        "torchvision_version",
        "device",
        "determinism",
        "git_commit",
        "git_dirty",
        "git_dirty_paths",
        "roi_cache_path",
        "weights_id",
        "pretrained_weights_loaded",
        "python_version",
        "platform",
    }
    assert required.issubset(metadata.keys())
    assert metadata["total_parameters"] == sum(p.numel() for p in model.parameters())


def test_metadata_conformant_when_effective_cfg_matches_contract():
    model = _toy_model()
    metadata = _toy_metadata(model)  # default: effective_cfg = the real committed contract
    assert metadata["contract_conformant"] is True
    assert metadata["contract_deviations"] == []


def test_metadata_non_conformant_when_effective_cfg_deviates():
    model = _toy_model()
    contract_cfg = config.load_contract("training")
    deviating_cfg = {
        "batch_size": 999,  # deviates from the committed contract
        "max_epochs": contract_cfg["max_epochs"],
        "optimizer": dict(contract_cfg["optimizer"]),
        "scheduler": dict(contract_cfg["scheduler"]),
        "early_stopping": dict(contract_cfg["early_stopping"]),
    }
    metadata = _toy_metadata(model, effective_cfg=deviating_cfg)
    assert metadata["contract_conformant"] is False
    deviation_keys = {d["key"] for d in metadata["contract_deviations"]}
    assert "batch_size" in deviation_keys
    batch_size_deviation = next(d for d in metadata["contract_deviations"] if d["key"] == "batch_size")
    assert batch_size_deviation["contract_value"] == contract_cfg["batch_size"]
    assert batch_size_deviation["effective_value"] == 999


def test_load_checkpoint_moves_model_to_requested_device(tmp_path):
    # Regression test: load_checkpoint() must move `model` to `device` itself
    # -- load_state_dict's copy_ leaves params on their PRIOR device
    # regardless of the checkpoint's own device, so a model freshly built on
    # CPU (e.g. via build_model()) stayed on CPU even when device="mps",
    # causing a real cross-device RuntimeError in cli/evaluate.py the first
    # time it ran on real MPS hardware. Exercised here on CPU (every target
    # device, including "cpu", must satisfy the same postcondition).
    model = _toy_model()
    metadata = _toy_metadata(model)
    save_best_checkpoint(tmp_path, model=model, metadata=metadata)

    model2 = nn.Linear(10, 3)
    load_checkpoint(tmp_path / "best.pt", model2, device="cpu")
    assert next(model2.parameters()).device == torch.device("cpu")


def test_metadata_records_effective_cfg_not_contract_values():
    model = _toy_model()
    contract_cfg = config.load_contract("training")
    deviating_cfg = {
        "batch_size": 4,
        "max_epochs": contract_cfg["max_epochs"],
        "optimizer": {
            "weight_decay": 0.5,  # deviates from contract's weight_decay
            "betas": list(contract_cfg["optimizer"]["betas"]),
            "eps": contract_cfg["optimizer"]["eps"],
        },
        "scheduler": dict(contract_cfg["scheduler"]),
        "early_stopping": dict(contract_cfg["early_stopping"]),
    }
    metadata = _toy_metadata(model, effective_cfg=deviating_cfg)
    # weight_decay/batch_size in metadata reflect what was ACTUALLY used, never
    # silently substituted with the committed contract's values.
    assert metadata["batch_size"] == 4
    assert metadata["weight_decay"] == 0.5
    assert metadata["effective_cfg"]["batch_size"] == 4
