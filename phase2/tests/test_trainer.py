import math

import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from btdl.data.sampler import EpochSampler
from btdl.training.device import select_device
from btdl.training.trainer import FitResult, TrainerError, fit
from tests.toy_training import ToyDataset, ToyModel, TupleOutputModel, make_training_cfg

DEVICE = select_device("cpu")


def _build_loaders(*, train_n=30, val_n=12, batch_size=8, seed=42, train_data_seed=42, val_data_seed=43):
    train_ds = ToyDataset(train_n, augment=True, seed=train_data_seed)
    val_ds = ToyDataset(val_n, augment=False, seed=val_data_seed)
    train_sampler = EpochSampler(len(train_ds), shuffle=True, seed=seed, epoch_aware=True)
    val_sampler = EpochSampler(len(val_ds), shuffle=False, seed=seed, epoch_aware=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler)
    return train_loader, train_sampler, val_loader


def test_loss_decreases_and_val_macro_f1_rises(tmp_path):
    torch.manual_seed(0)
    model = ToyModel()
    train_loader, train_sampler, val_loader = _build_loaders()
    cfg = make_training_cfg(max_epochs=40)

    result = fit(
        model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        class_weights=torch.ones(3),
        cfg=cfg,
        lr=5e-2,
        seed=42,
        run_dir=tmp_path / "run",
        device=DEVICE,
    )

    history = pd.read_csv(result.history_path)
    assert history["train_aug_loss"].iloc[-1] < history["train_aug_loss"].iloc[0]
    assert history["val_macro_f1"].iloc[-1] > history["val_macro_f1"].iloc[0]


def test_parameters_change_after_first_step(tmp_path):
    torch.manual_seed(0)
    model = ToyModel()
    before = {name: param.clone() for name, param in model.named_parameters()}

    train_loader, train_sampler, val_loader = _build_loaders()
    cfg = make_training_cfg(max_epochs=1)
    fit(
        model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        class_weights=torch.ones(3),
        cfg=cfg,
        lr=5e-2,
        seed=42,
        run_dir=tmp_path / "run",
        device=DEVICE,
    )

    changed = any(not torch.equal(before[name], param) for name, param in model.named_parameters())
    assert changed


def test_lr_follows_cosine_schedule(tmp_path):
    torch.manual_seed(0)
    model = ToyModel()
    train_loader, train_sampler, val_loader = _build_loaders()
    max_epochs = 10
    cfg = make_training_cfg(max_epochs=max_epochs)
    lr = 5e-2

    result = fit(
        model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        class_weights=torch.ones(3),
        cfg=cfg,
        lr=lr,
        seed=42,
        run_dir=tmp_path / "run",
        device=DEVICE,
    )
    history = pd.read_csv(result.history_path)

    eta_min = cfg["scheduler"]["eta_min"]
    for epoch, recorded_lr in zip(history["epoch"], history["lr"]):
        expected_lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / max_epochs)) / 2
        assert recorded_lr == pytest.approx(expected_lr, rel=1e-5)


def test_loss_equals_manual_weighted_ce_on_one_batch(tmp_path):
    torch.manual_seed(0)
    model = ToyModel()

    # Single batch covering the whole (tiny) train set.
    train_loader, train_sampler, val_loader = _build_loaders(train_n=8, val_n=4, batch_size=8)
    class_weights = torch.tensor([1.5, 0.8, 1.2])

    batch = next(iter(train_loader))
    with torch.no_grad():
        manual_output = model(batch["image"])
        manual_loss = nn.CrossEntropyLoss(weight=class_weights)(manual_output, batch["target"])

    cfg = make_training_cfg(max_epochs=1)
    result = fit(
        model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        class_weights=class_weights,
        cfg=cfg,
        lr=1e-2,
        seed=42,
        run_dir=tmp_path / "run",
        device=DEVICE,
    )
    history = pd.read_csv(result.history_path)
    assert history["train_aug_loss"].iloc[0] == pytest.approx(manual_loss.item(), abs=1e-5)


def test_early_stopping_halts_and_best_matches_history_argmax(tmp_path):
    torch.manual_seed(0)
    model = ToyModel()
    train_loader, train_sampler, val_loader = _build_loaders()
    cfg = make_training_cfg(max_epochs=40, early_stopping={"patience": 3, "min_delta": 0.0, "mode": "max"})

    result = fit(
        model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        class_weights=torch.ones(3),
        cfg=cfg,
        lr=5e-2,
        seed=42,
        run_dir=tmp_path / "run",
        device=DEVICE,
    )

    assert result.stop_reason == "early_stopping"
    assert result.epochs_run < 40

    history = pd.read_csv(result.history_path)
    # Independently recompute the best epoch: highest val_macro_f1, ties
    # broken by lowest val_loss -- matching EarlyStopping's own rule.
    best_row = None
    for _, row in history.iterrows():
        if best_row is None:
            best_row = row
        elif row["val_macro_f1"] > best_row["val_macro_f1"] + 1e-12:
            best_row = row
        elif abs(row["val_macro_f1"] - best_row["val_macro_f1"]) <= 1e-12 and row["val_loss"] < best_row["val_loss"]:
            best_row = row

    assert result.best_epoch == int(best_row["epoch"])
    assert result.best_val_macro_f1 == pytest.approx(best_row["val_macro_f1"])


def test_same_seed_gives_identical_history_and_weights(tmp_path):
    def _run(run_dir):
        torch.manual_seed(0)
        model = ToyModel()
        train_loader, train_sampler, val_loader = _build_loaders()
        cfg = make_training_cfg(max_epochs=5)
        result = fit(
            model,
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            class_weights=torch.ones(3),
            cfg=cfg,
            lr=5e-2,
            seed=42,
            run_dir=run_dir,
            device=DEVICE,
        )
        return model, result

    model_a, result_a = _run(tmp_path / "run_a")
    model_b, result_b = _run(tmp_path / "run_b")

    history_a = pd.read_csv(result_a.history_path)
    history_b = pd.read_csv(result_b.history_path)
    # epoch_seconds is wall-clock timing, not expected to be bit-identical
    # across runs -- everything else must be.
    non_timing_columns = [c for c in history_a.columns if c != "epoch_seconds"]
    pd.testing.assert_frame_equal(history_a[non_timing_columns], history_b[non_timing_columns])

    for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert name_a == name_b
        assert torch.equal(p_a, p_b)


def test_resume_equals_uninterrupted_run(tmp_path):
    def _fresh_model():
        torch.manual_seed(0)
        return ToyModel()

    # Uninterrupted 8-epoch run.
    model_full = _fresh_model()
    train_loader, train_sampler, val_loader = _build_loaders()
    cfg_full = make_training_cfg(max_epochs=8)
    fit(
        model_full,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        class_weights=torch.ones(3),
        cfg=cfg_full,
        lr=5e-2,
        seed=42,
        run_dir=tmp_path / "full",
        device=DEVICE,
    )

    # Split: train to epoch 4 (4 epochs: 0..3), then resume to epoch 8.
    # Both calls use the SAME cfg (max_epochs=8): stop_after_epoch is what
    # pauses the first call, not a smaller cfg["max_epochs"] -- the cosine
    # schedule's T_max must stay fixed at the true 8-epoch horizon across
    # both calls, or the two phases follow different schedules.
    model_part = _fresh_model()
    train_loader2, train_sampler2, val_loader2 = _build_loaders()
    cfg_part1 = make_training_cfg(max_epochs=8)
    result_part1 = fit(
        model_part,
        train_loader=train_loader2,
        train_sampler=train_sampler2,
        val_loader=val_loader2,
        class_weights=torch.ones(3),
        cfg=cfg_part1,
        lr=5e-2,
        seed=42,
        run_dir=tmp_path / "part",
        device=DEVICE,
        stop_after_epoch=3,
    )
    assert result_part1.stop_reason == "stop_after_epoch"
    assert result_part1.epochs_run == 4

    model_resumed = ToyModel()  # fresh, uninitialized -- overwritten by resume
    train_loader3, train_sampler3, val_loader3 = _build_loaders()
    cfg_part2 = make_training_cfg(max_epochs=8)
    result_part2 = fit(
        model_resumed,
        train_loader=train_loader3,
        train_sampler=train_sampler3,
        val_loader=val_loader3,
        class_weights=torch.ones(3),
        cfg=cfg_part2,
        lr=5e-2,
        seed=42,
        run_dir=tmp_path / "part",
        device=DEVICE,
        resume=True,
    )

    history_full = pd.read_csv(tmp_path / "full" / "history.csv")
    history_resumed = pd.read_csv(tmp_path / "part" / "history.csv")
    non_timing_columns = [c for c in history_full.columns if c != "epoch_seconds"]
    pd.testing.assert_frame_equal(
        history_full[non_timing_columns], history_resumed[non_timing_columns]
    )

    for (name_a, p_a), (name_b, p_b) in zip(model_full.named_parameters(), model_resumed.named_parameters()):
        assert name_a == name_b
        assert torch.equal(p_a, p_b)


def test_tuple_output_model_raises_clear_error(tmp_path):
    torch.manual_seed(0)
    model = TupleOutputModel()
    train_loader, train_sampler, val_loader = _build_loaders()
    cfg = make_training_cfg(max_epochs=1)

    with pytest.raises(TrainerError):
        fit(
            model,
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            class_weights=torch.ones(3),
            cfg=cfg,
            lr=1e-2,
            seed=42,
            run_dir=tmp_path / "run",
            device=DEVICE,
        )


def test_injected_nan_loss_raises_with_epoch_and_batch_info(tmp_path):
    torch.manual_seed(0)
    model = ToyModel()
    with torch.no_grad():
        model.fc.weight.fill_(float("nan"))

    train_loader, train_sampler, val_loader = _build_loaders()
    cfg = make_training_cfg(max_epochs=1)

    with pytest.raises(TrainerError) as exc_info:
        fit(
            model,
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            class_weights=torch.ones(3),
            cfg=cfg,
            lr=1e-2,
            seed=42,
            run_dir=tmp_path / "run",
            device=DEVICE,
        )
    message = str(exc_info.value)
    assert "epoch" in message.lower()
    assert "batch" in message.lower()


def _run_and_pause(tmp_path, *, lr=5e-2, seed=42, cfg=None):
    torch.manual_seed(0)
    model = ToyModel()
    train_loader, train_sampler, val_loader = _build_loaders()
    cfg = cfg or make_training_cfg(max_epochs=8)
    fit(
        model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        class_weights=torch.ones(3),
        cfg=cfg,
        lr=lr,
        seed=seed,
        run_dir=tmp_path / "run",
        device=DEVICE,
        stop_after_epoch=1,
    )


def test_resume_refuses_mismatched_lr(tmp_path):
    _run_and_pause(tmp_path, lr=5e-2)
    model2 = ToyModel()
    train_loader, train_sampler, val_loader = _build_loaders()
    with pytest.raises(TrainerError, match="lr"):
        fit(
            model2,
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            class_weights=torch.ones(3),
            cfg=make_training_cfg(max_epochs=8),
            lr=1e-2,  # different lr
            seed=42,
            run_dir=tmp_path / "run",
            device=DEVICE,
            resume=True,
        )


def test_resume_refuses_mismatched_seed(tmp_path):
    _run_and_pause(tmp_path, seed=42)
    model2 = ToyModel()
    train_loader, train_sampler, val_loader = _build_loaders()
    with pytest.raises(TrainerError, match="seed"):
        fit(
            model2,
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            class_weights=torch.ones(3),
            cfg=make_training_cfg(max_epochs=8),
            lr=5e-2,
            seed=43,  # different seed
            run_dir=tmp_path / "run",
            device=DEVICE,
            resume=True,
        )


def test_resume_refuses_mismatched_effective_cfg(tmp_path):
    _run_and_pause(tmp_path, cfg=make_training_cfg(max_epochs=8))
    model2 = ToyModel()
    train_loader, train_sampler, val_loader = _build_loaders()
    different_cfg = make_training_cfg(max_epochs=8, batch_size=999)
    with pytest.raises(TrainerError, match="effective_cfg"):
        fit(
            model2,
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            class_weights=torch.ones(3),
            cfg=different_cfg,
            lr=5e-2,
            seed=42,
            run_dir=tmp_path / "run",
            device=DEVICE,
            resume=True,
        )


def test_resume_refuses_mismatched_contract_hash(tmp_path, monkeypatch):
    _run_and_pause(tmp_path)
    model2 = ToyModel()
    train_loader, train_sampler, val_loader = _build_loaders()

    import btdl.training.trainer as trainer_module

    monkeypatch.setattr(trainer_module.config, "contract_dir_sha256", lambda: "deadbeef")
    with pytest.raises(TrainerError, match="contract_dir_sha256"):
        fit(
            model2,
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            class_weights=torch.ones(3),
            cfg=make_training_cfg(max_epochs=8),
            lr=5e-2,
            seed=42,
            run_dir=tmp_path / "run",
            device=DEVICE,
            resume=True,
        )


def test_resume_succeeds_when_everything_matches(tmp_path):
    _run_and_pause(tmp_path)
    model2 = ToyModel()
    train_loader, train_sampler, val_loader = _build_loaders()
    result = fit(
        model2,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        class_weights=torch.ones(3),
        cfg=make_training_cfg(max_epochs=8),
        lr=5e-2,
        seed=42,
        run_dir=tmp_path / "run",
        device=DEVICE,
        resume=True,
    )
    # _run_and_pause stops after epoch 1 (epochs 0-1 = 2 epochs); this call
    # resumes through epoch 7 (6 more epochs). Total history: 8 rows.
    assert result.epochs_run == 6
    history = pd.read_csv(result.history_path)
    assert len(history) == 8
    assert list(history["epoch"]) == list(range(8))


def test_existing_run_dir_without_resume_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    torch.manual_seed(0)
    model = ToyModel()
    train_loader, train_sampler, val_loader = _build_loaders()
    cfg = make_training_cfg(max_epochs=1)

    with pytest.raises(TrainerError):
        fit(
            model,
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            class_weights=torch.ones(3),
            cfg=cfg,
            lr=1e-2,
            seed=42,
            run_dir=run_dir,
            device=DEVICE,
        )
