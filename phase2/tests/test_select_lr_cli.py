import json

import pytest

from btdl import config
from btdl.cli.select_lr import SelectLrError, run_select_lr
from btdl.runs import run_dir_for


def _write_fake_grid_run(tmp_path, model_name, lr, seed, *, best_val_macro_f1, best_val_loss, conformant=True, dirty=False):
    run_dir = run_dir_for(model_name, lr, seed, smoke=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best.pt").write_bytes(f"fake-checkpoint-{lr}".encode())
    (run_dir / "history.csv").write_text("epoch,val_macro_f1\n0,{}\n".format(best_val_macro_f1))
    metadata = {
        "model_name": model_name,
        "lr": lr,
        "seed": seed,
        "best_epoch": 0,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_loss": best_val_loss,
        "contract_conformant": conformant,
        "git_dirty": dirty,
    }
    with (run_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle)
    return run_dir


def _lr_grid_and_seed():
    training_cfg = config.load_contract("training")
    return list(training_cfg["lr_grid"]), training_cfg["grid_seed"]


def test_refuses_when_a_grid_run_is_missing(fake_repo):
    lr_grid, grid_seed = _lr_grid_and_seed()
    # Only write runs for the first two lrs, leave the last missing.
    for lr in lr_grid[:-1]:
        _write_fake_grid_run(None, "tiny_cnn", lr, grid_seed, best_val_macro_f1=0.5, best_val_loss=1.0)

    with pytest.raises(SelectLrError, match="missing or incomplete"):
        run_select_lr("tiny_cnn")


def test_refuses_when_a_grid_run_is_non_conformant(fake_repo):
    lr_grid, grid_seed = _lr_grid_and_seed()
    for lr in lr_grid:
        _write_fake_grid_run(
            None, "tiny_cnn", lr, grid_seed, best_val_macro_f1=0.5, best_val_loss=1.0,
            conformant=(lr != lr_grid[0]),
        )

    with pytest.raises(SelectLrError, match="contract_conformant"):
        run_select_lr("tiny_cnn")


def test_refuses_when_a_grid_run_is_dirty(fake_repo):
    lr_grid, grid_seed = _lr_grid_and_seed()
    for lr in lr_grid:
        _write_fake_grid_run(
            None, "tiny_cnn", lr, grid_seed, best_val_macro_f1=0.5, best_val_loss=1.0,
            dirty=(lr == lr_grid[0]),
        )

    with pytest.raises(SelectLrError, match="git_dirty"):
        run_select_lr("tiny_cnn")


def test_selects_by_best_val_macro_f1(fake_repo):
    lr_grid, grid_seed = _lr_grid_and_seed()
    f1_values = [0.5, 0.9, 0.3]  # middle lr wins
    for lr, f1 in zip(lr_grid, f1_values):
        _write_fake_grid_run(None, "tiny_cnn", lr, grid_seed, best_val_macro_f1=f1, best_val_loss=1.0)

    selection = run_select_lr("tiny_cnn")
    assert selection["selected_lr"] == lr_grid[1]


def test_tie_break_by_lower_val_loss(fake_repo):
    lr_grid, grid_seed = _lr_grid_and_seed()
    # All tie on macro-F1; lr_grid[2] has the lowest val loss.
    losses = [1.0, 0.8, 0.5]
    for lr, loss in zip(lr_grid, losses):
        _write_fake_grid_run(None, "tiny_cnn", lr, grid_seed, best_val_macro_f1=0.7, best_val_loss=loss)

    selection = run_select_lr("tiny_cnn")
    assert selection["selected_lr"] == lr_grid[2]


def test_tie_break_by_lower_lr(fake_repo):
    lr_grid, grid_seed = _lr_grid_and_seed()
    # All tie on macro-F1 AND val loss; lowest lr should win.
    for lr in lr_grid:
        _write_fake_grid_run(None, "tiny_cnn", lr, grid_seed, best_val_macro_f1=0.7, best_val_loss=0.9)

    selection = run_select_lr("tiny_cnn")
    assert selection["selected_lr"] == min(lr_grid)


def test_writes_tracked_selection_file_with_required_fields(fake_repo):
    tmp_path, manifest, specs = fake_repo
    lr_grid, grid_seed = _lr_grid_and_seed()
    for lr in lr_grid:
        _write_fake_grid_run(None, "tiny_cnn", lr, grid_seed, best_val_macro_f1=0.7, best_val_loss=0.9)

    selection = run_select_lr("tiny_cnn")
    selection_path = tmp_path / "phase2" / "artifacts" / "selection" / "tiny_cnn.json"
    assert selection_path.is_file()
    assert len(selection["candidates"]) == len(lr_grid)
    for candidate in selection["candidates"]:
        assert {"lr", "best_epoch", "best_val_macro_f1", "best_val_loss", "run_dir", "best_pt_sha256"}.issubset(
            candidate.keys()
        )
    assert "contract_dir_sha256" in selection
    assert "git_commit" in selection


def test_refuses_to_overwrite_existing_selection(fake_repo):
    lr_grid, grid_seed = _lr_grid_and_seed()
    for lr in lr_grid:
        _write_fake_grid_run(None, "tiny_cnn", lr, grid_seed, best_val_macro_f1=0.7, best_val_loss=0.9)

    run_select_lr("tiny_cnn")
    with pytest.raises(SelectLrError, match="already exists"):
        run_select_lr("tiny_cnn")
