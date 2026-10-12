import csv
import dataclasses
import json
import subprocess

import pytest

from btdl import config
from btdl.cli.final_test import FinalTestError, run_final_test
from btdl.cli.freeze import run_freeze
from btdl.cli.register_frozen import run_register_frozen
from btdl.evaluation.reload_check import ReloadEquivalenceError
from btdl.evaluation.results import validate_evaluation_dir
from btdl.models.registry import MODEL_REGISTRY


def _frozen_run(trained_run):
    tmp_path, run_dir, model, device = trained_run
    run_freeze(run_dir)
    return tmp_path, run_dir, model, device


def _enable_real_mode(monkeypatch, model_name="tiny_cnn"):
    """Temporarily mark the model as NOT reference_only, so the real
    (non-rehearsal) final_test path can be exercised -- tiny_cnn is
    reference_only=True in the actual registry and can only ever rehearse."""

    original_spec = MODEL_REGISTRY[model_name]
    fake_spec = dataclasses.replace(original_spec, reference_only=False)
    monkeypatch.setitem(MODEL_REGISTRY, model_name, fake_spec)


def _write_matching_selection(tmp_path, run_dir, model_name="tiny_cnn"):
    with (run_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    selection_path = tmp_path / "phase2" / "artifacts" / "selection" / f"{model_name}.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({"selected_lr": metadata["lr"]}))
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add selection"], cwd=tmp_path, check=True, capture_output=True)


def _make_unblinding_ready(tmp_path, run_dir, device, model_name="tiny_cnn"):
    """D18: fake_repo's study.yaml already covers only `model_name`
    (conftest.py), so registering it frozen alone makes unblinding_status
    READY. register_frozen requires a frozen+rehearsed run at the selected
    lr for EVERY seed in final_seeds (42/43/44 for the real contract) --
    rather than training two more models, the single trained run is
    duplicated filesystem-wise to the other seeds' run_dirs (register_frozen
    only reads metadata.json/FROZEN.json/eval/rehearsal_val/, never best.pt's
    internal contents beyond its sha256, which a verbatim copy preserves)."""

    import shutil

    from btdl.runs import run_dir_for

    rehearsal_dir = run_dir / "eval" / "rehearsal_val"
    if not rehearsal_dir.is_dir():
        run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)

    with (run_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    lr = metadata["lr"]
    own_seed = metadata["seed"]

    training_cfg = config.load_contract("training")
    for seed in training_cfg["final_seeds"]:
        if seed == own_seed:
            continue
        target_dir = run_dir_for(model_name, lr, seed, smoke=False)
        if target_dir.exists():
            continue
        shutil.copytree(run_dir, target_dir)
        meta_path = target_dir / "metadata.json"
        with meta_path.open() as handle:
            meta = json.load(handle)
        meta["seed"] = seed
        with meta_path.open("w") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

    run_register_frozen(model_name)

    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "register frozen"], cwd=tmp_path, check=True, capture_output=True)


# ---- base gates (apply in both real and rehearsal mode) -----------------------------


def test_final_test_requires_confirm_flag(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    with pytest.raises(FinalTestError, match="confirm-final-test"):
        run_final_test(run_dir, confirm_final_test=False, device=device, batch_size=2)


def test_final_test_requires_frozen_run(trained_run):
    tmp_path, run_dir, model, device = trained_run  # not frozen
    with pytest.raises(FinalTestError, match="not frozen"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_refuses_mismatched_frozen_sha256(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    frozen_path = run_dir / "FROZEN.json"
    with frozen_path.open() as handle:
        frozen = json.load(handle)
    frozen["best_pt_sha256"] = "deadbeef" * 8
    with frozen_path.open("w") as handle:
        json.dump(frozen, handle)

    with pytest.raises(FinalTestError, match="sha256"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_refuses_contract_mismatch(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    frozen_path = run_dir / "FROZEN.json"
    with frozen_path.open() as handle:
        frozen = json.load(handle)
    frozen["contract_dir_sha256"] = "deadbeef" * 8
    with frozen_path.open("w") as handle:
        json.dump(frozen, handle)

    with pytest.raises(FinalTestError, match="contract"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_refuses_non_conformant_metadata(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    metadata["contract_conformant"] = False
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(FinalTestError, match="contract_conformant"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_refuses_dirty_working_tree(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    # Untracked, under phase2/ -> dirty under the scoped definition (A1).
    (tmp_path / "phase2" / "uncommitted_file.txt").write_text("not committed")

    with pytest.raises(FinalTestError, match="clean"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_root_untracked_file_does_not_block(trained_run):
    # A1: an untracked file outside phase2/ must not block final_test.
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    (tmp_path / "untracked_at_root.txt").write_text("not mine to touch")

    metrics = run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)
    assert metrics["split"] == "val"


# ---- C6: real-mode-only gates -----------------------------------------------------------


def test_real_mode_refuses_reference_only_model(trained_run):
    # tiny_cnn is reference_only=True in the real registry -- real mode must
    # always refuse it, regardless of anything else.
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    with pytest.raises(FinalTestError, match="reference_only"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_real_mode_refuses_seed_not_in_final_seeds(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)

    metadata_path = run_dir / "metadata.json"
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    metadata["seed"] = 999  # not in final_seeds
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle)
    # run_dir is under phase2/runs/, which is gitignored -- editing a file
    # inside it never dirties the tree, so no commit is needed (or possible).

    with pytest.raises(FinalTestError, match="final_seeds"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_real_mode_refuses_missing_selection_file(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)

    with pytest.raises(FinalTestError, match="selection"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


def test_real_mode_refuses_lr_not_matching_selection(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)

    selection_path = tmp_path / "phase2" / "artifacts" / "selection" / "tiny_cnn.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({"selected_lr": 9.99}))  # does not match run's lr
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add mismatched selection"], cwd=tmp_path, check=True, capture_output=True)

    with pytest.raises(FinalTestError, match="does not match"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


# ---- C6 success path + log row ---------------------------------------------------------


@pytest.mark.slow
def test_real_mode_succeeds_and_appends_one_log_row(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)
    _make_unblinding_ready(tmp_path, run_dir, device)

    metrics = run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)
    assert metrics["split"] == "test"
    validate_evaluation_dir(run_dir / "eval" / "test")

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    assert log_path.is_file()
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["model_name"] == "tiny_cnn"


@pytest.mark.slow
def test_real_mode_refuses_repeated_run(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)
    _make_unblinding_ready(tmp_path, run_dir, device)

    run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)

    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "log test access"], cwd=tmp_path, check=True, capture_output=True)

    with pytest.raises(FinalTestError, match="already"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1  # the refused second call appended nothing


# ---- D18: simultaneous unblinding gates (real mode only) ------------------------------------


def test_real_mode_refuses_unregistered_model(trained_run, monkeypatch):
    # No cli/register_frozen.py call at all -- no artifacts/frozen/<model>.json.
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)

    with pytest.raises(FinalTestError, match="frozen registration"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


@pytest.mark.slow
def test_real_mode_refuses_sha_mismatch_vs_registration(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)
    _make_unblinding_ready(tmp_path, run_dir, device)

    # Corrupt the registration's recorded sha for this seed.
    frozen_path = tmp_path / "phase2" / "artifacts" / "frozen" / "tiny_cnn.json"
    with frozen_path.open() as handle:
        registration = json.load(handle)
    for entry in registration["seeds"]:
        if entry["seed"] == json.load((run_dir / "metadata.json").open())["seed"]:
            entry["best_pt_sha256"] = "deadbeef" * 8
    with frozen_path.open("w") as handle:
        json.dump(registration, handle)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "corrupt registration sha"], cwd=tmp_path, check=True, capture_output=True)

    with pytest.raises(FinalTestError, match="does not match the frozen"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


@pytest.mark.slow
def test_real_mode_refuses_seed_not_in_registration(trained_run, monkeypatch):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)
    _write_matching_selection(tmp_path, run_dir)
    _make_unblinding_ready(tmp_path, run_dir, device)

    frozen_path = tmp_path / "phase2" / "artifacts" / "frozen" / "tiny_cnn.json"
    with frozen_path.open() as handle:
        registration = json.load(handle)
    own_seed = json.load((run_dir / "metadata.json").open())["seed"]
    registration["seeds"] = [entry for entry in registration["seeds"] if entry["seed"] != own_seed]
    with frozen_path.open("w") as handle:
        json.dump(registration, handle)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "drop seed from registration"], cwd=tmp_path, check=True, capture_output=True)

    with pytest.raises(FinalTestError, match="not registered"):
        run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)


@pytest.mark.slow
@pytest.mark.slow
def test_real_mode_succeeds_when_only_this_model_is_registered(trained_run, monkeypatch):
    # D18 revised: a study.yaml with TWO models, where only tiny_cnn (this
    # run's model) is registered -- unblinding_status/"Registration status"
    # is NOT READY overall (the other study model never registers), but
    # final_test for tiny_cnn must succeed anyway: models are now tested
    # independently, with no cross-model readiness gate.
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)

    study_path = tmp_path / "phase2" / "configs" / "study.yaml"
    study_path.write_text(
        "study_version: '1.0.0'\nmodels:\n  - tiny_cnn\n  - some_other_model\n"
        "baseline_reference: reports/experiments/fake_baseline/experiment_summary.json\n"
    )
    _write_matching_selection(tmp_path, run_dir)
    _make_unblinding_ready(tmp_path, run_dir, device)

    from btdl.cli.unblinding_status import compute_status

    assert compute_status()["ready"] is False  # sanity: overall status genuinely NOT READY

    metrics = run_final_test(run_dir, confirm_final_test=True, device=device, batch_size=2)
    assert metrics["split"] == "test"


@pytest.mark.slow
def test_rehearsal_still_works_when_unregistered(trained_run, monkeypatch):
    # --rehearsal must never be blocked by the registration gate -- it
    # never touches the test split or the access log, so there is nothing
    # to protect.
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    _enable_real_mode(monkeypatch)

    study_path = tmp_path / "phase2" / "configs" / "study.yaml"
    study_path.write_text(
        "study_version: '1.0.0'\nmodels:\n  - tiny_cnn\n  - some_other_model\n"
        "baseline_reference: reports/experiments/fake_baseline/experiment_summary.json\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "two-model study"], cwd=tmp_path, check=True, capture_output=True)
    # No registration at all -- still must succeed in rehearsal mode.
    metrics = run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)
    assert metrics["split"] == "val"


# ---- C7: rehearsal -----------------------------------------------------------------------


@pytest.mark.slow
def test_rehearsal_succeeds_without_selection_or_real_mode(trained_run):
    # tiny_cnn is reference_only and has no selection file -- rehearsal must
    # still succeed (selection + reference_only gates are waived).
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    metrics = run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)
    assert metrics["split"] == "val"
    validate_evaluation_dir(run_dir / "eval" / "rehearsal_val")


@pytest.mark.slow
def test_rehearsal_writes_no_log_row(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    if log_path.is_file():
        with log_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 0


def test_rehearsal_still_requires_frozen(trained_run):
    tmp_path, run_dir, model, device = trained_run  # not frozen
    with pytest.raises(FinalTestError, match="not frozen"):
        run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)


def test_rehearsal_still_requires_clean_tree(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    (tmp_path / "phase2" / "uncommitted_file.txt").write_text("not committed")
    with pytest.raises(FinalTestError, match="clean"):
        run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)


@pytest.mark.slow
def test_rehearsal_can_be_run_multiple_times(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)
    # Should not raise: rehearsal has no "already logged" gate.
    run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)


# ---- C8: reload equivalence --------------------------------------------------------------


def test_reload_mismatch_is_detected(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    val_predictions_path = run_dir / "val_predictions.csv"
    import pandas as pd

    frame = pd.read_csv(val_predictions_path)
    # Corrupt one probability column so it can no longer match a fresh reload.
    prob_columns = [c for c in frame.columns if c.startswith("prob_")]
    frame.loc[0, prob_columns[0]] = 0.0
    frame.to_csv(val_predictions_path, index=False)
    # run_dir is gitignored -- no commit needed (or possible) for this edit.

    with pytest.raises(ReloadEquivalenceError):
        run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)


# ---- B2/B3: append-only test-access log; no intermediate commits needed -------------------


def _fully_ready_for_export(trained_run, monkeypatch, model_name="tiny_cnn"):
    """Evaluate + freeze + efficiency + rehearsal for the trained run,
    duplicate to every final_seed, register frozen -- everything
    final_test and export_results need, for all 3 final seeds."""

    from btdl.cli.efficiency import run_efficiency
    from btdl.cli.evaluate import run_evaluate
    from btdl.cli.register_frozen import run_register_frozen

    tmp_path, run_dir, model, device = trained_run
    run_evaluate(run_dir, split="val", device=device, batch_size=2)
    run_freeze(run_dir)
    run_efficiency(run_dir, device=device)
    _enable_real_mode(monkeypatch, model_name)
    _write_matching_selection(tmp_path, run_dir, model_name)
    run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True, batch_size=2)

    import shutil

    from btdl.runs import run_dir_for

    with (run_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    lr, own_seed = metadata["lr"], metadata["seed"]
    training_cfg = config.load_contract("training")
    for seed in training_cfg["final_seeds"]:
        if seed == own_seed:
            continue
        target_dir = run_dir_for(model_name, lr, seed, smoke=False)
        if target_dir.exists():
            continue
        shutil.copytree(run_dir, target_dir)
        meta_path = target_dir / "metadata.json"
        with meta_path.open() as handle:
            meta = json.load(handle)
        meta["seed"] = seed
        with meta_path.open("w") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

    run_register_frozen(model_name)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "ready for export"], cwd=tmp_path, check=True, capture_output=True)

    return tmp_path, device, training_cfg["final_seeds"], lr


@pytest.mark.slow
def test_final_test_then_export_sequence_needs_no_intermediate_commits(trained_run, monkeypatch):
    from btdl.cli.export_results import run_export_results
    from btdl.runs import run_dir_for

    tmp_path, device, final_seeds, lr = _fully_ready_for_export(trained_run, monkeypatch)

    # final_test for all 3 seeds, in order, with NO commit between calls.
    for seed in final_seeds:
        seed_run_dir = run_dir_for("tiny_cnn", lr, seed, smoke=False)
        metrics = run_final_test(seed_run_dir, confirm_final_test=True, device=device, batch_size=2)
        assert metrics["split"] == "test"

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3  # one row per seed, all appended without any commit

    # export_results for all 3 seeds, still with no commit anywhere above.
    for seed in final_seeds:
        seed_run_dir = run_dir_for("tiny_cnn", lr, seed, smoke=False)
        dest_dir = run_export_results(seed_run_dir)
        assert dest_dir.is_dir()

    # Only now commit everything together (B3's documented ordering).
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "log + results"], cwd=tmp_path, check=True, capture_output=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True)
    assert status.stdout == ""


@pytest.mark.slow
def test_non_append_edit_to_log_makes_tree_dirty(trained_run, monkeypatch):
    tmp_path, device, final_seeds, lr = _fully_ready_for_export(trained_run, monkeypatch)
    from btdl.runs import run_dir_for

    seed0, seed1 = final_seeds[0], final_seeds[1]
    run_dir0 = run_dir_for("tiny_cnn", lr, seed0, smoke=False)
    run_final_test(run_dir0, confirm_final_test=True, device=device, batch_size=2)

    # Commit so there is a real, COMMITTED log to compare the next edit
    # against (editing a still-uncommitted file is indistinguishable from
    # writing a fresh well-formed file -- the exemption is about not
    # rewriting COMMITTED history, not about the file's age).
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "log seed0"], cwd=tmp_path, check=True, capture_output=True)

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    lines = log_path.read_text().splitlines(keepends=True)
    # Rewrite the (committed) header line in place (an edit, not an
    # append) -- no longer an exact byte prefix of HEAD, so this must
    # dirty the tree.
    lines[0] = lines[0].replace("run_dir", "RUN_DIR")
    log_path.write_text("".join(lines))

    run_dir1 = run_dir_for("tiny_cnn", lr, seed1, smoke=False)
    with pytest.raises(FinalTestError, match="clean"):
        run_final_test(run_dir1, confirm_final_test=True, device=device, batch_size=2)
