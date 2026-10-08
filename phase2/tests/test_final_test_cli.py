import csv
import json

import pytest

from btdl.cli.final_test import FinalTestError, run_final_test
from btdl.cli.freeze import run_freeze
from btdl.evaluation.results import validate_evaluation_dir


def _frozen_run(trained_run):
    tmp_path, run_dir, model, device = trained_run
    run_freeze(run_dir)
    return tmp_path, run_dir, model, device


def test_final_test_requires_confirm_flag(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    with pytest.raises(FinalTestError, match="confirm-final-test"):
        run_final_test(run_dir, model, confirm_final_test=False, device=device, batch_size=2)


def test_final_test_requires_frozen_run(trained_run):
    tmp_path, run_dir, model, device = trained_run  # not frozen
    with pytest.raises(FinalTestError, match="not frozen"):
        run_final_test(run_dir, model, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_refuses_mismatched_frozen_sha256(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    frozen_path = run_dir / "FROZEN.json"
    with frozen_path.open() as handle:
        frozen = json.load(handle)
    frozen["best_pt_sha256"] = "deadbeef" * 8
    with frozen_path.open("w") as handle:
        json.dump(frozen, handle)

    with pytest.raises(FinalTestError, match="sha256"):
        run_final_test(run_dir, model, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_refuses_contract_mismatch(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    frozen_path = run_dir / "FROZEN.json"
    with frozen_path.open() as handle:
        frozen = json.load(handle)
    frozen["contract_dir_sha256"] = "deadbeef" * 8
    with frozen_path.open("w") as handle:
        json.dump(frozen, handle)

    with pytest.raises(FinalTestError, match="contract"):
        run_final_test(run_dir, model, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_succeeds_and_appends_one_log_row(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    metrics = run_final_test(run_dir, model, confirm_final_test=True, device=device, batch_size=2)
    assert metrics["split"] == "test"
    validate_evaluation_dir(run_dir / "eval" / "test")

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    assert log_path.is_file()
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_dir"] == "phase2/runs/run1"
    assert rows[0]["model_name"] == "ToyModel"


def test_final_test_refuses_repeated_run(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    run_final_test(run_dir, model, confirm_final_test=True, device=device, batch_size=2)

    # A successful final_test writes to the tracked test_access_log.csv,
    # which would otherwise make the SECOND call fail on "clean working
    # tree" first; commit it (a realistic workflow step) so this test
    # isolates the "already logged" refusal specifically.
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "log test access"], cwd=tmp_path, check=True, capture_output=True)

    with pytest.raises(FinalTestError, match="already"):
        run_final_test(run_dir, model, confirm_final_test=True, device=device, batch_size=2)

    log_path = tmp_path / "phase2" / "artifacts" / "test_access_log.csv"
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1  # the refused second call appended nothing


def test_final_test_refuses_non_conformant_metadata(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    metadata["contract_conformant"] = False
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(FinalTestError, match="contract_conformant"):
        run_final_test(run_dir, model, confirm_final_test=True, device=device, batch_size=2)


def test_final_test_refuses_dirty_working_tree(trained_run):
    tmp_path, run_dir, model, device = _frozen_run(trained_run)
    (tmp_path / "uncommitted_file.txt").write_text("not committed")

    with pytest.raises(FinalTestError, match="clean"):
        run_final_test(run_dir, model, confirm_final_test=True, device=device, batch_size=2)
