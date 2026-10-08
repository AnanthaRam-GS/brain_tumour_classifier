import pytest

from btdl.cli.evaluate import EvaluateCliError, run_evaluate
from btdl.evaluation.results import validate_evaluation_dir


@pytest.mark.slow
def test_evaluate_on_val_works(trained_run):
    tmp_path, run_dir, model, device = trained_run
    metrics = run_evaluate(run_dir, split="val", device=device, batch_size=2)
    assert metrics["split"] == "val"
    validate_evaluation_dir(run_dir / "eval" / "val")


@pytest.mark.slow
def test_evaluate_writes_plots(trained_run):
    tmp_path, run_dir, model, device = trained_run
    run_evaluate(run_dir, split="val", device=device, batch_size=2)
    out_dir = run_dir / "eval" / "val"
    assert (out_dir / "confusion_matrix.png").is_file()
    assert (out_dir / "roc_curves.png").is_file()
    assert (out_dir / "training_curves.png").is_file()


def test_evaluate_rejects_test_split(trained_run):
    tmp_path, run_dir, model, device = trained_run
    with pytest.raises(EvaluateCliError, match="final_test"):
        run_evaluate(run_dir, split="test", device=device, batch_size=2)


def test_evaluate_rejects_unknown_split(trained_run):
    tmp_path, run_dir, model, device = trained_run
    with pytest.raises(EvaluateCliError):
        run_evaluate(run_dir, split="bogus", device=device, batch_size=2)


@pytest.mark.slow
def test_evaluate_rebuilds_model_from_registry_not_caller(trained_run):
    # run_evaluate no longer takes a model argument -- it rebuilds one via
    # build_model(metadata.model_name) and loads best.pt into it.
    tmp_path, run_dir, model, device = trained_run
    metrics = run_evaluate(run_dir, split="val", device=device, batch_size=2)
    assert metrics["run"]["model_name"] == "tiny_cnn"


@pytest.mark.slow
def test_evaluate_records_reload_max_abs_diff_within_atol(trained_run):
    tmp_path, run_dir, model, device = trained_run
    metrics = run_evaluate(run_dir, split="val", device=device, batch_size=2)
    assert "reload_max_abs_diff" in metrics
    assert metrics["reload_max_abs_diff"] >= 0.0
    assert metrics["reload_max_abs_diff"] <= 1e-5
