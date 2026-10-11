import json

import pytest

from btdl.cli.efficiency_central import (
    COMPARISON_DIR_RELPATH,
    EfficiencyCentralError,
    run_efficiency_central,
)
from btdl.training.device import select_device

DEVICE = select_device("cpu")


def _write_study(tmp_path, models):
    study_path = tmp_path / "phase2" / "configs" / "study.yaml"
    models_block = "\n".join(f"  - {name}" for name in models)
    study_path.write_text(
        f"study_version: '1.0.0'\nmodels:\n{models_block}\n"
        "baseline_reference: reports/experiments/fake_baseline/experiment_summary.json\n"
    )


@pytest.mark.slow
def test_present_model_measured_and_missing_model_skipped(fake_repo):
    tmp_path, manifest, specs = fake_repo
    _write_study(tmp_path, ["tiny_cnn", "some_unregistered_model"])

    payload = run_efficiency_central(DEVICE, num_threads=1)

    assert payload["present_models"] == ["tiny_cnn"]
    assert payload["missing_models"] == ["some_unregistered_model"]
    assert "tiny_cnn" in payload["results"]
    assert "some_unregistered_model" not in payload["results"]
    assert payload["results"]["tiny_cnn"]["parameters"]["total"] > 0
    assert "1" in payload["results"]["tiny_cnn"]["inference"]


@pytest.mark.slow
def test_writes_tracked_output_file(fake_repo):
    tmp_path, manifest, specs = fake_repo
    _write_study(tmp_path, ["tiny_cnn"])

    run_efficiency_central(DEVICE, num_threads=1)

    out_path = tmp_path / COMPARISON_DIR_RELPATH / f"efficiency_{DEVICE.type}.json"
    assert out_path.is_file()
    with out_path.open() as handle:
        payload = json.load(handle)
    assert payload["present_models"] == ["tiny_cnn"]


@pytest.mark.slow
def test_refuses_overwrite_without_flag(fake_repo):
    tmp_path, manifest, specs = fake_repo
    _write_study(tmp_path, ["tiny_cnn"])

    run_efficiency_central(DEVICE, num_threads=1)
    with pytest.raises(EfficiencyCentralError, match="overwrite"):
        run_efficiency_central(DEVICE, num_threads=1)


@pytest.mark.slow
def test_overwrite_flag_allows_replace(fake_repo):
    tmp_path, manifest, specs = fake_repo
    _write_study(tmp_path, ["tiny_cnn"])

    run_efficiency_central(DEVICE, num_threads=1)
    payload = run_efficiency_central(DEVICE, num_threads=1, overwrite=True)
    assert payload["present_models"] == ["tiny_cnn"]


@pytest.mark.slow
def test_num_threads_recorded_for_cpu(fake_repo):
    tmp_path, manifest, specs = fake_repo
    _write_study(tmp_path, ["tiny_cnn"])

    payload = run_efficiency_central(DEVICE, num_threads=2)
    assert payload["machine"]["num_threads"] == 2


def test_no_study_models_present_means_empty_results(fake_repo):
    tmp_path, manifest, specs = fake_repo
    _write_study(tmp_path, ["not_a_real_model_a", "not_a_real_model_b"])

    payload = run_efficiency_central(DEVICE, num_threads=1)
    assert payload["present_models"] == []
    assert sorted(payload["missing_models"]) == ["not_a_real_model_a", "not_a_real_model_b"]
    assert payload["results"] == {}
