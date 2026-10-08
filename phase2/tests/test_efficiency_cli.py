import pytest
import json

from btdl.cli.efficiency import run_efficiency


@pytest.mark.slow
def test_efficiency_writes_json_with_parameters_and_inference(trained_run):
    tmp_path, run_dir, model, device = trained_run
    payload = run_efficiency(run_dir, device=device)

    efficiency_path = run_dir / "efficiency.json"
    assert efficiency_path.is_file()
    with efficiency_path.open() as handle:
        on_disk = json.load(handle)
    assert on_disk == payload

    assert payload["parameters"]["total"] > 0
    assert payload["parameters"]["trainable"] > 0
    assert "1" in payload["inference"] or 1 in payload["inference"]


@pytest.mark.slow
def test_efficiency_includes_device_description(trained_run):
    tmp_path, run_dir, model, device = trained_run
    payload = run_efficiency(run_dir, device=device)
    for stats in payload["inference"].values():
        assert stats["device"]["type"] == "cpu"
