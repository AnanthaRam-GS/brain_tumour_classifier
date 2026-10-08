import json

import pytest

from btdl.cli.freeze import FreezeError, run_freeze


def test_freeze_succeeds_on_conformant_clean_run(trained_run):
    tmp_path, run_dir, model, device = trained_run
    frozen = run_freeze(run_dir)
    assert "best_pt_sha256" in frozen
    assert "frozen_at_commit" in frozen
    assert "contract_dir_sha256" in frozen
    assert (run_dir / "FROZEN.json").is_file()


def test_freeze_refuses_double_freeze(trained_run):
    tmp_path, run_dir, model, device = trained_run
    run_freeze(run_dir)
    with pytest.raises(FreezeError, match="already frozen"):
        run_freeze(run_dir)


def test_freeze_refuses_non_conformant_run(trained_run):
    tmp_path, run_dir, model, device = trained_run
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    metadata["contract_conformant"] = False
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(FreezeError, match="contract_conformant"):
        run_freeze(run_dir)


def test_freeze_refuses_dirty_run(trained_run):
    tmp_path, run_dir, model, device = trained_run
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    metadata["git_dirty"] = True
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(FreezeError, match="git_dirty"):
        run_freeze(run_dir)


def test_freeze_refuses_missing_best_pt(trained_run):
    tmp_path, run_dir, model, device = trained_run
    (run_dir / "best.pt").unlink()
    with pytest.raises(FreezeError, match="best.pt"):
        run_freeze(run_dir)


def test_freeze_refuses_missing_metadata(trained_run):
    tmp_path, run_dir, model, device = trained_run
    (run_dir / "metadata.json").unlink()
    with pytest.raises(FreezeError, match="metadata.json"):
        run_freeze(run_dir)
