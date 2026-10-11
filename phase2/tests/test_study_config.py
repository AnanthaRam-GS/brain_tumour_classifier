import pytest

from btdl import config


def test_load_study_real_file():
    study = config.load_study()
    assert study["study_version"] == "1.0.0"
    assert "efficientnet_b0" in study["models"]
    assert isinstance(study["baseline_reference"], str)


def test_load_study_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="study file not found"):
        config.load_study()


def test_load_study_missing_key_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    study_dir = tmp_path / "phase2" / "configs"
    study_dir.mkdir(parents=True)
    (study_dir / "study.yaml").write_text("study_version: '1.0.0'\n")
    with pytest.raises(ValueError, match="missing required key"):
        config.load_study()


def test_load_study_wrong_type_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    study_dir = tmp_path / "phase2" / "configs"
    study_dir.mkdir(parents=True)
    (study_dir / "study.yaml").write_text(
        "study_version: '1.0.0'\nmodels: not_a_list\nbaseline_reference: x\n"
    )
    with pytest.raises(ValueError, match="must be list"):
        config.load_study()


def test_load_study_non_string_model_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    study_dir = tmp_path / "phase2" / "configs"
    study_dir.mkdir(parents=True)
    (study_dir / "study.yaml").write_text(
        "study_version: '1.0.0'\nmodels: [1, 2]\nbaseline_reference: x\n"
    )
    with pytest.raises(ValueError, match="list of strings"):
        config.load_study()


def test_study_yaml_is_frozen():
    study = config.load_study()
    with pytest.raises((TypeError, AttributeError)):
        study["models"] = []
