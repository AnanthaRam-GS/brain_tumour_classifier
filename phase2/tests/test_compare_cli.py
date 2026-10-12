import json

import numpy as np
import pandas as pd
import pytest

from btdl import config
from btdl.comparison.loaders import ComparisonIntegrityError
from btdl.comparison.plots import PREVIEW_LABEL
from btdl.comparison.report import CompareError, run_compare
from btdl.contracts import label_to_index
from btdl.data.dataset import RoiDataset
from btdl.evaluation.predict import Predictions
from btdl.evaluation.results import write_evaluation

RESULTS_DIR_RELPATH = "phase2/artifacts/results"
FROZEN_DIR_RELPATH = "phase2/artifacts/frozen"

pytestmark = pytest.mark.slow


def _fake_predictions(rng, *, correct_fraction=0.9):
    ds = RoiDataset("test", augment=False, seed=0, allow_test=True)
    sample_ids = ds.sample_ids
    patient_ids = ds.patient_ids
    y_true_idx = np.array([label_to_index(label) for label in ds.labels])
    n = len(sample_ids)
    probs = np.full((n, 3), 0.05)
    for i, idx in enumerate(y_true_idx):
        if rng.random() < correct_fraction:
            probs[i, idx] = 0.9
        else:
            probs[i, (idx + 1) % 3] = 0.9
    probs = probs / probs.sum(axis=1, keepdims=True)
    return Predictions(sample_ids=sample_ids, patient_ids=patient_ids, y_true_idx=y_true_idx, probs=probs)


def _build_fake_export(tmp_path, model_name, seed, lr, rng, *, correct_fraction=0.9, split="test"):
    predictions = _fake_predictions(rng, correct_fraction=correct_fraction)
    run_metadata = {
        "model_name": model_name,
        "model_version": "1.0.0",
        "seed": seed,
        "lr": lr,
        "best_epoch": 3,
        "git_commit": "deadbeef",
        "contract_dir_sha256": config.contract_dir_sha256(),
        "contract_conformant": True,
    }

    seed_dir = tmp_path / RESULTS_DIR_RELPATH / model_name / f"seed{seed}"
    out_dir = seed_dir / "eval" / split
    write_evaluation(out_dir, predictions=predictions, run_metadata=run_metadata, split=split)

    seed_dir.mkdir(parents=True, exist_ok=True)
    with (seed_dir / "metadata.json").open("w") as handle:
        json.dump(run_metadata, handle)

    best_pt_sha256 = f"{'ab' * 20}{model_name}{seed}"[:64]
    frozen = {
        "best_pt_sha256": best_pt_sha256,
        "contract_dir_sha256": config.contract_dir_sha256(),
        "frozen_at_commit": "deadbeef",
    }
    with (seed_dir / "FROZEN.json").open("w") as handle:
        json.dump(frozen, handle)

    pd.DataFrame({"epoch": [0, 1, 2], "val_macro_f1": [0.5, 0.6, 0.7]}).to_csv(seed_dir / "history.csv", index=False)
    return best_pt_sha256


def _write_registration(tmp_path, model_name, lr, seeds_and_shas):
    registration = {
        "model_name": model_name,
        "model_version": "1.0.0",
        "weights_id": None,
        "selected_lr": lr,
        "contract_dir_sha256": config.contract_dir_sha256(),
        "foundation_lock_version": "1.1.0",
        "git_commit": "deadbeef",
        "seeds": [
            {
                "seed": seed,
                "run_dir": f"phase2/runs/{model_name}/lr_fake_seed{seed}",
                "best_pt_sha256": sha,
                "frozen_at_commit": "deadbeef",
                "best_epoch": 3,
                "best_val_macro_f1": 0.7,
            }
            for seed, sha in seeds_and_shas
        ],
    }
    reg_dir = tmp_path / FROZEN_DIR_RELPATH
    reg_dir.mkdir(parents=True, exist_ok=True)
    with (reg_dir / f"{model_name}.json").open("w") as handle:
        json.dump(registration, handle)


def _write_local_val_run(tmp_path, model_name, seed, lr, rng, *, correct_fraction=0.9):
    """A registered run's LOCAL eval/val/ (for val-preview mode) -- deliberately
    at phase2/runs/, with NO eval/test/ created at all, to prove val-preview
    never needs it."""

    run_dir = tmp_path / "phase2" / "runs" / model_name / f"lr_fake_seed{seed}"
    val_predictions = _fake_predictions(rng, correct_fraction=correct_fraction)
    # Reuse the "test" split's samples as a stand-in val prediction set purely
    # for this synthetic test (the real pipeline never does this -- it's only
    # to avoid needing a second RoiDataset split fixture here).
    run_metadata = {
        "model_name": model_name,
        "model_version": "1.0.0",
        "seed": seed,
        "lr": lr,
        "best_epoch": 3,
        "git_commit": "deadbeef",
        "contract_dir_sha256": config.contract_dir_sha256(),
        "contract_conformant": True,
    }
    write_evaluation(run_dir / "eval" / "val", predictions=val_predictions, run_metadata=run_metadata, split="val")
    pd.DataFrame({"epoch": [0, 1, 2], "val_macro_f1": [0.5, 0.6, 0.7]}).to_csv(run_dir / "history.csv", index=False)
    return run_dir


def _write_matching_fake_baseline(tmp_path):
    """A baseline CSV whose sample_ids exactly match fake_repo's test split
    ("7", "8" -- see conftest.py's FAKE_REPO_SPECS)."""

    ds = RoiDataset("test", augment=False, seed=0, allow_test=True)
    baseline_dir = tmp_path / "reports" / "experiments" / "fake_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "sample_id": ds.sample_ids,
            "patient_id": ds.patient_ids,
            "true_label": ds.labels,
            "predicted_label": ds.labels,
            "prob_meningioma": [0.8 if lbl == 1 else 0.1 for lbl in ds.labels],
            "prob_glioma": [0.8 if lbl == 2 else 0.1 for lbl in ds.labels],
            "prob_pituitary": [0.8 if lbl == 3 else 0.1 for lbl in ds.labels],
        }
    )
    frame.to_csv(baseline_dir / "test_predictions.csv", index=False)
    (baseline_dir / "experiment_summary.json").write_text("{}")


def _full_study_with_two_models(tmp_path, models=("tiny_cnn", "other_model")):
    study_path = tmp_path / "phase2" / "configs" / "study.yaml"
    models_block = "\n".join(f"  - {m}" for m in models)
    study_path.write_text(
        f"study_version: '1.0.0'\nmodels:\n{models_block}\n"
        "baseline_reference: reports/experiments/fake_baseline/experiment_summary.json\n"
    )


@pytest.fixture
def exported_model(fake_repo):
    """Fully exported (3 final-seed) test-mode results + matching frozen
    registration for 'tiny_cnn', with study.yaml covering ONLY tiny_cnn so
    unblinding_status is READY by construction."""

    tmp_path, manifest, specs = fake_repo
    rng = np.random.default_rng(0)
    training_cfg = config.load_contract("training")
    lr = training_cfg["lr_grid"][0]

    seeds_and_shas = []
    for seed in training_cfg["final_seeds"]:
        sha = _build_fake_export(tmp_path, "tiny_cnn", seed, lr, rng)
        seeds_and_shas.append((seed, sha))
    _write_registration(tmp_path, "tiny_cnn", lr, seeds_and_shas)

    study_path = tmp_path / "phase2" / "configs" / "study.yaml"
    study_path.write_text(
        "study_version: '1.0.0'\nmodels:\n  - tiny_cnn\n"
        "baseline_reference: reports/experiments/fake_baseline/experiment_summary.json\n"
    )
    _write_matching_fake_baseline(tmp_path)
    return tmp_path, lr, seeds_and_shas


# ---- missing-model refusal in test mode -------------------------------------------------------


def test_missing_model_refused_in_test_mode(exported_model):
    tmp_path, lr, seeds_and_shas = exported_model
    # Add a second study model that's registered frozen (so unblinding_status
    # stays READY) but never actually exported -- compare must refuse citing
    # it as missing, distinct from an unblinding-readiness failure.
    _write_registration(tmp_path, "other_model", lr, seeds_and_shas)
    _full_study_with_two_models(tmp_path, models=("tiny_cnn", "other_model"))

    with pytest.raises(CompareError, match="other_model"):
        run_compare("test")


def test_not_ready_refused_in_test_mode(exported_model):
    tmp_path, lr, seeds_and_shas = exported_model
    # A study model with NO registration at all -> unblinding_status NOT READY.
    _full_study_with_two_models(tmp_path, models=("tiny_cnn", "other_model"))

    with pytest.raises(CompareError, match="NOT READY"):
        run_compare("test")


# ---- sha mismatch refusal -----------------------------------------------------------------------


def test_sha_mismatch_refused(exported_model):
    tmp_path, lr, seeds_and_shas = exported_model
    seed_dir = tmp_path / RESULTS_DIR_RELPATH / "tiny_cnn" / f"seed{seeds_and_shas[0][0]}"
    frozen_path = seed_dir / "FROZEN.json"
    with frozen_path.open() as handle:
        frozen = json.load(handle)
    frozen["best_pt_sha256"] = "deadbeef" * 8
    with frozen_path.open("w") as handle:
        json.dump(frozen, handle)

    with pytest.raises(ComparisonIntegrityError, match="does not match"):
        run_compare("test")


# ---- val-preview: labelling + never needs eval/test ----------------------------------------------


def test_preview_labelling(fake_repo):
    tmp_path, manifest, specs = fake_repo
    rng = np.random.default_rng(1)
    training_cfg = config.load_contract("training")
    lr = training_cfg["lr_grid"][0]
    seeds_and_shas = []
    for seed in training_cfg["final_seeds"]:
        run_dir = _write_local_val_run(tmp_path, "tiny_cnn", seed, lr, rng)
        seeds_and_shas.append((seed, f"fakesha{seed}"))
    registration = {
        "model_name": "tiny_cnn", "model_version": "1.0.0", "weights_id": None, "selected_lr": lr,
        "contract_dir_sha256": config.contract_dir_sha256(), "foundation_lock_version": "1.1.0",
        "git_commit": "deadbeef",
        "seeds": [
            {"seed": s, "run_dir": f"phase2/runs/tiny_cnn/lr_fake_seed{s}", "best_pt_sha256": sha,
             "frozen_at_commit": "deadbeef", "best_epoch": 3, "best_val_macro_f1": 0.7}
            for s, sha in seeds_and_shas
        ],
    }
    reg_dir = tmp_path / FROZEN_DIR_RELPATH
    reg_dir.mkdir(parents=True, exist_ok=True)
    with (reg_dir / "tiny_cnn.json").open("w") as handle:
        json.dump(registration, handle)

    outcome = run_compare("val-preview")
    assert outcome["result"]["label"] == PREVIEW_LABEL
    assert outcome["results_md_path"].read_text().count(PREVIEW_LABEL) >= 1
    # Never touched eval/test/ -- it was never even created for this model.
    assert not (tmp_path / "phase2" / "runs" / "tiny_cnn" / "lr_fake_seed42" / "eval" / "test").exists()


def test_val_preview_never_requires_eval_test_to_exist(fake_repo):
    # Same fixture as above, but explicitly asserts no eval/test/ dir exists
    # ANYWHERE under phase2/runs/ before OR after the preview run.
    tmp_path, manifest, specs = fake_repo
    rng = np.random.default_rng(2)
    training_cfg = config.load_contract("training")
    lr = training_cfg["lr_grid"][0]
    seeds_and_shas = []
    for seed in training_cfg["final_seeds"]:
        _write_local_val_run(tmp_path, "tiny_cnn", seed, lr, rng)
        seeds_and_shas.append((seed, f"fakesha{seed}"))
    _write_registration(tmp_path, "tiny_cnn", lr, seeds_and_shas)

    assert list((tmp_path / "phase2" / "runs").rglob("eval/test")) == []
    run_compare("val-preview")
    assert list((tmp_path / "phase2" / "runs").rglob("eval/test")) == []


# ---- determinism ----------------------------------------------------------------------------------


def test_compare_is_deterministic(exported_model):
    tmp_path, lr, seeds_and_shas = exported_model
    outcome1 = run_compare("test")
    # Re-run against the identical inputs: must be byte-identical when
    # JSON-serialized with sort_keys=True.
    outcome2 = run_compare("test")
    serialized1 = json.dumps(outcome1["result"], sort_keys=True)
    serialized2 = json.dumps(outcome2["result"], sort_keys=True)
    assert serialized1 == serialized2
