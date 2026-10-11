"""Loads per-model, per-seed prediction data for the comparison pipeline.

Test mode reads ONLY exported phase2/artifacts/results/<model>/seed<k>/
directories (never phase2/runs/, never the test split directly); every
seed's checkpoint identity is re-verified against its frozen registration
(D18) before anything is used. Val-preview mode reads registered models'
LOCAL eval/val/ directories (via their frozen registration's run_dir) and
never touches eval/test/ or constructs a test-split dataset.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from btdl import config
from btdl.cli.register_frozen import FROZEN_REGISTRATION_DIR_RELPATH
from btdl.contracts import CLASS_NAMES_BY_INDEX
from btdl.evaluation.results import validate_evaluation_dir

RESULTS_DIR_RELPATH = "phase2/artifacts/results"


class ComparisonInputError(ValueError):
    """Data for a model/seed is not yet available -- caught by
    load_study_models() and reported as 'missing', never a hard failure."""


class ComparisonIntegrityError(ValueError):
    """A real integrity violation (e.g. a checkpoint sha no longer matches
    its frozen registration) -- always propagates; never swallowed into a
    'missing model' list."""


def _read_predictions_csv(path):
    frame = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str})
    label_to_index = {1: 0, 2: 1, 3: 2}
    y_true_idx = frame["true_label"].map(label_to_index).to_numpy()
    prob_columns = [f"prob_{name}" for name in CLASS_NAMES_BY_INDEX]
    probs = frame[prob_columns].to_numpy(dtype=np.float64)
    return y_true_idx, probs, tuple(frame["sample_id"]), tuple(frame["patient_id"])


def _load_registration(model_name, *, repo_root):
    registration_path = repo_root / FROZEN_REGISTRATION_DIR_RELPATH / f"{model_name}.json"
    if not registration_path.is_file():
        raise ComparisonInputError(f"no frozen registration for {model_name!r} at {registration_path}")
    with registration_path.open() as handle:
        return json.load(handle)


def load_test_seed(model_name, seed, *, registration, repo_root=None) -> dict:
    """One seed's exported test-mode data. Raises ComparisonInputError if
    not yet exported, ComparisonIntegrityError if the exported checkpoint's
    sha256 no longer matches the frozen registration for this seed."""

    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    seed_dir = repo_root / RESULTS_DIR_RELPATH / model_name / f"seed{seed}"
    if not seed_dir.is_dir():
        raise ComparisonInputError(f"{seed_dir} does not exist -- run cli/export_results.py for this seed")

    frozen_path = seed_dir / "FROZEN.json"
    if not frozen_path.is_file():
        raise ComparisonInputError(f"{seed_dir} is missing FROZEN.json")
    with frozen_path.open() as handle:
        frozen = json.load(handle)

    registered_sha = next(
        (entry["best_pt_sha256"] for entry in registration.get("seeds", []) if entry["seed"] == seed), None
    )
    if registered_sha is None or frozen.get("best_pt_sha256") != registered_sha:
        raise ComparisonIntegrityError(
            f"{seed_dir}: exported FROZEN.json best_pt_sha256 ({frozen.get('best_pt_sha256')}) does not "
            f"match the frozen registration for seed {seed} ({registered_sha})"
        )

    test_dir = seed_dir / "eval" / "test"
    if not test_dir.is_dir():
        raise ComparisonInputError(f"{test_dir} does not exist")
    validate_evaluation_dir(test_dir, require_contract=True)

    y_true_idx, probs, sample_ids, patient_ids = _read_predictions_csv(test_dir / "predictions.csv")
    with (test_dir / "metrics.json").open() as handle:
        metrics = json.load(handle)
    with (test_dir / "stratified.json").open() as handle:
        stratified = json.load(handle)

    history_path = seed_dir / "history.csv"
    return {
        "seed": seed,
        "y_true_idx": y_true_idx,
        "probs": probs,
        "sample_ids": sample_ids,
        "patient_ids": patient_ids,
        "metrics": metrics,
        "stratified": stratified,
        "history_path": history_path if history_path.is_file() else None,
    }


def load_test_model(model_name, *, repo_root=None) -> list:
    """All final_seeds' exported test-mode data for a model, in seed order."""

    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    registration = _load_registration(model_name, repo_root=repo_root)

    training_cfg = config.load_contract("training")
    final_seeds = list(training_cfg["final_seeds"])
    return [
        load_test_seed(model_name, seed, registration=registration, repo_root=repo_root) for seed in final_seeds
    ]


def load_val_preview_model(model_name, *, repo_root=None) -> list:
    """Registered seeds' LOCAL eval/val/ data (never eval/test/, never a
    test-split dataset)."""

    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    registration = _load_registration(model_name, repo_root=repo_root)

    seeds_data = []
    for entry in registration.get("seeds", []):
        run_dir = repo_root / entry["run_dir"]
        val_dir = run_dir / "eval" / "val"
        if not val_dir.is_dir():
            raise ComparisonInputError(f"{val_dir} does not exist")
        validate_evaluation_dir(val_dir, require_contract=True)

        y_true_idx, probs, sample_ids, patient_ids = _read_predictions_csv(val_dir / "predictions.csv")
        with (val_dir / "metrics.json").open() as handle:
            metrics = json.load(handle)
        with (val_dir / "stratified.json").open() as handle:
            stratified = json.load(handle)

        history_path = run_dir / "history.csv"
        seeds_data.append(
            {
                "seed": entry["seed"],
                "y_true_idx": y_true_idx,
                "probs": probs,
                "sample_ids": sample_ids,
                "patient_ids": patient_ids,
                "metrics": metrics,
                "stratified": stratified,
                "history_path": history_path if history_path.is_file() else None,
            }
        )
    return sorted(seeds_data, key=lambda d: d["seed"])


def load_study_models(mode: str, *, repo_root=None) -> tuple:
    """Returns (data_by_model: dict[name, list[seed_dict]], missing: list[str]).
    `missing` lists study models without usable `mode` data. A
    ComparisonIntegrityError (sha mismatch) is NEVER caught here -- it
    always propagates."""

    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    study = config.load_study()
    loader = load_test_model if mode == "test" else load_val_preview_model

    data_by_model = {}
    missing = []
    for model_name in study["models"]:
        try:
            data_by_model[model_name] = loader(model_name, repo_root=repo_root)
        except ComparisonInputError:
            missing.append(model_name)
    return data_by_model, missing


def load_central_efficiency(*, repo_root=None) -> dict:
    """{device_type: payload} for every artifacts/comparison/efficiency_*.json present."""

    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    comparison_dir = repo_root / "phase2" / "artifacts" / "comparison"
    results = {}
    if not comparison_dir.is_dir():
        return results
    for path in sorted(comparison_dir.glob("efficiency_*.json")):
        device_type = path.stem[len("efficiency_") :]
        with path.open() as handle:
            results[device_type] = json.load(handle)
    return results
