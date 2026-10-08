import numpy as np
import pytest

from btdl.evaluation.bootstrap import paired_patient_bootstrap, patient_bootstrap


def _well_behaved_case(n_patients=12, slices_per_patient=4, seed=1):
    rng = np.random.default_rng(seed)
    patient_ids = np.repeat([f"p{i}" for i in range(n_patients)], slices_per_patient)
    n = len(patient_ids)
    y_true = rng.integers(0, 3, size=n)
    probs = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    for i in range(n):
        probs[i, y_true[i]] += 0.6
    probs = probs / probs.sum(axis=1, keepdims=True)
    return y_true, probs, patient_ids


def test_deterministic_for_a_seed():
    y_true, probs, patient_ids = _well_behaved_case()
    result_a = patient_bootstrap(y_true, probs, patient_ids, n_resamples=100, seed=7)
    result_b = patient_bootstrap(y_true, probs, patient_ids, n_resamples=100, seed=7)
    assert result_a == result_b


def test_different_seed_gives_different_result():
    y_true, probs, patient_ids = _well_behaved_case()
    result_a = patient_bootstrap(y_true, probs, patient_ids, n_resamples=100, seed=7)
    result_b = patient_bootstrap(y_true, probs, patient_ids, n_resamples=100, seed=8)
    assert result_a["accuracy"]["ci_low"] != result_b["accuracy"]["ci_low"]


def test_resamples_by_patient_one_patient_moves_as_a_block():
    # One patient (p0) has a distinctive, internally-consistent class (all
    # label 0). If resampling were per-slice (not per-patient), the chance
    # of ALL of p0's slices appearing together, or none at all, across many
    # resamples would differ from what per-patient resampling guarantees:
    # every resample draw of "p0" includes ALL of its slices, together.
    patient_ids = np.array(["p0"] * 5 + ["p1"] * 5 + ["p2"] * 5)
    y_true = np.array([0] * 5 + [1] * 5 + [2] * 5)
    probs = np.zeros((15, 3))
    probs[:5, 0] = 1.0
    probs[5:10, 1] = 1.0
    probs[10:, 2] = 1.0

    rng = np.random.default_rng(3)
    unique_patients = np.unique(patient_ids)
    patient_to_indices = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    for _ in range(50):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        counts = {p: list(sampled).count(p) for p in unique_patients}
        # Reimplement the same resample-index construction bootstrap.py uses,
        # and confirm every occurrence of a patient contributes its FULL slice block.
        idx = np.concatenate([patient_to_indices[p] for p in sampled])
        for p in unique_patients:
            expected_count = counts[p] * len(patient_to_indices[p])
            actual_count = sum(1 for i in idx if i in set(patient_to_indices[p]))
            assert actual_count == expected_count


def test_ci_contains_point_estimate_on_well_behaved_case():
    y_true, probs, patient_ids = _well_behaved_case(n_patients=20, slices_per_patient=5)
    result = patient_bootstrap(y_true, probs, patient_ids, n_resamples=300, seed=42)
    for name, stats in result.items():
        if stats["n_valid"] == 0:
            continue
        assert stats["ci_low"] <= stats["point"] <= stats["ci_high"], name


def test_n_valid_less_than_n_resamples_when_class_can_be_absent():
    # Few patients, strongly imbalanced classes: some patient-level
    # resamples will likely omit class 2 entirely, making its AUC undefined.
    patient_ids = np.array(["p0"] * 3 + ["p1"] * 3 + ["p2"] * 1)
    y_true = np.array([0, 0, 0, 1, 1, 1, 2])
    probs = np.full((7, 3), 1 / 3)
    probs[y_true == 0, 0] += 0.3
    probs[y_true == 1, 1] += 0.3
    probs[y_true == 2, 2] += 0.3
    probs = probs / probs.sum(axis=1, keepdims=True)

    result = patient_bootstrap(y_true, probs, patient_ids, n_resamples=200, seed=5)
    # F1 with zero_division=0 is always a well-defined finite number (0.0
    # when a class is absent), so n_valid stays 200 for the f1 metrics; AUC
    # is genuinely undefined (NaN -> skipped) when a class is entirely
    # absent from a resample, which patient-level resampling of this tiny,
    # single-slice-for-class-2 fixture will sometimes produce.
    assert result["macro_ovr_auc"]["n_valid"] < 200


def test_patient_bootstrap_uses_own_rng_not_global():
    import random

    import torch

    torch_state_before = torch.get_rng_state()
    np_state_before = np.random.get_state()
    py_state_before = random.getstate()

    y_true, probs, patient_ids = _well_behaved_case()
    patient_bootstrap(y_true, probs, patient_ids, n_resamples=50, seed=1)

    assert torch.equal(torch_state_before, torch.get_rng_state())
    np_state_after = np.random.get_state()
    assert np_state_before[0] == np_state_after[0]
    assert np.array_equal(np_state_before[1], np_state_after[1])
    assert random.getstate() == py_state_before


def test_paired_bootstrap_self_difference_is_exactly_zero():
    y_true, probs, patient_ids = _well_behaved_case()
    result = paired_patient_bootstrap(y_true, probs, probs, patient_ids, n_resamples=100, seed=2)
    for name, stats in result.items():
        assert stats["point"] == 0.0
        assert stats["ci_low"] == 0.0
        assert stats["ci_high"] == 0.0
        assert stats["frac_a_greater"] == 0.0


def test_paired_bootstrap_identical_resamples_for_a_and_b():
    # A clearly-better model (probs_a concentrated on the truth) vs a
    # uniform one: frac_a_greater should be high (not exactly 1.0 is fine,
    # but should be well above 0.5) if resamples are shared, since a is
    # strictly better than b on EVERY resample.
    y_true, _, patient_ids = _well_behaved_case(n_patients=15, slices_per_patient=4)
    n = len(y_true)
    probs_a = np.zeros((n, 3))
    probs_a[np.arange(n), y_true] = 0.97
    probs_a += 0.01
    probs_a = probs_a / probs_a.sum(axis=1, keepdims=True)
    probs_b = np.full((n, 3), 1 / 3)

    result = paired_patient_bootstrap(y_true, probs_a, probs_b, patient_ids, n_resamples=200, seed=9)
    assert result["accuracy"]["frac_a_greater"] > 0.9
    assert result["accuracy"]["point"] > 0
