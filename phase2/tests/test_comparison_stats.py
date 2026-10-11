import numpy as np
import pytest

from btdl.comparison import stats


def _full_metrics_dict(macro_f1, accuracy, meningioma_f1, glioma_f1, pituitary_f1, auc):
    return {
        "accuracy": accuracy,
        "balanced_accuracy": accuracy,
        "macro_precision": macro_f1,
        "macro_recall": macro_f1,
        "macro_f1": macro_f1,
        "weighted_precision": macro_f1,
        "weighted_recall": macro_f1,
        "weighted_f1": macro_f1,
        "log_loss": 0.5,
        "roc_auc": {"macro_ovr": auc, "weighted_ovr": auc},
        "per_class": {
            "meningioma": {"precision": meningioma_f1, "recall": meningioma_f1, "f1": meningioma_f1, "support": 10},
            "glioma": {"precision": glioma_f1, "recall": glioma_f1, "f1": glioma_f1, "support": 10},
            "pituitary": {"precision": pituitary_f1, "recall": pituitary_f1, "f1": pituitary_f1, "support": 10},
        },
    }


# ---- seed aggregation -----------------------------------------------------------------------


def test_seed_aggregate_headline_mean_and_std():
    seeds_metrics = [
        _full_metrics_dict(0.8, 0.8, 0.7, 0.9, 0.8, 0.95),
        _full_metrics_dict(0.9, 0.9, 0.8, 0.95, 0.85, 0.96),
        _full_metrics_dict(0.85, 0.85, 0.75, 0.92, 0.82, 0.955),
    ]
    result = stats.seed_aggregate_headline(seeds_metrics)
    assert result["macro_f1"]["mean"] == pytest.approx(np.mean([0.8, 0.9, 0.85]))
    assert result["macro_f1"]["std"] == pytest.approx(np.std([0.8, 0.9, 0.85], ddof=1))
    assert result["macro_f1"]["values"] == [0.8, 0.9, 0.85]
    assert result["macro_auc"]["mean"] == pytest.approx(np.mean([0.95, 0.96, 0.955]))


def test_seed_aggregate_headline_single_seed_std_is_zero():
    result = stats.seed_aggregate_headline([_full_metrics_dict(0.8, 0.8, 0.7, 0.9, 0.8, 0.95)])
    assert result["macro_f1"]["mean"] == 0.8
    assert result["macro_f1"]["std"] == 0.0


def test_seed_aggregate_per_class():
    seeds_metrics = [
        _full_metrics_dict(0.8, 0.8, 0.6, 0.9, 0.8, 0.95),
        _full_metrics_dict(0.8, 0.8, 0.8, 0.9, 0.8, 0.95),
    ]
    result = stats.seed_aggregate_per_class(seeds_metrics)
    assert result["meningioma"]["f1"]["mean"] == pytest.approx(0.7)
    assert result["meningioma"]["f1"]["values"] == [0.6, 0.8]


def test_seed_aggregate_stratified():
    seeds_stratified = [
        {"tertiles": {"small": {"per_class_recall": {"meningioma": 0.7, "glioma": 0.8, "pituitary": 0.9}},
                       "medium": {"per_class_recall": {"meningioma": 0.6, "glioma": 0.7, "pituitary": 0.8}},
                       "large": {"per_class_recall": {"meningioma": 0.5, "glioma": 0.6, "pituitary": 0.7}}}},
        {"tertiles": {"small": {"per_class_recall": {"meningioma": 0.9, "glioma": 0.8, "pituitary": 0.7}},
                       "medium": {"per_class_recall": {"meningioma": 0.6, "glioma": 0.7, "pituitary": 0.8}},
                       "large": {"per_class_recall": {"meningioma": 0.5, "glioma": 0.6, "pituitary": 0.7}}}},
    ]
    result = stats.seed_aggregate_stratified(seeds_stratified)
    assert result["small"]["meningioma"]["mean"] == pytest.approx(0.8)
    assert result["large"]["pituitary"]["mean"] == pytest.approx(0.7)


# ---- multi-seed patient bootstrap --------------------------------------------------------------


def _synthetic_probs(y_true_idx, correct_fraction, rng):
    n = len(y_true_idx)
    probs = np.full((n, 3), 0.05)
    for i, label in enumerate(y_true_idx):
        if rng.random() < correct_fraction:
            probs[i] = 0.05
            probs[i, label] = 0.9
        else:
            wrong = (label + 1) % 3
            probs[i] = 0.05
            probs[i, wrong] = 0.9
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def test_paired_bootstrap_model_vs_itself_gives_exactly_zero():
    rng = np.random.default_rng(0)
    y_true_idx = rng.integers(0, 3, size=60)
    patient_ids = np.array([f"p{i // 3}" for i in range(60)])  # 20 patients, 3 samples each
    probs_by_seed = {
        42: _synthetic_probs(y_true_idx, 0.8, rng),
        43: _synthetic_probs(y_true_idx, 0.75, rng),
        44: _synthetic_probs(y_true_idx, 0.85, rng),
    }

    result = stats.paired_multi_seed_patient_bootstrap(
        y_true_idx, probs_by_seed, probs_by_seed, patient_ids, n_resamples=200, seed=0
    )
    for metric_name, stat in result.items():
        assert stat["point"] == 0.0, metric_name
        assert stat["ci_low"] == 0.0, metric_name
        assert stat["ci_high"] == 0.0, metric_name
        assert stat["frac_a_greater"] == 0.0, metric_name


def test_paired_bootstrap_better_model_has_positive_point_diff():
    rng = np.random.default_rng(1)
    y_true_idx = rng.integers(0, 3, size=90)
    patient_ids = np.array([f"p{i // 3}" for i in range(90)])
    probs_a = {42: _synthetic_probs(y_true_idx, 0.95, rng), 43: _synthetic_probs(y_true_idx, 0.95, rng)}
    probs_b = {42: _synthetic_probs(y_true_idx, 0.4, rng), 43: _synthetic_probs(y_true_idx, 0.4, rng)}

    result = stats.paired_multi_seed_patient_bootstrap(y_true_idx, probs_a, probs_b, patient_ids, n_resamples=200, seed=0)
    assert result["macro_f1"]["point"] > 0
    assert result["macro_f1"]["frac_a_greater"] > 0.9


def test_pooled_bootstrap_point_matches_seed_mean():
    rng = np.random.default_rng(2)
    y_true_idx = rng.integers(0, 3, size=60)
    patient_ids = np.array([f"p{i // 3}" for i in range(60)])
    probs_by_seed = {42: _synthetic_probs(y_true_idx, 0.8, rng), 43: _synthetic_probs(y_true_idx, 0.7, rng)}

    result = stats.multi_seed_patient_bootstrap(y_true_idx, probs_by_seed, patient_ids, n_resamples=100, seed=0)
    assert 0.0 <= result["accuracy"]["point"] <= 1.0
    assert result["accuracy"]["ci_low"] <= result["accuracy"]["point"] <= result["accuracy"]["ci_high"]


def test_bootstrap_is_deterministic_given_same_seed():
    rng = np.random.default_rng(3)
    y_true_idx = rng.integers(0, 3, size=60)
    patient_ids = np.array([f"p{i // 3}" for i in range(60)])
    probs_by_seed = {42: _synthetic_probs(y_true_idx, 0.8, rng)}

    result1 = stats.multi_seed_patient_bootstrap(y_true_idx, probs_by_seed, patient_ids, n_resamples=100, seed=7)
    result2 = stats.multi_seed_patient_bootstrap(y_true_idx, probs_by_seed, patient_ids, n_resamples=100, seed=7)
    assert result1 == result2


# ---- ECE on known cases -------------------------------------------------------------------------


def test_ece_perfect_calibration_is_zero():
    # Every prediction's confidence equals its empirical accuracy exactly:
    # construct a case where confidence == 1.0 and predictions are always
    # correct -- ECE must be exactly 0.
    y_true_idx = np.array([0, 1, 2, 0, 1, 2])
    probs = np.zeros((6, 3))
    for i, label in enumerate(y_true_idx):
        probs[i, label] = 1.0
    result = stats.expected_calibration_error(y_true_idx, probs, n_bins=15)
    assert result["ece"] == pytest.approx(0.0, abs=1e-9)


def test_ece_overconfident_wrong_predictions_is_large():
    # Every prediction is maximally confident (1.0) but always WRONG ->
    # accuracy=0 in the top bin, confidence=1 -> ECE == 1.0 exactly.
    y_true_idx = np.array([0, 1, 2, 0])
    probs = np.zeros((4, 3))
    wrong = [1, 2, 0, 1]
    for i, w in enumerate(wrong):
        probs[i, w] = 1.0
    result = stats.expected_calibration_error(y_true_idx, probs, n_bins=15)
    assert result["ece"] == pytest.approx(1.0, abs=1e-9)


def test_ece_bins_structure():
    y_true_idx = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    rng = np.random.default_rng(4)
    probs = rng.dirichlet([1, 1, 1], size=8)
    result = stats.expected_calibration_error(y_true_idx, probs, n_bins=15)
    assert len(result["bins"]) == 15
    assert sum(b["count"] for b in result["bins"]) == 8
    assert 0.0 <= result["ece"] <= 1.0


def test_seed_aggregate_calibration():
    rng = np.random.default_rng(5)
    y_true_idx = rng.integers(0, 3, size=30)
    seeds_data = [
        {"y_true_idx": y_true_idx, "probs": _synthetic_probs(y_true_idx, 0.8, rng)},
        {"y_true_idx": y_true_idx, "probs": _synthetic_probs(y_true_idx, 0.8, rng)},
    ]
    result = stats.seed_aggregate_calibration(seeds_data, n_bins=15)
    assert len(result["values"]) == 2
    assert result["mean"] == pytest.approx(np.mean(result["values"]))
    assert len(result["per_seed_bins"]) == 2
