from btdl.training.early_stopping import EarlyStopping


def test_first_step_always_improves():
    es = EarlyStopping(patience=3)
    result = es.step(0.5, 1.0, epoch=0)
    assert result.improved is True
    assert result.should_stop is False
    assert es.best_epoch == 0


def test_higher_f1_improves():
    es = EarlyStopping(patience=3)
    es.step(0.5, 1.0, epoch=0)
    result = es.step(0.6, 1.0, epoch=1)
    assert result.improved is True
    assert es.best_f1 == 0.6
    assert es.best_epoch == 1


def test_lower_f1_does_not_improve():
    es = EarlyStopping(patience=3)
    es.step(0.6, 1.0, epoch=0)
    result = es.step(0.5, 1.0, epoch=1)
    assert result.improved is False
    assert es.best_f1 == 0.6
    assert es.best_epoch == 0


def test_tie_break_on_lower_val_loss():
    es = EarlyStopping(patience=3)
    es.step(0.6, 1.0, epoch=0)
    result = es.step(0.6, 0.8, epoch=1)  # same f1, lower loss
    assert result.improved is True
    assert es.best_loss == 0.8
    assert es.best_epoch == 1


def test_tie_with_higher_or_equal_loss_does_not_improve():
    es = EarlyStopping(patience=3)
    es.step(0.6, 1.0, epoch=0)
    result = es.step(0.6, 1.0, epoch=1)  # same f1, same loss
    assert result.improved is False
    result2 = es.step(0.6, 1.2, epoch=2)  # same f1, higher loss
    assert result2.improved is False


def test_min_delta_requires_meaningful_improvement():
    es = EarlyStopping(patience=3, min_delta=0.05)
    es.step(0.6, 1.0, epoch=0)
    result = es.step(0.62, 1.0, epoch=1)  # improvement of 0.02 < min_delta
    assert result.improved is False
    result2 = es.step(0.70, 1.0, epoch=2)  # improvement of 0.10 > min_delta
    assert result2.improved is True


def test_patience_triggers_stop():
    es = EarlyStopping(patience=2)
    es.step(0.6, 1.0, epoch=0)  # improved, counter=0
    r1 = es.step(0.5, 1.0, epoch=1)  # no improve, counter=1
    assert r1.should_stop is False
    r2 = es.step(0.5, 1.0, epoch=2)  # no improve, counter=2 == patience
    assert r2.should_stop is True


def test_improvement_resets_patience_counter():
    es = EarlyStopping(patience=2)
    es.step(0.6, 1.0, epoch=0)
    es.step(0.5, 1.0, epoch=1)  # counter=1
    r = es.step(0.7, 1.0, epoch=2)  # improves, counter resets to 0
    assert r.improved is True
    assert es.epochs_without_improvement == 0


def test_state_dict_round_trip():
    es = EarlyStopping(patience=5, min_delta=0.01)
    es.step(0.6, 1.0, epoch=0)
    es.step(0.5, 1.0, epoch=1)
    state = es.state_dict()

    es2 = EarlyStopping(patience=1, min_delta=0.0)  # different initial config
    es2.load_state_dict(state)

    assert es2.best_f1 == es.best_f1
    assert es2.best_loss == es.best_loss
    assert es2.best_epoch == es.best_epoch
    assert es2.epochs_without_improvement == es.epochs_without_improvement
    assert es2.patience == es.patience
    assert es2.min_delta == es.min_delta

    # Continuing from the restored state behaves identically to the original.
    r_original = es.step(0.5, 1.0, epoch=2)
    r_restored = es2.step(0.5, 1.0, epoch=2)
    assert r_original.should_stop == r_restored.should_stop
    assert r_original.improved == r_restored.improved


def test_invalid_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        EarlyStopping(patience=3, mode="min")
