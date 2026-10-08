from btdl.runs import format_lr, run_dir_for


def test_format_lr_canonical():
    assert format_lr(1e-3) == "1e-03"
    assert format_lr(3e-4) == "3e-04"
    assert format_lr(1e-4) == "1e-04"


def test_run_dir_for_non_smoke(fake_repo):
    tmp_path, manifest, specs = fake_repo
    run_dir = run_dir_for("tiny_cnn", 1e-3, 42, smoke=False)
    assert run_dir == tmp_path / "phase2" / "runs" / "tiny_cnn" / "lr1e-03_seed42"


def test_run_dir_for_smoke(fake_repo):
    tmp_path, manifest, specs = fake_repo
    run_dir = run_dir_for("tiny_cnn", 1e-3, 42, smoke=True)
    assert run_dir == tmp_path / "phase2" / "runs" / "_smoke" / "tiny_cnn" / "lr1e-03_seed42"


def test_run_dir_for_different_lr_seed_gives_different_dirs(fake_repo):
    tmp_path, manifest, specs = fake_repo
    a = run_dir_for("tiny_cnn", 1e-4, 42, smoke=False)
    b = run_dir_for("tiny_cnn", 3e-4, 42, smoke=False)
    c = run_dir_for("tiny_cnn", 1e-4, 43, smoke=False)
    assert a != b != c != a
