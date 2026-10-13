"""Tests for the read-only walkthrough reporting helpers (btdl.walkthrough),
on synthetic fixtures, plus an AST guard that the module never constructs a
test-split dataset."""

import ast
import json
import subprocess

import pytest

from btdl import config, walkthrough

# ---- AST guard: walkthrough.py never calls RoiDataset with split="test" or allow_test=True --


def _roi_dataset_test_split_offenses(source: str):
    tree = ast.parse(source)
    offenses = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
        if name != "RoiDataset":
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "test":
            offenses.append(("positional split='test'", node.lineno))
        for kw in node.keywords:
            if kw.arg == "split" and isinstance(kw.value, ast.Constant) and kw.value.value == "test":
                offenses.append(("keyword split='test'", node.lineno))
            if kw.arg == "allow_test" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                offenses.append(("allow_test=True", node.lineno))
    return offenses


def test_walkthrough_never_constructs_test_split_dataset():
    path = config.repo_root() / "phase2" / "src" / "btdl" / "walkthrough.py"
    offenses = _roi_dataset_test_split_offenses(path.read_text())
    assert not offenses, f"walkthrough.py must never touch the test split: {offenses}"


def _enclosing_function_name(tree, target_node):
    """The name of the nearest ast.FunctionDef containing target_node, or
    None if target_node is at module level."""

    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and target_node in ast.walk(node):
            if best is None or node.lineno > best.lineno:
                best = node
    return best.name if best is not None else None


def test_run_final_test_called_only_inside_run_official_final_test():
    # run_final_test is the ONE function allowed to set allow_test=True (in
    # its own real-mode branch) -- walkthrough.py may call it, but only
    # from the one explicitly gated place.
    path = config.repo_root() / "phase2" / "src" / "btdl" / "walkthrough.py"
    source = path.read_text()
    tree = ast.parse(source)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
        if name != "run_final_test":
            continue
        enclosing = _enclosing_function_name(tree, node)
        if enclosing != "run_official_final_test":
            offenders.append((enclosing, node.lineno))
    assert not offenders, f"run_final_test must only be called inside run_official_final_test: {offenders}"

    # Sanity: confirm it DOES appear inside run_official_final_test (the test isn't vacuous).
    assert "run_final_test(" in source


def test_run_official_final_test_begins_with_the_confirm_check():
    path = config.repo_root() / "phase2" / "src" / "btdl" / "walkthrough.py"
    tree = ast.parse(path.read_text())
    func = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run_official_final_test")

    body = func.body
    # Skip a leading docstring Expr, if present.
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]

    assert body, "run_official_final_test has no statements after its docstring"
    first = body[0]
    assert isinstance(first, ast.If), f"first statement must be the confirm check (an `if`), got {type(first).__name__}"

    test_source = ast.dump(first.test)
    assert "confirm" in test_source, f"first `if` must test `confirm`, got: {test_source}"

    raises_in_body = any(isinstance(stmt, ast.Raise) for stmt in ast.walk(first))
    assert raises_in_body, "the confirm check's `if` body must raise"


# ---- 1. dataset_overview ------------------------------------------------------------------


def test_dataset_overview_reports_tables_and_overlap_check(fake_repo):
    # The tiny synthetic fixture doesn't have every class in every split, so
    # validate_split() legitimately raises here (it checks more than just
    # patient overlap) -- what matters is that dataset_overview() surfaces
    # that as a readable False/detail pair rather than crashing the notebook.
    result = walkthrough.dataset_overview()
    assert isinstance(result["patient_overlap_ok"], bool)
    assert isinstance(result["patient_overlap_detail"], str)
    assert result["n_samples"] == 8
    assert set(result["split_table"].index) == {"train", "val", "test"}
    assert result["split_table"].loc["train", "n_samples"] == 4
    assert result["split_table"].loc["train", "n_patients"] == 2


def test_dataset_overview_overlap_ok_true_on_the_real_split():
    # The real, frozen patient_split.csv (not the synthetic fixture) has no
    # missing classes and no patient overlap -- confirms the happy path.
    result = walkthrough.dataset_overview()
    assert result["patient_overlap_ok"] is True


# ---- 2. show_roi_pipeline -------------------------------------------------------------------


def test_show_roi_pipeline_refuses_test_split_sample(fake_repo):
    split_index = __import__("btdl.data.split", fromlist=["load_split"]).load_split()
    test_sample = next(sid for sid, row in split_index.items() if row["split"] == "test")
    with pytest.raises(walkthrough.WalkthroughError, match="test split"):
        walkthrough.show_roi_pipeline(test_sample)


def test_show_roi_pipeline_on_train_sample_returns_tensor_stats(fake_repo):
    split_index = __import__("btdl.data.split", fromlist=["load_split"]).load_split()
    train_sample = next(sid for sid, row in split_index.items() if row["split"] == "train")
    result = walkthrough.show_roi_pipeline(train_sample)
    assert result["split"] == "train"
    assert result["cache_only_checkout"] is False
    assert result["model_input_tensor_shape"] == (3, 224, 224)
    assert "mean" in result["model_input_tensor_stats"]


def test_show_roi_pipeline_unknown_sample_raises(fake_repo):
    with pytest.raises(walkthrough.WalkthroughError, match="not in the split index"):
        walkthrough.show_roi_pipeline("does-not-exist")


# ---- 3. show_qa_sheets -----------------------------------------------------------------------


def test_show_qa_sheets_reports_missing_when_absent(fake_repo):
    tmp_path, manifest, specs = fake_repo
    sheets = walkthrough.show_qa_sheets()
    assert sheets["augmentation_sheet.png"] is None
    assert sheets["roi_contact_sheet.png"] is None


def test_show_qa_sheets_reports_path_when_present(fake_repo):
    tmp_path, manifest, specs = fake_repo
    qa_dir = tmp_path / "phase2" / "artifacts" / "qa"
    qa_dir.mkdir(parents=True)
    (qa_dir / "augmentation_sheet.png").write_bytes(b"fake-png")
    sheets = walkthrough.show_qa_sheets()
    assert sheets["augmentation_sheet.png"] == qa_dir / "augmentation_sheet.png"
    assert sheets["roi_contact_sheet.png"] is None


# ---- 4. model_overview -----------------------------------------------------------------------


def test_model_overview_default_does_not_instantiate():
    result = walkthrough.model_overview("tiny_cnn")
    assert result["name"] == "tiny_cnn"
    assert result["reference_only"] is True
    assert "total_parameters" not in result


def test_model_overview_instantiate_true_counts_parameters():
    result = walkthrough.model_overview("tiny_cnn", instantiate=True)
    assert result["total_parameters"] > 0
    assert result["trainable_parameters"] > 0
    assert "Conv2d" in result["module_tree"] or "Linear" in result["module_tree"]


def test_model_overview_unknown_model_raises():
    with pytest.raises(ValueError):
        walkthrough.model_overview("not_a_real_model")


# ---- test_access_status (lightweight -- JSON-only, no real run dirs) -------------------------


def _git(args, cwd):
    subprocess.run(["git"] + args, cwd=cwd, check=True, capture_output=True, text=True)


def test_test_access_status_reports_registered_not_logged(fake_repo):
    tmp_path, manifest, specs = fake_repo
    (tmp_path / ".gitignore").write_text("phase2/runs/\n")
    _git(["init"], cwd=tmp_path)
    _git(["config", "user.email", "test@example.com"], cwd=tmp_path)
    _git(["config", "user.name", "Test User"], cwd=tmp_path)

    frozen_dir = tmp_path / "phase2" / "artifacts" / "frozen"
    frozen_dir.mkdir(parents=True)
    (frozen_dir / "tiny_cnn.json").write_text(
        json.dumps(
            {
                "model_name": "tiny_cnn",
                "seeds": [
                    {"seed": 42, "run_dir": "phase2/runs/tiny_cnn/lr1e-04_seed42"},
                    {"seed": 43, "run_dir": "phase2/runs/tiny_cnn/lr1e-04_seed43"},
                    {"seed": 44, "run_dir": "phase2/runs/tiny_cnn/lr1e-04_seed44"},
                ],
            }
        )
    )
    _git(["add", "-A"], cwd=tmp_path)
    _git(["commit", "-m", "register"], cwd=tmp_path)

    result = walkthrough.test_access_status("tiny_cnn")
    assert result["per_seed"][42]["registered"] is True
    assert result["per_seed"][42]["logged"] is False
    assert result["per_seed"][42]["exported"] is False


def test_test_access_status_unregistered_model(fake_repo):
    result = walkthrough.test_access_status("tiny_cnn")
    for seed_status in result["per_seed"].values():
        assert seed_status["registered"] is False
        assert seed_status["logged"] is False


# ---- phase1_comparison_doc ---------------------------------------------------------------------


def test_phase1_comparison_doc_not_available_when_missing(fake_repo):
    result = walkthrough.phase1_comparison_doc("tiny_cnn")
    assert result["available"] is False


def test_phase1_comparison_doc_reads_matching_file(fake_repo):
    tmp_path, manifest, specs = fake_repo
    docs_dir = tmp_path / "phase2" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "wavelet_rf_vs_tiny_cnn.md").write_text("# comparison\n\nsome numbers\n")

    result = walkthrough.phase1_comparison_doc("tiny_cnn")
    assert result["available"] is True
    assert result["content"] == "# comparison\n\nsome numbers\n"
    assert result["path"].name == "wavelet_rf_vs_tiny_cnn.md"


def test_phase1_comparison_doc_does_not_match_a_different_model(fake_repo):
    tmp_path, manifest, specs = fake_repo
    docs_dir = tmp_path / "phase2" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "wavelet_rf_vs_some_other_model.md").write_text("not this one\n")

    result = walkthrough.phase1_comparison_doc("tiny_cnn")
    assert result["available"] is False


# ---- split_results / protocol_overview / example_predictions (need real runs: slow) ----------


@pytest.fixture
def three_seed_runs(git_fake_repo):
    """tiny_cnn trained (via fit() directly, not the train CLI) for all
    three of training.yaml's final_seeds at lr_grid[0], each evaluated on
    val, plus a hand-written selection file pointing at that lr -- enough
    for protocol_overview/split_results("val") without running a real LR
    grid search."""

    import torch

    from btdl.cli.evaluate import run_evaluate
    from btdl.data.class_weights import compute_class_weights
    from btdl.data.dataset import RoiDataset
    from btdl.data.loader import make_loader
    from btdl.models.registry import build_model
    from btdl.runs import run_dir_for
    from btdl.training.device import select_device
    from btdl.training.trainer import fit

    tmp_path, manifest, specs = git_fake_repo
    cfg = config.load_contract("training")
    lr = cfg["lr_grid"][0]
    device = select_device("cpu")

    run_dirs = {}
    for seed in cfg["final_seeds"]:
        train_ds = RoiDataset("train", augment=True, seed=seed)
        val_ds = RoiDataset("val", augment=False, seed=seed)
        train_loader, train_sampler = make_loader(
            train_ds, batch_size=2, shuffle=True, seed=seed, num_workers=0, device_type="cpu"
        )
        val_loader, _ = make_loader(
            val_ds, batch_size=2, shuffle=False, seed=seed, num_workers=0, device_type="cpu"
        )
        class_weights = compute_class_weights(train_ds)
        torch.manual_seed(seed)
        model = build_model("tiny_cnn", pretrained=False)
        run_dir = run_dir_for("tiny_cnn", lr, seed, smoke=False)
        fit(
            model,
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            class_weights=class_weights,
            cfg=cfg,
            lr=lr,
            seed=seed,
            run_dir=run_dir,
            device=device,
            model_name="tiny_cnn",
        )
        run_evaluate(run_dir, split="val", device=device, batch_size=2, num_workers=0)
        run_dirs[seed] = run_dir

    selection_dir = tmp_path / "phase2" / "artifacts" / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_path = selection_dir / "tiny_cnn.json"
    selection_path.write_text(
        json.dumps(
            {
                "model_name": "tiny_cnn",
                "candidates": [
                    {
                        "lr": lr,
                        "best_epoch": 0,
                        "best_pt_sha256": "x",
                        "best_val_loss": 0.5,
                        "best_val_macro_f1": 0.5,
                        "run_dir": str(run_dirs[cfg["grid_seed"]]),
                    }
                ],
                "selected_lr": lr,
                "selected_run_dir": str(run_dirs[cfg["grid_seed"]]),
                "contract_dir_sha256": config.contract_dir_sha256(),
                "git_commit": "x",
            }
        )
    )

    return tmp_path, run_dirs, device


@pytest.mark.slow
def test_protocol_overview_reads_selection_and_seed_table(three_seed_runs):
    tmp_path, run_dirs, device = three_seed_runs
    result = walkthrough.protocol_overview("tiny_cnn")
    assert len(result["grid_table"]) == 1
    assert len(result["seed_table"]) == 3
    assert set(result["training_curves"].keys()) == {42, 43, 44}
    assert "best_val_macro_f1" in result["seed_summary"]


@pytest.mark.slow
def test_split_results_val_available_for_all_seeds(three_seed_runs):
    tmp_path, run_dirs, device = three_seed_runs
    result = walkthrough.split_results("tiny_cnn", "val")
    assert result["available"] is True
    assert result["missing_seeds"] == []
    assert set(result["per_seed"].keys()) == {42, 43, 44}
    assert "headline" in result["aggregates"]
    assert "calibration" in result["aggregates"]


@pytest.mark.slow
def test_split_results_rehearsal_val_not_available_yet(three_seed_runs):
    result = walkthrough.split_results("tiny_cnn", "rehearsal_val")
    assert result["available"] is False


@pytest.mark.slow
def test_split_results_test_not_available_yet(three_seed_runs):
    result = walkthrough.split_results("tiny_cnn", "test")
    assert result["available"] is False


def test_split_results_invalid_split_raises(fake_repo):
    with pytest.raises(walkthrough.WalkthroughError, match="split must be one of"):
        walkthrough.split_results("tiny_cnn", "bogus")


@pytest.mark.slow
def test_example_predictions_reads_val_predictions(three_seed_runs):
    result = walkthrough.example_predictions("tiny_cnn", "val", n=3)
    assert result["available"] is True
    assert 0 < len(result["examples"]) <= 3
    for example in result["examples"]:
        assert "sample_id" in example["row"]


@pytest.mark.slow
def test_example_predictions_not_available_yet(three_seed_runs):
    # rehearsal_val has never been run for these synthetic runs.
    result = walkthrough.example_predictions("tiny_cnn", "rehearsal_val", n=3)
    assert result["available"] is False
