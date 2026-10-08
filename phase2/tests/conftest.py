import hashlib
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from btdl import config
from btdl.cli.build_cache import build_cache
from btdl.data.manifest import build_manifest

FAKE_REPO_SPECS = [
    # (sample_id, patient_id, label, split, image_size)
    ("1", "p1", 1, "train", 30),
    ("2", "p1", 2, "train", 24),
    ("3", "p2", 3, "train", 18),
    ("4", "p2", 1, "train", 20),
    ("5", "p3", 2, "val", 22),
    ("6", "p4", 3, "val", 26),
    ("7", "p5", 1, "test", 28),
    ("8", "p6", 2, "test", 20),
]


def _write_npz(path, *, sample_id, patient_id, label, image, mask):
    np.savez(
        path,
        image_raw=image.astype(np.float32),
        image_normalized=image.astype(np.float32),
        tumor_mask=mask.astype(np.uint8),
        tumor_border=np.array([1.0, 2.0]),
        label=np.array(label),
        patient_id=np.array(patient_id),
        sample_id=np.array(sample_id),
    )


def _build_fake_repo(tmp_path, specs):
    data_root = tmp_path
    samples_dir = data_root / "data" / "processed" / "samples"
    samples_dir.mkdir(parents=True)
    raw_dir = data_root / "raw"
    raw_dir.mkdir()

    rng = np.random.RandomState(0)
    for sample_id, patient_id, label, _split, size in specs:
        image = rng.rand(size, size).astype(np.float32)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[size // 4 : size // 2, size // 4 : size // 2] = 1
        _write_npz(
            samples_dir / f"{sample_id}.npz",
            sample_id=sample_id,
            patient_id=patient_id,
            label=label,
            image=image,
            mask=mask,
        )
        (raw_dir / f"{sample_id}.mat").write_bytes(b"fake")

    split_csv = data_root / "split.csv"
    pd.DataFrame(
        {
            "sample_id": [s[0] for s in specs],
            "patient_id": [s[1] for s in specs],
            "label": [s[2] for s in specs],
            "split": [s[3] for s in specs],
        }
    ).to_csv(split_csv, index=False)

    manifest = build_manifest(
        samples_dir=samples_dir, raw_dir=raw_dir, split_csv=split_csv, data_root=data_root
    )
    artifacts_dir = tmp_path / "phase2" / "artifacts" / "contract"
    artifacts_dir.mkdir(parents=True)
    manifest_path = artifacts_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    contract_dir = tmp_path / "phase2" / "configs" / "contract"
    contract_dir.mkdir(parents=True)

    split_sha256 = hashlib.sha256(split_csv.read_bytes()).hexdigest()

    split_counts = {}
    patients_by_split = {}
    class_counts = {1: 0, 2: 0, 3: 0}
    for sample_id, patient_id, label, split, _size in specs:
        split_counts[split] = split_counts.get(split, 0) + 1
        patients_by_split.setdefault(split, set()).add(patient_id)
        class_counts[label] += 1
    for name in ("train", "val", "test"):
        split_counts.setdefault(name, 0)
        patients_by_split.setdefault(name, set())
    patient_counts = {name: len(ids) for name, ids in patients_by_split.items()}

    data_yaml = {
        "contract_version": "1.0.0",
        "raw_dir": "raw",
        "samples_dir": "data/processed/samples",
        "split_csv": "split.csv",
        "split_sha256": split_sha256,
        "manifest": "phase2/artifacts/contract/manifest.csv",
        "expected": {
            "n_samples": len(specs),
            "n_patients": len({s[1] for s in specs}),
            "class_counts": class_counts,
            "split_samples": split_counts,
            "split_patients": patient_counts,
        },
    }
    with (contract_dir / "data.yaml").open("w") as handle:
        yaml.safe_dump(data_yaml, handle)

    real_contract_dir = config.repo_root() / "phase2" / "configs" / "contract"
    shutil.copy(real_contract_dir / "input.yaml", contract_dir / "input.yaml")
    shutil.copy(real_contract_dir / "augmentation.yaml", contract_dir / "augmentation.yaml")
    shutil.copy(real_contract_dir / "training.yaml", contract_dir / "training.yaml")

    return manifest


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A tiny synthetic repo (8 samples, 6 patients, train/val/test) with a built
    ROI cache (incl. roi_geometry.csv) and size_tertiles.json, for Dataset/
    Sampler/Loader/class-weight/evaluation-CLI tests. Yields
    (tmp_path, manifest, specs)."""

    from btdl.cli.size_tertiles import build_and_write as build_size_tertiles

    manifest = _build_fake_repo(tmp_path, FAKE_REPO_SPECS)
    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)
    build_cache()
    build_size_tertiles()
    return tmp_path, manifest, FAKE_REPO_SPECS


def _git(args, cwd):
    import subprocess

    subprocess.run(["git"] + args, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def trained_run(fake_repo):
    """A real (minimal) git repo around the fake_repo fixture, with one toy
    model trained to a conformant run_dir under phase2/runs/ (gitignored, so
    the working tree reads as clean even after writing checkpoints) --
    for cli/evaluate.py, cli/freeze.py, cli/final_test.py tests.
    Yields (tmp_path, run_dir, model, device)."""

    import torch

    from btdl import config
    from btdl.data.class_weights import compute_class_weights
    from btdl.data.dataset import RoiDataset
    from btdl.data.loader import make_loader
    from btdl.training.device import select_device
    from btdl.training.trainer import fit
    from tests.toy_training import ToyModel

    tmp_path, manifest, specs = fake_repo

    (tmp_path / ".gitignore").write_text("phase2/runs/\n")
    _git(["init"], cwd=tmp_path)
    _git(["config", "user.email", "test@example.com"], cwd=tmp_path)
    _git(["config", "user.name", "Test User"], cwd=tmp_path)
    _git(["add", "-A"], cwd=tmp_path)
    _git(["commit", "-m", "init"], cwd=tmp_path)

    train_ds = RoiDataset("train", augment=True, seed=42)
    val_ds = RoiDataset("val", augment=False, seed=42)
    train_loader, train_sampler = make_loader(
        train_ds, batch_size=2, shuffle=True, seed=42, num_workers=0, device_type="cpu"
    )
    val_loader, _ = make_loader(val_ds, batch_size=2, shuffle=False, seed=42, num_workers=0, device_type="cpu")

    class_weights = compute_class_weights(train_ds)
    torch.manual_seed(0)
    model = ToyModel()
    device = select_device("cpu")
    cfg = config.load_contract("training")  # the REAL, unmodified contract -> conformant
    run_dir = tmp_path / "phase2" / "runs" / "run1"
    fit(
        model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        class_weights=class_weights,
        cfg=cfg,
        lr=cfg["lr_grid"][0],
        seed=42,
        run_dir=run_dir,
        device=device,
    )

    return tmp_path, run_dir, model, device
