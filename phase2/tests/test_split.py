import hashlib

import pandas as pd
import pytest
import yaml

from btdl.data.split import SplitIntegrityError, SplitIndex, validate_split


def _write_contract(tmp_path, split_csv_path, *, split_samples=None, split_patients=None):
    split_samples = split_samples or {"train": 2, "val": 1, "test": 1}
    split_patients = split_patients or {"train": 1, "val": 1, "test": 1}
    contract_dir = tmp_path / "phase2" / "configs" / "contract"
    contract_dir.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256(split_csv_path.read_bytes()).hexdigest()
    payload = {
        "contract_version": "1.0.0",
        "raw_dir": "1512427",
        "samples_dir": "data/processed/samples",
        "split_csv": str(split_csv_path.relative_to(tmp_path)),
        "split_sha256": sha256,
        "manifest": "phase2/artifacts/contract/manifest.csv",
        "expected": {
            "n_samples": sum(split_samples.values()),
            "n_patients": sum(split_patients.values()),
            "class_counts": {1: 2, 2: 1, 3: 1},
            "split_samples": split_samples,
            "split_patients": split_patients,
        },
    }
    with (contract_dir / "data.yaml").open("w") as handle:
        yaml.safe_dump(payload, handle)
    return sha256


def _write_split_csv(path, rows):
    pd.DataFrame(rows, columns=["sample_id", "patient_id", "label", "split"]).to_csv(
        path, index=False
    )


VALID_ROWS = [
    ("1", "p1", 1, "train"),
    ("2", "p1", 2, "train"),
    ("3", "p1", 3, "train"),
    ("4", "p2", 1, "val"),
    ("5", "p2", 2, "val"),
    ("6", "p2", 3, "val"),
    ("7", "p3", 1, "test"),
    ("8", "p3", 2, "test"),
    ("9", "p3", 3, "test"),
]


def _manifest_from_rows(rows):
    return pd.DataFrame(
        [{"sample_id": r[0], "patient_id": r[1], "label": r[2]} for r in rows]
    )


# ---- load_split: hash verification -----------------------------------------------


def test_load_split_rejects_hash_mismatch(tmp_path, monkeypatch):
    split_csv = tmp_path / "split.csv"
    _write_split_csv(split_csv, VALID_ROWS)
    _write_contract(tmp_path, split_csv)

    # Mutate the file after the contract hash was computed from the original content.
    _write_split_csv(split_csv, VALID_ROWS + [("5", "p4", 2, "train")])

    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)
    from btdl.data.split import load_split

    with pytest.raises(SplitIntegrityError):
        load_split()


def test_load_split_succeeds_on_matching_hash(tmp_path, monkeypatch):
    split_csv = tmp_path / "split.csv"
    _write_split_csv(split_csv, VALID_ROWS)
    _write_contract(
        tmp_path,
        split_csv,
        split_samples={"train": 3, "val": 3, "test": 3},
        split_patients={"train": 1, "val": 1, "test": 1},
    )
    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)
    from btdl.data.split import load_split

    split_index = load_split()
    assert len(split_index) == 9
    assert split_index["1"]["split"] == "train"


# ---- SplitIndex.sample_ids ---------------------------------------------------------


def test_sample_ids_returns_sorted_tuple():
    rows = {sid: {"split": split, "patient_id": pid, "label": label} for sid, pid, label, split in VALID_ROWS}
    index = SplitIndex(rows)
    assert index.sample_ids("train") == ("1", "2", "3")


def test_sample_ids_rejects_invalid_split_name():
    rows = {sid: {"split": split, "patient_id": pid, "label": label} for sid, pid, label, split in VALID_ROWS}
    index = SplitIndex(rows)
    with pytest.raises(ValueError):
        index.sample_ids("bogus")


# ---- validate_split -----------------------------------------------------------------


def _index_from_rows(rows):
    return SplitIndex(
        {sid: {"split": split, "patient_id": pid, "label": label} for sid, pid, label, split in rows}
    )


def test_validate_split_passes_on_consistent_data(tmp_path, monkeypatch):
    (tmp_path / "unused.csv").write_text("x")
    _write_contract(
        tmp_path,
        tmp_path / "unused.csv",
        split_samples={"train": 3, "val": 3, "test": 3},
        split_patients={"train": 1, "val": 1, "test": 1},
    )
    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)

    index = _index_from_rows(VALID_ROWS)
    manifest = _manifest_from_rows(VALID_ROWS)
    assert validate_split(index, manifest) is True


def test_validate_split_rejects_missing_sample_in_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)
    index = _index_from_rows(VALID_ROWS)
    manifest = _manifest_from_rows(VALID_ROWS[:-1])  # drop sample "4"
    with pytest.raises(SplitIntegrityError):
        validate_split(index, manifest)


def test_validate_split_rejects_extra_sample_in_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)
    index = _index_from_rows(VALID_ROWS[:-1])
    manifest = _manifest_from_rows(VALID_ROWS)  # extra sample "4"
    with pytest.raises(SplitIntegrityError):
        validate_split(index, manifest)


def test_validate_split_rejects_label_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)
    index = _index_from_rows(VALID_ROWS)
    bad_rows = list(VALID_ROWS)
    bad_rows[0] = ("1", "p1", 2, "train")  # label changed from 1 to 2
    manifest = _manifest_from_rows(bad_rows)
    with pytest.raises(SplitIntegrityError):
        validate_split(index, manifest)


def test_validate_split_rejects_patient_overlap(tmp_path, monkeypatch):
    (tmp_path / "unused.csv").write_text("x")
    _write_contract(
        tmp_path,
        tmp_path / "unused.csv",
        split_samples={"train": 2, "val": 1, "test": 1},
        split_patients={"train": 1, "val": 1, "test": 1},
    )
    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)

    overlapping_rows = [
        ("1", "p1", 1, "train"),
        ("2", "p1", 2, "val"),  # p1 appears in both train and val
        ("3", "p2", 3, "val"),
        ("4", "p3", 1, "test"),
    ]
    index = _index_from_rows(overlapping_rows)
    manifest = _manifest_from_rows(overlapping_rows)
    with pytest.raises(SplitIntegrityError):
        validate_split(index, manifest)


def test_validate_split_rejects_missing_class_in_a_split(tmp_path, monkeypatch):
    monkeypatch.setattr("btdl.config.repo_root", lambda: tmp_path)
    rows_missing_class = [
        ("1", "p1", 1, "train"),
        ("2", "p1", 1, "train"),  # train only has class 1, not 2 or 3
        ("3", "p2", 2, "val"),
        ("4", "p2", 3, "val"),
        ("5", "p3", 1, "test"),
        ("6", "p3", 2, "test"),
        ("7", "p3", 3, "test"),
    ]
    index = _index_from_rows(rows_missing_class)
    manifest = _manifest_from_rows(rows_missing_class)
    with pytest.raises(SplitIntegrityError):
        validate_split(index, manifest)
