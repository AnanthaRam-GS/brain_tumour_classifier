import pandas as pd
import pytest

from btdl import config
from btdl.data.manifest import verify_manifest
from btdl.data.split import load_split, validate_split

def _real_data_available():
    contract = config.load_contract("data")
    samples_dir = config.resolve_data_path(contract["samples_dir"])
    return samples_dir.is_dir() and any(samples_dir.glob("*.npz"))


pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not _real_data_available(), reason="real NPZ dataset not present on disk"),
]


def _load_committed_manifest():
    contract = config.load_contract("data")
    manifest_path = config.repo_root() / contract["manifest"]
    return pd.read_csv(manifest_path, dtype={"sample_id": str, "patient_id": str})


@pytest.mark.slow
def test_committed_manifest_matches_local_data():
    manifest = _load_committed_manifest()
    data_root = config.data_root()
    mismatches = verify_manifest(manifest, data_root)
    assert mismatches.empty, f"manifest content_sha256 mismatches: {mismatches.to_dict('records')}"


def test_committed_manifest_matches_expected_counts():
    manifest = _load_committed_manifest()
    contract = config.load_contract("data")
    expected = contract["expected"]
    assert len(manifest) == expected["n_samples"]
    assert manifest["patient_id"].nunique() == expected["n_patients"]


def test_real_split_validates_against_committed_manifest():
    manifest = _load_committed_manifest()
    split_index = load_split()
    assert validate_split(split_index, manifest) is True
