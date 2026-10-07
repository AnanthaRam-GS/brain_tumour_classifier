"""Repository/data-root path resolution and contract config loading.

Paths in configs/contract/*.yaml are relative to the DATA ROOT (see
docs/DECISIONS.md D2/D3): normally the repository root, but overridable via
BTDL_DATA_ROOT for environments (Colab, Kaggle) where the data lives
elsewhere than the code.
"""

import hashlib
import os
from pathlib import Path
from types import MappingProxyType

import yaml

_PHASE2_DIR_NAME = "phase2"


def repo_root() -> Path:
    """The directory containing phase2/, derived from this package's own location."""

    # phase2/src/btdl/config.py -> btdl -> src -> phase2 -> repo root
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """BTDL_DATA_ROOT if set (must exist), otherwise repo_root()."""

    override = os.environ.get("BTDL_DATA_ROOT")
    if override is None:
        return repo_root()
    path = Path(override).expanduser()
    if not path.is_dir():
        raise ValueError(f"BTDL_DATA_ROOT does not exist or is not a directory: {path}")
    return path.resolve()


def resolve_data_path(rel) -> Path:
    """A path under data_root() for a path relative to the data root."""

    return data_root() / Path(rel)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


# One explicit schema per contract file name. Each entry is a list of
# (key_path, expected_type) checks against the raw (pre-freeze) dict. No
# silent defaults: every key listed here must be present and of the given
# type, or load_contract raises.
_SCHEMAS = {
    "data": [
        ("contract_version", str),
        ("raw_dir", str),
        ("samples_dir", str),
        ("split_csv", str),
        ("split_sha256", str),
        ("manifest", str),
        ("expected", dict),
        ("expected.n_samples", int),
        ("expected.n_patients", int),
        ("expected.class_counts", dict),
        ("expected.split_samples", dict),
        ("expected.split_samples.train", int),
        ("expected.split_samples.val", int),
        ("expected.split_samples.test", int),
        ("expected.split_patients", dict),
        ("expected.split_patients.train", int),
        ("expected.split_patients.val", int),
        ("expected.split_patients.test", int),
    ],
    "input": [
        ("contract_version", str),
        ("source_field", str),
        ("roi", dict),
        ("roi.bbox", str),
        ("roi.padding_fraction", (int, float)),
        ("roi.padding_basis", str),
        ("roi.padding_rounding", str),
        ("roi.padding_per_side", bool),
        ("roi.square", str),
        ("roi.center_rounding", str),
        ("roi.out_of_bounds", str),
        ("roi.apply_mask", bool),
        ("resize", dict),
        ("resize.size", list),
        ("resize.mode", str),
        ("resize.antialias", bool),
        ("resize.align_corners", bool),
        ("resize.backend", str),
        ("resize.clamp", list),
        ("cache", dict),
        ("cache.dtype", str),
        ("cache.name", str),
        ("model_input", dict),
        ("model_input.channels", int),
        ("model_input.channel_policy", str),
        ("model_input.normalization", str),
        ("model_input.mean", list),
        ("model_input.std", list),
    ],
    "augmentation": [
        ("contract_version", str),
        ("applies_to", str),
        ("order", list),
        ("hflip", dict),
        ("hflip.p", (int, float)),
        ("affine", dict),
        ("affine.degrees", (int, float)),
        ("affine.translate", (int, float)),
        ("affine.scale", list),
        ("affine.shear", (int, float)),
        ("affine.interpolation", str),
        ("affine.fill", (int, float)),
        ("brightness", dict),
        ("brightness.factor_range", list),
        ("contrast", dict),
        ("contrast.factor_range", list),
        ("clamp", list),
    ],
}


def _lookup(payload, key_path):
    node = payload
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"contract is missing required key {key_path!r}")
        node = node[part]
    return node


def _validate_schema(name, payload):
    schema = _SCHEMAS.get(name)
    if schema is None:
        raise ValueError(f"no schema registered for contract {name!r}")
    for key_path, expected_type in schema:
        value = _lookup(payload, key_path)
        allowed_types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
        type_names = "/".join(t.__name__ for t in allowed_types)
        if bool not in allowed_types and int in allowed_types and isinstance(value, bool):
            raise ValueError(f"contract key {key_path!r} must be {type_names}, got bool")
        if not isinstance(value, allowed_types):
            raise ValueError(
                f"contract key {key_path!r} must be {type_names}, "
                f"got {type(value).__name__}: {value!r}"
            )


def load_contract(name):
    """Load configs/contract/<name>.yaml, validate its schema, return an immutable mapping."""

    path = repo_root() / _PHASE2_DIR_NAME / "configs" / "contract" / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"contract file not found: {path}")
    with path.open("r") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"contract {name!r} must parse to a mapping, got {type(payload).__name__}")
    _validate_schema(name, payload)
    return _freeze(payload)


def contract_dir_sha256() -> str:
    """Deterministic SHA-256 over every file in configs/contract/.

    Hashes sorted relative paths and file bytes, in that fixed order, so the
    result is stable across runs/machines and changes if any file's name,
    presence, or content changes.
    """

    contract_dir = repo_root() / _PHASE2_DIR_NAME / "configs" / "contract"
    digest = hashlib.sha256()
    file_paths = sorted(
        path for path in contract_dir.rglob("*") if path.is_file()
    )
    for path in file_paths:
        relative = path.relative_to(contract_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
