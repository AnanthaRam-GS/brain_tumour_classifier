"""Repository/data-root path resolution and contract config loading.

Paths in configs/contract/*.yaml are relative to the DATA ROOT (see
docs/DECISIONS.md D2/D3): normally the repository root, but overridable via
BTDL_DATA_ROOT for environments (Colab, Kaggle) where the data lives
elsewhere than the code.
"""

import hashlib
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import yaml

_PHASE2_DIR_NAME = "phase2"


class FrozenDict(Mapping):
    """Immutable, picklable mapping.

    types.MappingProxyType is immutable but NOT picklable, which breaks
    torch DataLoader worker processes (num_workers > 0) when a Dataset
    stores a loaded contract as an instance attribute -- the Dataset object
    has to be pickled to reach each worker. This is a plain dict under the
    hood (picklable via __reduce__) with no mutation methods exposed.
    """

    __slots__ = ("_data",)

    def __init__(self, data):
        object.__setattr__(self, "_data", dict(data))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"FrozenDict({self._data!r})"

    def __reduce__(self):
        return (FrozenDict, (self._data,))


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


def cache_dir() -> Path:
    """BTDL_CACHE_DIR if set (must exist), otherwise phase2/cache/ under repo_root().

    roi_cache.py (read) and cli/build_cache.py (write) both resolve the ROI
    cache array through this function, so a teammate can point BTDL_CACHE_DIR
    at a Drive-mounted or pre-downloaded cache directory without touching
    code.
    """

    override = os.environ.get("BTDL_CACHE_DIR")
    if override is None:
        return repo_root() / _PHASE2_DIR_NAME / "cache"
    path = Path(override).expanduser()
    if not path.is_dir():
        raise ValueError(f"BTDL_CACHE_DIR does not exist or is not a directory: {path}")
    return path.resolve()


def git_dirty_paths(root=None) -> list:
    """Offending paths under the Phase 2 dirty-tree definition.

    dirty = any TRACKED file modified/staged/deleted anywhere in the repo,
    OR any untracked, non-ignored file under phase2/. An untracked file
    outside phase2/ (e.g. the user's own reference documents at the repo
    root) does not count -- `git status --porcelain` never lists ignored
    files, so every untracked line here is already non-ignored.
    """

    root = Path(root) if root is not None else repo_root()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    offending = []
    for line in status.splitlines():
        if not line:
            continue
        code, rest = line[:2], line[3:]
        path = rest.split(" -> ", 1)[-1] if " -> " in rest else rest
        if code == "??":
            if path == _PHASE2_DIR_NAME or path.startswith(_PHASE2_DIR_NAME + "/"):
                offending.append(path)
        else:
            offending.append(path)
    return offending


def git_is_dirty(root=None) -> bool:
    return bool(git_dirty_paths(root))


def _freeze(value):
    if isinstance(value, dict):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
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
    "training": [
        ("contract_version", str),
        ("batch_size", int),
        ("eval_batch_size", int),
        ("max_epochs", int),
        ("num_workers", int),
        ("precision", str),
        ("optimizer", dict),
        ("optimizer.name", str),
        ("optimizer.weight_decay", (int, float)),
        ("optimizer.betas", list),
        ("optimizer.eps", (int, float)),
        ("lr_grid", list),
        ("grid_seed", int),
        ("final_seeds", list),
        ("scheduler", dict),
        ("scheduler.name", str),
        ("scheduler.t_max", (int, str)),
        ("scheduler.eta_min", (int, float)),
        ("scheduler.step", str),
        ("loss", dict),
        ("loss.name", str),
        ("loss.class_weights", str),
        ("early_stopping", dict),
        ("early_stopping.monitor", str),
        ("early_stopping.mode", str),
        ("early_stopping.patience", int),
        ("early_stopping.min_delta", (int, float)),
        ("early_stopping.tie_break", str),
        ("selection", str),
    ],
    "evaluation": [
        ("contract_version", str),
        ("results_schema_version", str),
        ("bootstrap", dict),
        ("bootstrap.n_resamples", int),
        ("bootstrap.seed", int),
        ("bootstrap.alpha", (int, float)),
        ("efficiency", dict),
        ("efficiency.batch_sizes", list),
        ("efficiency.warmup", int),
        ("efficiency.iters", int),
        ("size_tertiles", str),
        ("reload_equivalence_atol", (int, float)),
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


_MODEL_CONFIG_ALLOWED_KEYS = {"name", "version", "weights_id", "reference_only", "description"}
_MODEL_CONFIG_TYPES = {
    "name": str,
    "version": str,
    "weights_id": (str, type(None)),
    "reference_only": bool,
    "description": str,
}


def load_model_config(name):
    """Load configs/models/<name>.yaml with a STRICT key whitelist (no hyperparameters)."""

    path = repo_root() / _PHASE2_DIR_NAME / "configs" / "models" / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"model config file not found: {path}")
    with path.open("r") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"model config {name!r} must parse to a mapping, got {type(payload).__name__}")

    unknown_keys = set(payload.keys()) - _MODEL_CONFIG_ALLOWED_KEYS
    if unknown_keys:
        raise ValueError(
            f"model config {name!r} has unknown keys (hyperparameters do not belong in a "
            f"model config): {sorted(unknown_keys)}"
        )
    missing_keys = _MODEL_CONFIG_ALLOWED_KEYS - set(payload.keys())
    if missing_keys:
        raise ValueError(f"model config {name!r} is missing required keys: {sorted(missing_keys)}")

    for key, expected_type in _MODEL_CONFIG_TYPES.items():
        value = payload[key]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"model config {name!r} key {key!r} must be {expected_type}, got {type(value).__name__}: {value!r}"
            )

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
