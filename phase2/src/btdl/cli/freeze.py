"""python -m btdl.cli.freeze --run-dir R

Requires best.pt and metadata.json with contract_conformant=true and
git_dirty=false (as recorded in metadata at training time). Writes
R/FROZEN.json {best_pt_sha256, frozen_at_commit, contract_dir_sha256}.
Refuses if the run is already frozen.
"""

import argparse
import hashlib
import json
from pathlib import Path


class FreezeError(ValueError):
    """Raised when a run cannot be frozen."""


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def run_freeze(run_dir) -> dict:
    run_dir = Path(run_dir)
    best_path = run_dir / "best.pt"
    frozen_path = run_dir / "FROZEN.json"
    metadata_path = run_dir / "metadata.json"

    if frozen_path.is_file():
        raise FreezeError(f"{run_dir} is already frozen ({frozen_path} exists)")
    if not best_path.is_file():
        raise FreezeError(f"best.pt not found in {run_dir}")
    if not metadata_path.is_file():
        raise FreezeError(f"metadata.json not found in {run_dir}")

    with metadata_path.open() as handle:
        metadata = json.load(handle)

    if metadata.get("contract_conformant") is not True:
        raise FreezeError(
            f"refusing to freeze {run_dir}: metadata.contract_conformant is "
            f"{metadata.get('contract_conformant')!r}, not True"
        )
    if metadata.get("git_dirty") is not False:
        raise FreezeError(
            f"refusing to freeze {run_dir}: metadata.git_dirty is {metadata.get('git_dirty')!r}, not False"
        )

    frozen = {
        "best_pt_sha256": _file_sha256(best_path),
        "frozen_at_commit": metadata.get("git_commit"),
        "contract_dir_sha256": metadata.get("contract_dir_sha256"),
    }
    with frozen_path.open("w") as handle:
        json.dump(frozen, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    frozen = run_freeze(args.run_dir)
    print(json.dumps(frozen, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
