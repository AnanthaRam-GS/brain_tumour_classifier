"""python -m btdl.cli.unblinding_status

Prints one row per study model: whether a selection file and a frozen
registration exist, and which final_seeds are registered. Overall READY
(exit 0) only when EVERY study model has a valid registration covering
every seed in training.yaml's final_seeds AND whose recorded
contract_dir_sha256 equals the current contract (exit 1 otherwise, "NOT
READY"). See D18 -- final_test (real mode) refuses to run until this is
READY.
"""

import json
import sys

from btdl import config
from btdl.cli.register_frozen import FROZEN_REGISTRATION_DIR_RELPATH


def compute_status() -> dict:
    repo_root = config.repo_root()
    study = config.load_study()
    training_cfg = config.load_contract("training")
    final_seeds = sorted(training_cfg["final_seeds"])
    current_contract_hash = config.contract_dir_sha256()

    rows = []
    for model_name in study["models"]:
        selection_path = repo_root / "phase2" / "artifacts" / "selection" / f"{model_name}.json"
        frozen_path = repo_root / FROZEN_REGISTRATION_DIR_RELPATH / f"{model_name}.json"

        selection_present = selection_path.is_file()
        frozen_present = frozen_path.is_file()
        seeds_registered = []
        contract_matches = False

        if frozen_present:
            with frozen_path.open() as handle:
                registration = json.load(handle)
            seeds_registered = sorted(entry["seed"] for entry in registration.get("seeds", []))
            contract_matches = registration.get("contract_dir_sha256") == current_contract_hash

        model_ready = frozen_present and contract_matches and seeds_registered == final_seeds
        rows.append(
            {
                "model_name": model_name,
                "selection_present": selection_present,
                "frozen_present": frozen_present,
                "seeds_registered": seeds_registered,
                "contract_matches": contract_matches,
                "ready": model_ready,
            }
        )

    overall_ready = len(rows) > 0 and all(row["ready"] for row in rows)
    return {"ready": overall_ready, "final_seeds": final_seeds, "models": rows}


def main():
    status = compute_status()

    header = f"{'model':<20}{'selection':<12}{'frozen':<10}{'seeds':<20}{'contract_ok':<14}{'ready'}"
    print(header)
    for row in status["models"]:
        print(
            f"{row['model_name']:<20}"
            f"{str(row['selection_present']):<12}"
            f"{str(row['frozen_present']):<10}"
            f"{str(row['seeds_registered']):<20}"
            f"{str(row['contract_matches']):<14}"
            f"{row['ready']}"
        )
    print()
    print("READY" if status["ready"] else "NOT READY")
    sys.exit(0 if status["ready"] else 1)


if __name__ == "__main__":
    main()
