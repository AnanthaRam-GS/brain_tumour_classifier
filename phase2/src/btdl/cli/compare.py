"""python -m btdl.cli.compare --mode {test,val-preview}

test mode: the real, post-unblinding comparison, built ONLY from exported
phase2/artifacts/results/<model>/seed<k>/ directories and the Phase 1
baseline's test predictions. Requires unblinding_status READY and every
study model exported; refuses with the list of what is missing otherwise.
Writes TRACKED tables/figures/comparison.json under
phase2/artifacts/comparison/ and phase2/docs/RESULTS.md.

val-preview mode: an exploratory preview from registered models' LOCAL
eval/val/ directories (never eval/test/, never the test split). Every
output is labelled "VALIDATION PREVIEW -- NOT RESULTS" and written to the
gitignored phase2/runs/_preview/comparison/.

Deterministic: two runs over identical inputs produce byte-identical
tables and comparison.json (bootstrap settings come from the committed
evaluation.yaml contract; nothing here uses wall-clock time or an
unseeded RNG).
"""

import argparse
import json

from btdl.comparison.report import CompareError, run_compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["test", "val-preview"])
    args = parser.parse_args()

    outcome = run_compare(args.mode)
    summary = {
        "mode": args.mode,
        "present_models": outcome["result"]["present_models"],
        "missing_models": outcome["result"]["missing_models"],
        "out_dir": str(outcome["out_dir"]),
        "comparison_json": str(outcome["comparison_json_path"]),
        "results_md": str(outcome["results_md_path"]),
        "figures": [str(p) for p in outcome["figure_paths"]],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
