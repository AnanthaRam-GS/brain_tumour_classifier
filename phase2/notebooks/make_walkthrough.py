"""python notebooks/make_walkthrough.py --model NAME   (run from phase2/)

Generates phase2/notebooks/<NAME>_walkthrough.ipynb from the cell
definitions in this script (the single source of truth -- the notebook
itself is a build artifact, never hand-edited). Outputs are always
cleared; the notebook is tagged to run on the "btdl-phase2" kernel.

Every notebook this script produces follows docs/DECISIONS.md D17/D18:
TEST_MODE defaults to "rehearsal" (runs the final_test pipeline on the
VALIDATION split, never the real test split) and CONFIRM_FINAL_TEST
defaults to False (the real test split is only ever touched when a
presenter deliberately flips both settings during an actual demo).
"""

import argparse
from pathlib import Path

import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).resolve().parent
KERNEL_NAME = "btdl-phase2"
KERNEL_DISPLAY_NAME = "Python (btdl phase2)"


def _md(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def _code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def build_cells(model: str) -> list:
    cells = []

    # ---- 0. Title ----------------------------------------------------------------------
    cells.append(
        _md(
            f"""
# Phase 2 walkthrough: `{model}`

**Objective.** This notebook demonstrates, end to end, how `{model}` was
trained and evaluated for the brain-tumour MRI classification study: the
patient-level split, the ROI preprocessing pipeline, the model itself, the
LR-selection and 3-seed training protocol, validation results, and
(gated, D17/D18) the official final test.

**How to run.** Copy this notebook to `phase2/notebooks/_live/` first (that
directory is gitignored -- it is where any notebook actually gets
*executed*, so a demo run never dirties the tracked copy or the tracked
`test_access_log.csv` by accident):

```bash
mkdir -p phase2/notebooks/_live
cp phase2/notebooks/{model}_walkthrough.ipynb phase2/notebooks/_live/
cd phase2 && jupyter lab notebooks/_live/{model}_walkthrough.ipynb
```

Select the **{KERNEL_DISPLAY_NAME}** kernel (`{KERNEL_NAME}`) if it isn't
already selected. Run cells top to bottom.
"""
        )
    )

    # ---- 1. Settings --------------------------------------------------------------------
    cells.append(_md("## 1. Settings"))
    cells.append(
        _code(
            f"""
MODEL = {model!r}  # baked when this notebook was generated
TEST_MODE = "rehearsal"  # "rehearsal" | "final" -- see docs/DECISIONS.md D17/D18
CONFIRM_FINAL_TEST = False  # must be flipped to True, together with TEST_MODE="final",
                             # to ever touch the real test split (see section 9)
"""
        )
    )

    # ---- 2. Environment check ------------------------------------------------------------
    cells.append(
        _md(
            """
## 2. Environment check

A PASS/FAIL/WARN checklist (versions, device, contracts, the foundation
lock, the manifest/split, the ROI cache, working-tree cleanliness, the
model registry). A FAIL here means the rest of the notebook cannot be
trusted to reproduce the official numbers.
"""
        )
    )
    cells.append(
        _code(
            """
from btdl.cli.check_setup import run_check_setup

any_fail = False
for check in run_check_setup():
    print(f"[{check.status}] {check.name}: {check.detail}")
    if check.status == "FAIL":
        any_fail = True
print()
print("ALL CHECKS PASSED" if not any_fail else "SOME CHECKS FAILED -- see above")
"""
        )
    )

    # ---- 3. Dataset and patient-level split ------------------------------------------------
    cells.append(
        _md(
            """
## 3. Dataset and patient-level split

The frozen train/val/test split (docs/DECISIONS.md D3) is patient-level:
no patient appears in more than one split. The table below is read
straight from the manifest and the split file -- nothing here is
recomputed.
"""
        )
    )
    cells.append(
        _code(
            """
from btdl import walkthrough

overview = walkthrough.dataset_overview()
print(f"{overview['n_samples']} samples from {overview['n_patients']} patients")
print(
    "Patient-level split integrity: OK"
    if overview["patient_overlap_ok"]
    else f"FAILED -- {overview['patient_overlap_detail']}"
)
overview["split_table"]
"""
        )
    )
    cells.append(_code("overview[\"class_table\"]"))

    # ---- 4. Preprocessing ------------------------------------------------------------------
    cells.append(
        _md(
            """
## 4. Preprocessing: the ROI pipeline

One TRAIN sample, shown at every stage: the raw normalised image with its
tumour mask and the derived crop box, the cached 224x224 ROI, and the
final 3-channel model-input tensor. Then the existing QA contact sheets
(augmentation examples and a cross-section of ROI crops), which were
generated once by `cli/aug_qa.py` / `cli/roi_qa.py` -- this notebook only
displays them, it never regenerates them.
"""
        )
    )
    cells.append(
        _code(
            """
import matplotlib.pyplot as plt

from btdl.data.split import load_split

split_index = load_split()
sample_id = split_index.sample_ids("train")[0]
pipeline = walkthrough.show_roi_pipeline(sample_id)
print(f"sample {sample_id!r}: patient {pipeline['patient_id']}, label {pipeline['label']}")

if pipeline["cache_only_checkout"]:
    print("(cache-only checkout -- raw NPZ not present locally; showing the cached ROI only)")
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    axes.imshow(pipeline["cached_roi_224"], cmap="gray")
    axes.set_title("cached 224x224 ROI")
    axes.axis("off")
else:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(pipeline["raw_image"], cmap="gray")
    axes[0].set_title("raw normalised image")
    axes[1].imshow(pipeline["raw_mask"], cmap="gray")
    axes[1].set_title("tumour mask")
    axes[2].imshow(pipeline["cached_roi_224"], cmap="gray")
    axes[2].set_title("cached 224x224 ROI")
    for ax in axes:
        ax.axis("off")
plt.show()

print("model-input tensor shape:", pipeline["model_input_tensor_shape"])
print("model-input tensor stats:", pipeline["model_input_tensor_stats"])
"""
        )
    )
    cells.append(
        _code(
            """
sheets = walkthrough.show_qa_sheets()
for name, path in sheets.items():
    if path is None:
        print(f"{name}: not generated locally (run cli/aug_qa.py or cli/roi_qa.py to produce it)")
        continue
    img = plt.imread(path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(name)
    plt.axis("off")
    plt.show()
"""
        )
    )

    # ---- 5. Model architecture ---------------------------------------------------------------
    cells.append(_md("## 5. Model architecture and parameters"))
    cells.append(
        _code(
            """
model_info = walkthrough.model_overview(MODEL, instantiate=True)
print(f"{model_info['name']} v{model_info['version']}")
print(f"pretrained weights: {model_info['weights_id']}")
print(f"total parameters: {model_info['total_parameters']:,}")
print(f"trainable parameters: {model_info['trainable_parameters']:,}")
print()
print(model_info["module_tree"])
"""
        )
    )

    # ---- 6. Live training demonstration --------------------------------------------------------
    cells.append(
        _md(
            """
## 6. Live training demonstration

A 2-epoch `--smoke` run at the study's selected learning rate, seed 42.
**This is a demonstration only -- not the official result** (a real final
run trains for up to `training.yaml`'s `max_epochs`, with early stopping).
Any previous smoke run for this model is moved aside first (never
deleted), under the gitignored `phase2/runs/_smoke/_previous/` tree.
"""
        )
    )
    cells.append(
        _code(
            """
import json
import shutil
from datetime import datetime, timezone

from btdl import config
from btdl.cli.train import run_train
from btdl.runs import run_dir_for
from btdl.training.device import select_device

with (config.repo_root() / "phase2" / "artifacts" / "selection" / f"{MODEL}.json").open() as handle:
    selection = json.load(handle)
selected_lr = selection["selected_lr"]

smoke_dir = run_dir_for(MODEL, selected_lr, 42, smoke=True)
if smoke_dir.exists():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    previous_dir = config.repo_root() / "phase2" / "runs" / "_smoke" / "_previous" / timestamp
    previous_dir.mkdir(parents=True, exist_ok=True)
    dest = previous_dir / f"{MODEL}_{smoke_dir.name}"
    shutil.move(str(smoke_dir), str(dest))
    print(f"moved existing smoke run to {dest}")

device = select_device("auto")
print("DEMONSTRATION ONLY -- not the official result")
result = run_train(MODEL, selected_lr, 42, smoke=True, device=device)
print(f"ran {result.epochs_run} epoch(s), stop reason: {result.stop_reason!r}")

import pandas as pd

pd.read_csv(result.history_path)
"""
        )
    )

    # ---- 7. Official protocol -----------------------------------------------------------------
    cells.append(
        _md(
            """
## 7. Official protocol: LR grid, selection, 3 seeds

The LR grid search (one seed), the selected learning rate, and the
training curves of every final-seed run at that learning rate -- read
directly from the artifacts committed when the protocol was run (this
notebook never re-runs the grid search or re-trains a final seed).
"""
        )
    )
    cells.append(
        _code(
            """
protocol = walkthrough.protocol_overview(MODEL)
print(f"selected lr: {protocol['selected_lr']}")
protocol["grid_table"]
"""
        )
    )
    cells.append(_code("protocol[\"seed_table\"]"))
    cells.append(
        _code(
            """
for name, stats in protocol["seed_summary"].items():
    print(f"{name}: {stats['mean']:.4f} +/- {stats['std']:.4f} (sample std, n=3 seeds)")
"""
        )
    )
    cells.append(
        _code(
            """
fig, ax = plt.subplots(figsize=(7, 4))
for seed, history in protocol["training_curves"].items():
    ax.plot(history["epoch"], history["val_macro_f1"], label=f"seed {seed}")
ax.set_xlabel("epoch")
ax.set_ylabel("val macro-F1")
ax.set_title(f"{MODEL}: training curves (3 final seeds)")
ax.legend()
plt.show()
"""
        )
    )

    # ---- 8. Validation results -----------------------------------------------------------------
    cells.append(
        _md(
            """
## 8. Validation results

Per-seed and seed-aggregated metrics on the VALIDATION split (metrics the
model selection itself was based on) -- confusion matrices, ROC, per-class
breakdown (meningioma highlighted), size-stratified recall, and
calibration.
"""
        )
    )
    cells.append(
        _code(
            """
val_results = walkthrough.split_results(MODEL, "val")
headline = val_results["aggregates"]["headline"]
for name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
    stats = headline[name]
    print(f"{name}: {stats['mean']:.4f} +/- {stats['std']:.4f}  (values: {stats['values']})")
"""
        )
    )
    cells.append(
        _code(
            """
per_class = val_results["aggregates"]["per_class"]
rows = []
for class_name, metrics in per_class.items():
    row = {"class": class_name}
    for metric_name, stats in metrics.items():
        row[metric_name] = f"{stats['mean']:.3f} +/- {stats['std']:.3f}"
    rows.append(row)
per_class_frame = pd.DataFrame(rows).set_index("class")
# meningioma highlighted: it is the clinically highest-stakes class in this study
per_class_frame
"""
        )
    )
    cells.append(
        _code(
            """
import numpy as np

confusion_sum = sum(seed_data["confusion_matrix"] for seed_data in val_results["per_seed"].values())
print("seed-summed confusion matrix:")
display(confusion_sum)

fig, axes = plt.subplots(1, len(val_results["per_seed"]), figsize=(4 * len(val_results["per_seed"]), 4))
for ax, (seed, seed_data) in zip(np.atleast_1d(axes), val_results["per_seed"].items()):
    ax.imshow(seed_data["confusion_matrix"].to_numpy(), cmap="Blues")
    ax.set_title(f"seed {seed}")
plt.show()
"""
        )
    )
    cells.append(
        _code(
            """
fig, ax = plt.subplots(figsize=(5, 5))
for seed, seed_data in val_results["per_seed"].items():
    for class_name, curve in seed_data["roc_curves"].items():
        ax.plot(curve["fpr"], curve["tpr"], alpha=0.4, label=f"seed {seed} / {class_name}")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title(f"{MODEL}: ROC curves (val, all seeds/classes)")
ax.legend(fontsize=6, ncol=2)
plt.show()
"""
        )
    )
    cells.append(
        _code(
            """
first_seed = next(iter(val_results["per_seed"]))
ci = val_results["per_seed"][first_seed]["bootstrap_ci"]["metrics"]
pd.DataFrame(ci).T[["point", "ci_low", "ci_high"]]
"""
        )
    )
    cells.append(
        _code(
            """
stratified = val_results["aggregates"]["stratified"]
rows = []
for tertile, per_class_recall in stratified.items():
    row = {"size_tertile": tertile}
    for class_name, stats in per_class_recall.items():
        row[class_name] = f"{stats['mean']:.3f} +/- {stats['std']:.3f}" if stats["mean"] is not None else "n/a"
    rows.append(row)
pd.DataFrame(rows).set_index("size_tertile")
"""
        )
    )
    cells.append(
        _code(
            """
calibration = val_results["aggregates"].get("calibration")
if calibration is not None:
    print(f"expected calibration error (top-label, 15 bins): {calibration['mean']:.4f} +/- {calibration['std']:.4f}")
"""
        )
    )

    # ---- 9. Final test ------------------------------------------------------------------------
    cells.append(
        _md(
            """
## 9. Final test

**Rehearsal mode** (the default here) runs the IDENTICAL pipeline on the
VALIDATION split (docs/DECISIONS.md D17) -- it never touches the real
test split and never writes `test_access_log.csv`. **Final mode** is the
one and only sanctioned way to touch the real test split
(docs/DECISIONS.md D18): it refuses outright unless `CONFIRM_FINAL_TEST`
is also `True`, requires this model's frozen registration to already be
committed, and appends one row to the tracked `test_access_log.csv` per
seed. Every final-seed run is processed in turn; a seed that is already
logged is shown from the recorded result rather than re-run. The actual
gated call lives in `btdl.walkthrough.run_official_final_test` -- the one
place that module is allowed to touch the real test split.
"""
        )
    )
    cells.append(_code("final_seeds = sorted(config.load_contract(\"training\")[\"final_seeds\"])"))

    cells.append(
        _md(
            """
### 9a. Pre-test readiness check

Runs `final_test --check-only` for every final seed: every real-mode
gate, in order, with nothing written and no dataset constructed. This
runs the same way regardless of `TEST_MODE` -- in rehearsal mode it is
simply the demo-day preflight for when you DO flip to final mode.
"""
        )
    )
    cells.append(
        _code(
            """
from btdl.cli.final_test import check_final_test_readiness

readiness_all_pass = True
for seed in final_seeds:
    run_dir = run_dir_for(MODEL, selected_lr, seed, smoke=False)
    print(f"--- seed {seed} ({run_dir}) ---")
    gate_results = check_final_test_readiness(run_dir, repo_root=config.repo_root())
    for gate in gate_results:
        print(f"  [{'PASS' if gate['ok'] else 'FAIL'}] {gate['name']}: {gate['detail']}")
    seed_pass = all(gate["ok"] for gate in gate_results)
    readiness_all_pass = readiness_all_pass and seed_pass
    print("  ALL GATES PASSED" if seed_pass else "  SOME GATES FAILED")
    print()
print("READY FOR FINAL TEST" if readiness_all_pass else "NOT YET READY FOR FINAL TEST (see FAILs above)")
"""
        )
    )

    cells.append(
        _code(
            """
if TEST_MODE == "rehearsal":
    from btdl.cli.final_test import run_final_test

    for seed in final_seeds:
        run_dir = run_dir_for(MODEL, selected_lr, seed, smoke=False)
        rehearsal_dir = run_dir / "eval" / "rehearsal_val"
        if rehearsal_dir.is_dir():
            print(f"seed {seed}: rehearsal_val already exists at {rehearsal_dir} -- showing it")
            continue
        print(f"seed {seed}: running final_test --rehearsal ...")
        metrics = run_final_test(run_dir, confirm_final_test=True, device=device, rehearsal=True)
        print(f"seed {seed}: rehearsal macro_f1={metrics['macro_f1']:.4f}")
    print()
    print("REHEARSAL (VALIDATION DATA) -- not the official test result")

elif TEST_MODE == "final":
    result = walkthrough.run_official_final_test(MODEL, confirm=CONFIRM_FINAL_TEST, device=device)
    for seed, status in result["per_seed"].items():
        print(
            f"seed {seed}: {status['status']} -- {status['detail']} "
            f"(export: {status.get('export_status', 'n/a')})"
        )
    print()
    print("Afterwards, run:")
    for command in result["git_commands"]:
        print(" ", command)
else:
    raise ValueError(f"TEST_MODE must be 'rehearsal' or 'final', got {TEST_MODE!r}")
"""
        )
    )

    # ---- 10. Test / rehearsal results -----------------------------------------------------------
    cells.append(
        _md(
            """
## 10. Test results (test mode) or rehearsal results

Same shape as section 8, now on whichever split section 9 actually used:
`test` in final mode (official), `rehearsal_val` in rehearsal mode (a
dry run on validation data).
"""
        )
    )
    cells.append(
        _code(
            """
results_split = "test" if TEST_MODE == "final" else "rehearsal_val"
final_results = walkthrough.split_results(MODEL, results_split)
label = "TEST (OFFICIAL)" if results_split == "test" else "REHEARSAL (VALIDATION DATA)"
print(label)
if not final_results["available"]:
    print(final_results["note"])
else:
    headline = final_results["aggregates"]["headline"]
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        stats = headline[name]
        print(f"{name}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
"""
        )
    )
    cells.append(
        _code(
            """
if final_results["available"]:
    per_class_final = final_results["aggregates"]["per_class"]
    rows = []
    for class_name, metrics in per_class_final.items():
        row = {"class": class_name}
        for metric_name, stats in metrics.items():
            row[metric_name] = f"{stats['mean']:.3f} +/- {stats['std']:.3f}"
        rows.append(row)
    display(pd.DataFrame(rows).set_index("class"))

    confusion_sum_final = sum(
        seed_data["confusion_matrix"] for seed_data in final_results["per_seed"].values()
    )
    display(confusion_sum_final)

    fig, ax = plt.subplots(figsize=(5, 5))
    for seed, seed_data in final_results["per_seed"].items():
        for class_name, curve in seed_data["roc_curves"].items():
            ax.plot(curve["fpr"], curve["tpr"], alpha=0.4, label=f"seed {seed} / {class_name}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_title(f"{MODEL}: ROC curves ({results_split})")
    ax.legend(fontsize=6, ncol=2)
    plt.show()

    calibration_final = final_results["aggregates"].get("calibration")
    if calibration_final is not None:
        print(f"ECE: {calibration_final['mean']:.4f} +/- {calibration_final['std']:.4f}")
"""
        )
    )

    # ---- 11. Baseline comparison ---------------------------------------------------------------
    cells.append(
        _md(
            """
## 11. Comparison with the Phase 1 baseline (final mode only)

Only meaningful once the real test split has been evaluated AND the
cross-model comparison (`cli/compare.py --mode test`) has been run and
committed -- this notebook only reads that comparison's output, it never
recomputes it.
"""
        )
    )
    cells.append(
        _code(
            """
if TEST_MODE == "final":
    baseline = walkthrough.baseline_comparison(MODEL)
    if not baseline["available"]:
        print(baseline["note"])
    else:
        print("vs. Phase 1 GLCM v2 baseline:")
        display(pd.DataFrame({"model": baseline["model"], "baseline": baseline["baseline"]}))
        if baseline["paired_vs_baseline"] is not None:
            display(pd.DataFrame(baseline["paired_vs_baseline"]).T)
else:
    print("skipped -- only runs when TEST_MODE == 'final'")
"""
        )
    )

    # ---- 12. Example predictions ---------------------------------------------------------------
    cells.append(
        _md(
            """
## 12. Example predictions

A handful of predictions from the already-written `predictions.csv` for
this split -- images from the ROI cache, true vs. predicted class, and
per-class probabilities. At least one misclassified example is shown if
one exists. No new inference is run here.
"""
        )
    )
    cells.append(
        _code(
            """
examples = walkthrough.example_predictions(MODEL, results_split, n=6)
if not examples["available"]:
    print(examples["note"])
else:
    rows = [ex["row"] for ex in examples["examples"]]
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for ax, ex in zip(np.atleast_1d(axes), examples["examples"]):
        if ex["image"] is not None:
            ax.imshow(ex["image"], cmap="gray")
        row = ex["row"]
        correct_mark = "correct" if row["correct"] else "MISCLASSIFIED"
        ax.set_title(f"true={row['true_name']}\\npred={row['pred_name']} ({correct_mark})", fontsize=8)
        ax.axis("off")
    plt.show()
    pd.DataFrame(rows)
"""
        )
    )

    # ---- 13. Efficiency --------------------------------------------------------------------------
    cells.append(
        _md(
            """
## 13. Efficiency

Per-run inference timing figures (`cli/efficiency.py`), recorded once
alongside each frozen checkpoint. The single cross-model efficiency
comparison is measured centrally and reported separately
(`cli/efficiency_central.py`, `phase2/artifacts/comparison/`) -- the
numbers below are this model's own per-run figures only.
"""
        )
    )
    cells.append(
        _code(
            """
rows = []
for seed in final_seeds:
    run_dir = run_dir_for(MODEL, selected_lr, seed, smoke=False)
    efficiency_path = run_dir / "efficiency.json"
    if not efficiency_path.is_file():
        continue
    with efficiency_path.open() as handle:
        efficiency = json.load(handle)
    for batch_size, payload in efficiency.get("inference", {}).items():
        rows.append(
            {
                "seed": seed,
                "batch_size": batch_size,
                "device": payload["device"]["type"],
                "ms_per_image": payload["ms_per_image"],
            }
        )
pd.DataFrame(rows)
"""
        )
    )

    # ---- 14. Summary -----------------------------------------------------------------------------
    cells.append(
        _md(
            """
## 14. Summary

Generated directly from the numbers computed above -- no hand-written
claims.
"""
        )
    )
    cells.append(
        _code(
            """
sentences = []
sentences.append(
    f"{MODEL} was selected at lr={protocol['selected_lr']} and trained for 3 final seeds "
    f"({', '.join(str(s) for s in final_seeds)})."
)
val_f1 = val_results["aggregates"]["headline"]["macro_f1"]
sentences.append(
    f"On the validation split, {MODEL} reaches a macro-F1 of {val_f1['mean']:.3f} +/- {val_f1['std']:.3f} "
    f"across {len(val_f1['values'])} seeds."
)
if final_results["available"]:
    final_f1 = final_results["aggregates"]["headline"]["macro_f1"]
    sentences.append(
        f"On the {results_split} split, {MODEL} reaches a macro-F1 of "
        f"{final_f1['mean']:.3f} +/- {final_f1['std']:.3f}."
    )
else:
    sentences.append(f"No {results_split} results are available yet for {MODEL}.")

for sentence in sentences:
    print("-", sentence)
"""
        )
    )

    return cells


def make_notebook(model: str) -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = build_cells(model)
    notebook["metadata"]["kernelspec"] = {
        "name": KERNEL_NAME,
        "display_name": KERNEL_DISPLAY_NAME,
        "language": "python",
    }

    out_path = NOTEBOOKS_DIR / f"{model}_walkthrough.ipynb"
    with out_path.open("w") as handle:
        nbf.write(notebook, handle)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    out_path = make_notebook(args.model)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
