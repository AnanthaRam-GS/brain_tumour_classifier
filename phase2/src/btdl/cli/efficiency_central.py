"""python -m btdl.cli.efficiency_central --device {cpu,mps,cuda,auto} [--overwrite]

Central, single-machine, cross-architecture efficiency measurement (D18):
every study model's parameter count and inference timing, measured
together on one machine so the numbers are directly comparable --
separate from each developer's own per-run efficiency.json (kept as
provenance only, not the comparison's source of truth).

Models in study.yaml that aren't yet registered in the catalog are
skipped and listed (not an error -- this tool is run again once every
architecture lands). Weights are never loaded (pretrained=False) since
they don't affect timing. For CPU, torch.set_num_threads is fixed to a
recorded value -- default: the PERFORMANCE-core count when detectable
(`sysctl hw.perflevel0.physicalcpu` on Apple Silicon), else the physical
core count, else os.cpu_count() -- so the measurement's thread count is
itself part of the record, not incidental.

Thread-count policy (diagnosed in docs/DECISIONS.md D19's CPU timing
investigation): on a heterogeneous performance/efficiency-core CPU (e.g.
Apple Silicon), PyTorch's intra-op thread pool synchronizes across ALL
requested threads every op, including the slow efficiency cores -- for a
small batch (1 image), that synchronization overhead dominates the tiny
amount of actual compute and grows WORSE with more threads (measured:
~30ms/img at 1 thread -> ~86ms/img at 4 performance-core threads ->
~200ms/img at all 10 physical-core threads, while batch=32 stayed flat
across all three). Using only the performance cores is the general-purpose
fix: it does not require per-batch-size thread tuning, and avoids the
efficiency cores acting as stragglers in the parallel barrier.

Writes TRACKED phase2/artifacts/comparison/efficiency_<device>.json.
Overwriting an existing file requires --overwrite (these are
measurements, not provenance-gated results -- but still not meant to be
silently clobbered).
"""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

import torch

from btdl import config
from btdl.evaluation.efficiency import count_parameters, measure_inference
from btdl.models.registry import MODEL_REGISTRY, build_model
from btdl.training.device import describe_device, select_device

COMPARISON_DIR_RELPATH = "phase2/artifacts/comparison"


class EfficiencyCentralError(ValueError):
    """Raised when the output file already exists without --overwrite."""


def _physical_cpu_count() -> int:
    if platform.system() == "Darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            lscpu = subprocess.check_output(["lscpu", "-p=Core,Socket"], text=True)
            cores = {line for line in lscpu.splitlines() if line and not line.startswith("#")}
            if cores:
                return len(cores)
        except Exception:
            pass
    return os.cpu_count() or 1


def _performance_core_count():
    """Apple Silicon's performance-core count (hw.perflevel0.physicalcpu),
    or None where undetectable (non-Darwin, or a non-heterogeneous CPU)."""

    if platform.system() != "Darwin":
        return None
    try:
        count = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True).strip())
        return count if count > 0 else None
    except Exception:
        return None


def _default_cpu_threads() -> tuple:
    """Returns (thread_count, policy_name) -- see module docstring."""

    perf_cores = _performance_core_count()
    if perf_cores is not None:
        return perf_cores, "performance_cores"
    return _physical_cpu_count(), "physical_cores"


def _cpu_model_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def run_efficiency_central(device, *, num_threads: int = None, overwrite: bool = False, repo_root=None) -> dict:
    repo_root = Path(repo_root) if repo_root is not None else config.repo_root()
    study = config.load_study()

    out_path = repo_root / COMPARISON_DIR_RELPATH / f"efficiency_{device.type}.json"
    if out_path.is_file() and not overwrite:
        raise EfficiencyCentralError(f"{out_path} already exists -- pass --overwrite to replace it")

    thread_policy = None
    if device.type == "cpu":
        if num_threads is not None:
            thread_policy = "explicit"
        else:
            num_threads, thread_policy = _default_cpu_threads()
        torch.set_num_threads(num_threads)

    present_models = []
    missing_models = []
    results = {}
    for model_name in study["models"]:
        if model_name not in MODEL_REGISTRY:
            missing_models.append(model_name)
            continue
        present_models.append(model_name)
        model = build_model(model_name, pretrained=False)  # weights don't affect timing
        inference = measure_inference(model, device)
        results[model_name] = {
            "parameters": count_parameters(model),
            "inference": {str(batch_size): stats for batch_size, stats in inference.items()},
        }

    machine = {
        "platform": platform.platform(),
        "cpu_model": _cpu_model_name(),
        "torch_version": torch.__version__,
        "device": describe_device(device),
        "num_threads": torch.get_num_threads() if device.type == "cpu" else None,
        "thread_policy": thread_policy,
        "performance_core_count": _performance_core_count(),
        "physical_core_count": _physical_cpu_count(),
    }

    payload = {
        "study_version": study["study_version"],
        "machine": machine,
        "present_models": present_models,
        "missing_models": missing_models,
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    payload = run_efficiency_central(device, num_threads=args.num_threads, overwrite=args.overwrite)
    summary = {k: v for k, v in payload.items() if k != "results"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"measured: {payload['present_models']}; skipped (not yet in catalog): {payload['missing_models']}")


if __name__ == "__main__":
    main()
