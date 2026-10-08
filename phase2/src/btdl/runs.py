"""Canonical run_dir layout, shared by every CLI that reads or writes one."""

from pathlib import Path

from btdl import config

RUNS_RELPATH = "phase2/runs"
SMOKE_DIRNAME = "_smoke"


def format_lr(lr: float) -> str:
    """Canonical, filesystem-safe lr formatting: 1e-3 -> "1e-03"."""

    return f"{lr:.0e}"


def run_dir_for(model: str, lr: float, seed: int, smoke: bool = False) -> Path:
    """phase2/runs/<model>/lr<lr>_seed<seed>/, or phase2/runs/_smoke/<model>/... if smoke."""

    lr_str = format_lr(lr)
    base = config.repo_root() / RUNS_RELPATH
    if smoke:
        base = base / SMOKE_DIRNAME
    return base / model / f"lr{lr_str}_seed{seed}"
