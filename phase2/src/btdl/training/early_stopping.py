"""Early stopping on validation macro-F1, with a lower-val-loss tie-break (D10/D13)."""

from dataclasses import dataclass

_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class StepResult:
    improved: bool
    should_stop: bool


class EarlyStopping:
    """mode='max': higher val_macro_f1 is better.

    Improvement: f1 > best_f1 + min_delta, OR f1 equal to best_f1 (within
    1e-12) AND val_loss < best_loss (tie-break). should_stop becomes True
    once `patience` consecutive epochs pass without improvement.
    """

    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "max"):
        if mode != "max":
            raise ValueError("EarlyStopping currently only supports mode='max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_f1 = None
        self.best_loss = None
        self.best_epoch = None
        self.epochs_without_improvement = 0

    def step(self, val_macro_f1: float, val_loss: float, epoch: int) -> StepResult:
        if self.best_f1 is None:
            improved = True
        elif val_macro_f1 > self.best_f1 + self.min_delta:
            improved = True
        elif abs(val_macro_f1 - self.best_f1) <= _TIE_TOLERANCE and val_loss < self.best_loss:
            improved = True
        else:
            improved = False

        if improved:
            self.best_f1 = val_macro_f1
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        should_stop = self.epochs_without_improvement >= self.patience
        return StepResult(improved=improved, should_stop=should_stop)

    def state_dict(self) -> dict:
        return {
            "best_f1": self.best_f1,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_f1 = state["best_f1"]
        self.best_loss = state["best_loss"]
        self.best_epoch = state["best_epoch"]
        self.epochs_without_improvement = state["epochs_without_improvement"]
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.mode = state["mode"]
