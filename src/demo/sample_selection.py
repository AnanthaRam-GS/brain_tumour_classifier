"""Held-out TEST sample selection for the Review 1 live demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.feature_dataset import FeatureDatasetError, load_split_index


class DemoSampleSelectionError(ValueError):
    """Raised when a requested demo sample is not eligible."""


def get_test_sample_ids(
    split_csv: Path | str = "data/splits/patient_split.csv",
) -> list[str]:
    """Return canonical TEST sample IDs in deterministic numeric order."""

    split_index = load_split_index(split_csv)
    test_rows = split_index[split_index["split"].eq("test")]
    return test_rows.sort_values("sample_id", key=lambda column: column.astype(int))[
        "sample_id"
    ].tolist()


def select_test_sample(
    *,
    sample_id: str | int | None = None,
    seed: int | None = None,
    random_selection: bool = False,
    split_csv: Path | str = "data/splits/patient_split.csv",
) -> dict[str, object]:
    """Select one sample from the frozen TEST split only."""

    try:
        split_index = load_split_index(split_csv)
    except FeatureDatasetError as exc:
        raise DemoSampleSelectionError(str(exc)) from exc

    if sample_id is not None:
        normalized_id = str(int(sample_id))
        matches = split_index[split_index["sample_id"].eq(normalized_id)]
        if matches.empty:
            raise DemoSampleSelectionError(
                f"Sample {normalized_id} is not present in the canonical split."
            )
        record = matches.iloc[0]
        if record["split"] != "test":
            raise DemoSampleSelectionError(
                f"Sample {normalized_id} belongs to {record['split']}, not test. "
                "Demo samples must come from the held-out test set."
            )
        return record.to_dict()

    test_rows = split_index[split_index["split"].eq("test")].reset_index(drop=True)
    if test_rows.empty:
        raise DemoSampleSelectionError("No TEST samples are available in the canonical split.")
    rng = np.random.default_rng(None if random_selection else (42 if seed is None else seed))
    row_index = int(rng.integers(0, len(test_rows)))
    return test_rows.iloc[row_index].to_dict()

