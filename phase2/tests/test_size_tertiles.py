import pandas as pd

from btdl.cli.size_tertiles import compute_size_tertiles


def _geometry(train_sides, val_sides, test_sides):
    rows = []
    sid = 0
    for side in train_sides:
        rows.append({"sample_id": str(sid), "split": "train", "crop_side": side})
        sid += 1
    for side in val_sides:
        rows.append({"sample_id": str(sid), "split": "val", "crop_side": side})
        sid += 1
    for side in test_sides:
        rows.append({"sample_id": str(sid), "split": "test", "crop_side": side})
        sid += 1
    return pd.DataFrame(rows)


def test_edges_computed_from_train_only():
    train_sides = list(range(1, 10))  # 1..9
    geometry = _geometry(train_sides, val_sides=[100, 200], test_sides=[300, 400])
    result = compute_size_tertiles(geometry)
    assert result["n_train"] == 9
    # tertile edges of 1..9 at 1/3 and 2/3
    low, high = result["edges"]
    assert 3.0 <= low <= 4.0
    assert 6.0 <= high <= 7.0


def test_changing_val_test_geometry_does_not_change_edges():
    train_sides = list(range(1, 10))
    geometry_a = _geometry(train_sides, val_sides=[100], test_sides=[200])
    geometry_b = _geometry(train_sides, val_sides=[9999, 1], test_sides=[5, 5, 5])
    result_a = compute_size_tertiles(geometry_a)
    result_b = compute_size_tertiles(geometry_b)
    assert result_a["edges"] == result_b["edges"]


def test_min_max_reflect_train_only():
    train_sides = [10, 20, 30]
    geometry = _geometry(train_sides, val_sides=[1], test_sides=[1000])
    result = compute_size_tertiles(geometry)
    assert result["min"] == 10.0
    assert result["max"] == 30.0
