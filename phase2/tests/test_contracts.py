import pytest

from btdl.contracts import (
    CLASS_NAMES,
    CLASS_NAMES_BY_INDEX,
    CLASS_ORDER_INDEX,
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    NUM_CLASSES,
    PROJECT_LABELS,
    index_to_label,
    label_to_index,
)

# Values pinned to Phase 1's src/evaluation/metrics.py:
#   CLASS_ORDER = [1, 2, 3]
#   CLASS_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}
PHASE1_CLASS_ORDER = [1, 2, 3]
PHASE1_CLASS_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def test_project_labels_match_phase1_class_order():
    assert list(PROJECT_LABELS) == PHASE1_CLASS_ORDER


def test_class_names_match_phase1():
    assert dict(CLASS_NAMES) == PHASE1_CLASS_NAMES


def test_num_classes():
    assert NUM_CLASSES == 3
    assert len(PROJECT_LABELS) == NUM_CLASSES


@pytest.mark.parametrize("label", [1, 2, 3])
def test_label_to_index_round_trip(label):
    index = label_to_index(label)
    assert index_to_label(index) == label


def test_label_to_index_values():
    assert label_to_index(1) == 0
    assert label_to_index(2) == 1
    assert label_to_index(3) == 2


def test_mappings_are_inverses():
    for label, index in LABEL_TO_INDEX.items():
        assert INDEX_TO_LABEL[index] == label


def test_class_order_index_is_contiguous_zero_based():
    assert CLASS_ORDER_INDEX == (0, 1, 2)


def test_class_names_by_index_matches_label_order():
    assert CLASS_NAMES_BY_INDEX == ("meningioma", "glioma", "pituitary")


@pytest.mark.parametrize("bad_label", [0, 4, -1, 100])
def test_label_to_index_rejects_out_of_range(bad_label):
    with pytest.raises(ValueError):
        label_to_index(bad_label)


@pytest.mark.parametrize("bad_label", [True, False, 1.0, "1", None, [1]])
def test_label_to_index_rejects_wrong_type(bad_label):
    with pytest.raises(ValueError):
        label_to_index(bad_label)


def test_label_to_index_rejects_numpy_non_integer():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        label_to_index(np.float64(1.0))


def test_label_to_index_rejects_numpy_bool():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        label_to_index(np.bool_(True))


@pytest.mark.parametrize("np_int_type_name", ["int8", "int16", "int32", "int64"])
def test_label_to_index_accepts_numpy_integer_scalars(np_int_type_name):
    np = pytest.importorskip("numpy")
    np_int_type = getattr(np, np_int_type_name)
    index = label_to_index(np_int_type(2))
    assert index == 1
    assert type(index) is int


@pytest.mark.parametrize("np_int_type_name", ["int8", "int16", "int32", "int64"])
def test_index_to_label_accepts_numpy_integer_scalars(np_int_type_name):
    np = pytest.importorskip("numpy")
    np_int_type = getattr(np, np_int_type_name)
    label = index_to_label(np_int_type(1))
    assert label == 2
    assert type(label) is int


@pytest.mark.parametrize("bad_index", [-1, 3, 100])
def test_index_to_label_rejects_out_of_range(bad_index):
    with pytest.raises(ValueError):
        index_to_label(bad_index)


@pytest.mark.parametrize("bad_index", [True, False, 1.0, "0", None])
def test_index_to_label_rejects_wrong_type(bad_index):
    with pytest.raises(ValueError):
        index_to_label(bad_index)


def test_index_to_label_rejects_numpy_non_integer():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        index_to_label(np.float64(1.0))


def test_index_to_label_rejects_numpy_bool():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        index_to_label(np.bool_(True))
