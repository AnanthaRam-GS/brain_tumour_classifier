import pandas as pd

from btdl.evaluation.plots import plot_confusion_matrix, plot_roc_curves, plot_training_curves

CONFUSION_MATRIX = [[33, 54, 30], [17, 187, 18], [19, 37, 103]]
ROC_CURVES = {
    "meningioma": {"fpr": [0, 0.5, 1], "tpr": [0, 0.8, 1], "thresholds": [1, 0.5, 0]},
    "glioma": {"fpr": [0, 0.3, 1], "tpr": [0, 0.7, 1], "thresholds": [1, 0.5, 0]},
    "pituitary": {"fpr": [0, 0.2, 1], "tpr": [0, 0.9, 1], "thresholds": [1, 0.5, 0]},
}
ROC_AUC = {"macro_ovr": 0.807, "per_class": {"meningioma": 0.75, "glioma": 0.8, "pituitary": 0.85}}


def _history_csv(tmp_path):
    history = pd.DataFrame(
        {
            "epoch": range(5),
            "train_aug_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
            "val_loss": [1.1, 0.9, 0.7, 0.65, 0.6],
            "train_aug_macro_f1": [0.3, 0.5, 0.6, 0.7, 0.75],
            "val_macro_f1": [0.25, 0.45, 0.55, 0.6, 0.62],
        }
    )
    path = tmp_path / "history.csv"
    history.to_csv(path, index=False)
    return path


def test_confusion_matrix_plot_created_and_non_empty(tmp_path):
    paths = plot_confusion_matrix(CONFUSION_MATRIX, tmp_path)
    assert len(paths) == 1
    assert paths[0].is_file()
    assert paths[0].stat().st_size > 0


def test_roc_curves_plot_created_and_non_empty(tmp_path):
    paths = plot_roc_curves(ROC_CURVES, ROC_AUC, tmp_path)
    assert len(paths) == 1
    assert paths[0].is_file()
    assert paths[0].stat().st_size > 0


def test_training_curves_plot_created_and_non_empty(tmp_path):
    history_path = _history_csv(tmp_path)
    paths = plot_training_curves(history_path, tmp_path)
    assert len(paths) == 1
    assert paths[0].is_file()
    assert paths[0].stat().st_size > 0


def test_roc_curves_plot_handles_missing_per_class_auc(tmp_path):
    roc_auc = {"macro_ovr": None, "per_class": {"meningioma": None, "glioma": 0.8, "pituitary": 0.85}}
    paths = plot_roc_curves(ROC_CURVES, roc_auc, tmp_path)
    assert paths[0].is_file()


def test_training_curves_plot_with_single_epoch(tmp_path):
    history = pd.DataFrame(
        {
            "epoch": [0],
            "train_aug_loss": [1.0],
            "val_loss": [1.1],
            "train_aug_macro_f1": [0.3],
            "val_macro_f1": [0.25],
        }
    )
    path = tmp_path / "history.csv"
    history.to_csv(path, index=False)
    paths = plot_training_curves(path, tmp_path)
    assert paths[0].is_file()
