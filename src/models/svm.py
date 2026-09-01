import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from src.evaluation.metrics import CLASS_ORDER
from src.evaluation.results import (
    build_prediction_table,
    build_result_dict,
    scores_from_estimator,
    write_experiment_results,
)
from src.features.feature_table import merge_feature_tables, read_feature_table
from src.models.training import (
    prepare_feature_matrices,
    train_estimator,
    transform_feature_matrices,
)

DEFAULT_SPLIT_CSV = "data/splits/patient_split.csv"
DEFAULT_SPLIT_METADATA_JSON = "data/splits/split_metadata.json"
DEFAULT_OUTPUT_ROOT = "reports/experiments"
RANDOM_SEED = 42
SELECTION_METRIC = "validation macro F1"

C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
GAMMA_GRID = ("scale", "auto", 0.001, 0.01, 0.1, 1.0)


def read_json_field(path, field):
    file_path = Path(path)
    if not file_path.is_file():
        return "unknown"
    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return str(payload.get(field, "unknown"))


def load_combined_feature_table(feature_csv, *, split_csv=DEFAULT_SPLIT_CSV):
    paths = [feature_csv] if isinstance(feature_csv, (str, Path)) else list(feature_csv)
    if not paths:
        raise ValueError("feature_csv must contain at least one path")
    tables = [read_feature_table(path, canonical_split=split_csv) for path in paths]
    if len(tables) == 1:
        return tables[0]
    return merge_feature_tables(tables, canonical_split=split_csv)


def tune_svm_hyperparameters(
    X_train, y_train, X_val, y_val, *, c_grid=C_GRID, gamma_grid=GAMMA_GRID, seed=RANDOM_SEED
):
    trials = []
    best = None
    for C in c_grid:
        for gamma in gamma_grid:
            model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=seed)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            trial = {
                "C": C,
                "gamma": gamma,
                "val_accuracy": float(accuracy_score(y_val, val_pred)),
                "val_macro_f1": float(
                    f1_score(y_val, val_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
                ),
            }
            trials.append(trial)
            if best is None or trial["val_macro_f1"] > best["val_macro_f1"]:
                best = trial
    assert best is not None
    return best, trials


def run_svm_experiment(
    feature_csv,
    *,
    feature_set_name,
    feature_version,
    experiment_name,
    split_csv=DEFAULT_SPLIT_CSV,
    split_metadata_json=DEFAULT_SPLIT_METADATA_JSON,
    split_version=None,
    output_root=DEFAULT_OUTPUT_ROOT,
    seed=RANDOM_SEED,
    c_grid=C_GRID,
    gamma_grid=GAMMA_GRID,
):
    table = load_combined_feature_table(feature_csv, split_csv=split_csv)
    matrices = prepare_feature_matrices(table, canonical_split=str(split_csv))
    preprocessed = transform_feature_matrices(matrices, imputation="median", scaling="standard")

    best_params, trials = tune_svm_hyperparameters(
        preprocessed.X_train,
        matrices.y_train,
        preprocessed.X_val,
        matrices.y_val,
        c_grid=c_grid,
        gamma_grid=gamma_grid,
        seed=seed,
    )

    final_svm = SVC(
        kernel="rbf",
        C=best_params["C"],
        gamma=best_params["gamma"],
        probability=True,
        random_state=seed,
    )
    trained = train_estimator(
        final_svm,
        preprocessed.X_train,
        matrices.y_train,
        model_name="svm_rbf",
        feature_count=len(matrices.feature_columns),
    )

    y_test_pred = trained.estimator.predict(preprocessed.X_test)
    y_test_score, score_reason = scores_from_estimator(trained.estimator, preprocessed.X_test)
    predictions = build_prediction_table(
        matrices.test_metadata, matrices.y_test, y_test_pred, y_score=y_test_score
    )

    result = build_result_dict(
        experiment_name=experiment_name,
        feature_set=feature_set_name,
        model_name=trained.model_name,
        split_evaluated="test",
        feature_count=trained.feature_count,
        metadata=matrices.test_metadata,
        y_true=matrices.y_test,
        y_pred=y_test_pred,
        model_parameters=trained.model_parameters,
        preprocessing={
            "imputation": preprocessed.preprocessor.imputation,
            "scaling": preprocessed.preprocessor.scaling,
            "hyperparameter_selection_metric": SELECTION_METRIC,
        },
        random_seed=seed,
        split_version=split_version or read_json_field(split_metadata_json, "split_version"),
        feature_version=feature_version,
        y_score=y_test_score if y_test_score is not None else None,
    )

    output_dir = Path(output_root) / experiment_name
    written = write_experiment_results(result, predictions, output_dir)

    model_path = output_dir / "model.joblib"
    joblib.dump(
        {
            "model": trained.estimator,
            "preprocessor": preprocessed.preprocessor.pipeline,
            "feature_columns": matrices.feature_columns,
        },
        model_path,
    )
    written["model"] = model_path

    search_log = {
        "selection_metric": SELECTION_METRIC,
        "candidate_grid": {"C": list(c_grid), "gamma": list(gamma_grid)},
        "chosen_hyperparameters": {"C": best_params["C"], "gamma": best_params["gamma"]},
        "chosen_trial": best_params,
        "all_trials": trials,
        "score_availability_reason": score_reason,
    }
    search_log_path = output_dir / "hyperparameter_search.json"
    search_log_path.write_text(json.dumps(search_log, indent=2) + "\n", encoding="utf-8")
    written["hyperparameter_search"] = search_log_path

    return {"result": result, "written": written, "best_params": best_params, "trials": trials}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-csv", nargs="+", required=True, type=Path,
        help="One feature CSV, or several to merge by sample_id.",
    )
    parser.add_argument("--feature-set-name", required=True)
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split-csv", type=Path, default=Path(DEFAULT_SPLIT_CSV))
    parser.add_argument(
        "--split-metadata-json", type=Path, default=Path(DEFAULT_SPLIT_METADATA_JSON)
    )
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    feature_csv = args.feature_csv[0] if len(args.feature_csv) == 1 else args.feature_csv
    outcome = run_svm_experiment(
        feature_csv,
        feature_set_name=args.feature_set_name,
        feature_version=args.feature_version,
        experiment_name=args.experiment_name,
        split_csv=args.split_csv,
        split_metadata_json=args.split_metadata_json,
        output_root=args.output_root,
        seed=args.seed,
    )
    print(json.dumps(outcome["result"]["metrics"], indent=2))
    print(json.dumps({"chosen_hyperparameters": outcome["best_params"]}, indent=2))


if __name__ == "__main__":
    main()
