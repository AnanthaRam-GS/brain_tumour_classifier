"""Builds comparison tables (CSV + Markdown) from build_comparison()'s result dict."""

from pathlib import Path

import pandas as pd

from btdl.contracts import CLASS_NAMES_BY_INDEX


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Hand-rolled (no `tabulate` dependency) GitHub-flavored markdown table."""

    def _fmt(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_table(frame: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{stem}.csv", index=False)
    with (out_dir / f"{stem}.md").open("w") as handle:
        handle.write(_dataframe_to_markdown(frame))


def headline_table(result: dict) -> pd.DataFrame:
    rows = []
    for model_name, data in sorted(result["per_model"].items()):
        row = {"model": model_name}
        for metric, stat in data["headline"].items():
            row[f"{metric}_mean"] = stat["mean"]
            row[f"{metric}_std"] = stat["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_table(result: dict) -> pd.DataFrame:
    rows = []
    for model_name, data in sorted(result["per_model"].items()):
        for class_name in CLASS_NAMES_BY_INDEX:
            row = {"model": model_name, "class": class_name}
            for metric, stat in data["per_class"][class_name].items():
                row[f"{metric}_mean"] = stat["mean"]
                row[f"{metric}_std"] = stat["std"]
            rows.append(row)
    return pd.DataFrame(rows)


def paired_table(result: dict) -> pd.DataFrame:
    rows = []
    for pair_name, metrics in sorted(result["paired"].items()):
        for metric_name, stat in metrics.items():
            rows.append(
                {
                    "pair": pair_name,
                    "metric": metric_name,
                    "diff_point": stat["point"],
                    "ci_low": stat["ci_low"],
                    "ci_high": stat["ci_high"],
                    "frac_a_greater": stat["frac_a_greater"],
                    "n_valid": stat["n_valid"],
                }
            )
    return pd.DataFrame(rows)


def stratified_table(result: dict) -> pd.DataFrame:
    rows = []
    for model_name, data in sorted(result["per_model"].items()):
        stratified = data.get("stratified")
        if not stratified:
            continue
        for tertile, classes in stratified.items():
            for class_name, stat in classes.items():
                rows.append(
                    {
                        "model": model_name,
                        "tertile": tertile,
                        "class": class_name,
                        "recall_mean": stat["mean"],
                        "recall_std": stat["std"],
                    }
                )
    return pd.DataFrame(rows)


def calibration_table(result: dict) -> pd.DataFrame:
    rows = []
    for model_name, data in sorted(result["per_model"].items()):
        calibration = data.get("calibration")
        if not calibration:
            continue
        rows.append({"model": model_name, "ece_mean": calibration["mean"], "ece_std": calibration["std"]})
    return pd.DataFrame(rows)


def efficiency_table(result: dict) -> pd.DataFrame:
    rows = []
    for device_type, payload in sorted(result.get("efficiency", {}).items()):
        for model_name, model_result in payload.get("results", {}).items():
            row = {
                "device": device_type,
                "model": model_name,
                "parameters": model_result["parameters"]["total"],
            }
            for batch_size, stats_ in model_result["inference"].items():
                row[f"ms_per_img_batch{batch_size}"] = stats_["ms_per_image"]
            rows.append(row)
    return pd.DataFrame(rows)


TABLE_BUILDERS = {
    "headline": headline_table,
    "per_class": per_class_table,
    "paired": paired_table,
    "stratified": stratified_table,
    "calibration": calibration_table,
    "efficiency": efficiency_table,
}


def write_all_tables(result: dict, out_dir: Path) -> dict:
    """Writes every table as <stem>.csv + <stem>.md under out_dir. Returns
    {stem: DataFrame}."""

    frames = {}
    for stem, builder in TABLE_BUILDERS.items():
        frame = builder(result)
        write_table(frame, out_dir, stem)
        frames[stem] = frame
    return frames
