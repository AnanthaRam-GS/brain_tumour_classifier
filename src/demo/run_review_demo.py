"""CLI for the Review 1 single-image inference demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.demo.inference import DemoInferenceError, run_demo_inference
from src.demo.pipeline_registry import available_pipeline_ids, list_pipeline_statuses
from src.demo.sample_selection import DemoSampleSelectionError
from src.demo.visualize_demo import save_demo_outputs


def _format_status(status: str) -> str:
    return status.replace("_", " ").upper()


def print_pipeline_statuses() -> None:
    """Print registered pipelines and local readiness."""

    for spec in list_pipeline_statuses():
        print(f"{spec.display_name}")
        print(f"Status: {_format_status(spec.status)}")
        if spec.default_model_artifact is not None:
            print(f"Artifact: {spec.default_model_artifact}")
        print()


def _print_demo(prediction, outputs: dict[str, Path]) -> None:
    print("=" * 60)
    print("BRAIN TUMOR CLASSIFICATION - REVIEW 1 LIVE DEMO")
    print("=" * 60)
    print(f"Pipeline      : {prediction.feature_name} + {prediction.model_name}")
    print()
    print(f"Sample ID     : {prediction.sample_id}")
    print(f"Patient ID    : {prediction.patient_id}")
    print("Dataset Split : TEST")
    print()
    print(f"Actual Class  : {prediction.true_class_name}")
    print()
    print("MODEL PREDICTION")
    print(f"Predicted     : {prediction.predicted_class_name}")
    if prediction.confidence is None:
        print("Confidence    : Unavailable")
    else:
        print(f"Confidence    : {prediction.confidence * 100:.2f}%")
    if prediction.class_probabilities:
        print()
        print("Class probabilities:")
        for name, value in prediction.class_probabilities.items():
            print(f"{name:<13}: {value * 100:.2f}%")
    print()
    print(f"Result        : {'CORRECT' if prediction.correct else 'INCORRECT'}")
    print()
    print(f"Figure        : {outputs['figure']}")
    print(f"Metadata      : {outputs['metadata']}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=available_pipeline_ids())
    parser.add_argument("--sample-id", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--random", action="store_true", dest="random_selection")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list-pipelines", action="store_true")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_pipelines:
        print_pipeline_statuses()
        return 0
    if not args.pipeline:
        parser.error("--pipeline is required unless --list-pipelines is used")
    output_dir = args.output_dir or Path("reports/demo") / args.pipeline
    try:
        sample, roi, prediction = run_demo_inference(
            pipeline_id=args.pipeline,
            sample_id=args.sample_id,
            seed=args.seed,
            random_selection=args.random_selection,
        )
        outputs = save_demo_outputs(
            sample=sample,
            roi=roi,
            prediction=prediction,
            output_dir=output_dir,
        )
    except (DemoInferenceError, DemoSampleSelectionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_demo(prediction, outputs)
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
