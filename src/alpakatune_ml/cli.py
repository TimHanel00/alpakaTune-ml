"""Command line entry point for the complete offline ML workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .artifact import read_artifact
from .benchmark import benchmark_reference
from .collect import collect_campaign
from .contracts import ContractError, discover_histories, load_history
from .dataset import build_dataset, validate_split_set
from .evaluation import evaluate
from .plotting import plot_split
from .search_evaluation import evaluate_search_histories
from .training import (
    TrainingError,
    merge_member_artifacts,
    train,
    train_member,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpakatune-ml", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="run a full exhaustive campaign")
    collect.add_argument("campaign", type=Path)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--resume", action="store_true")
    collect.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate-histories", help="require complete exhaustive histories")
    validate.add_argument("inputs", nargs="+", type=Path)

    dataset = subparsers.add_parser("build-dataset", help="write strict whole-device splits")
    dataset.add_argument("inputs", nargs="+", type=Path)
    dataset.add_argument("--splits", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)

    split_validate = subparsers.add_parser("validate-splits", help="check all three split manifests")
    for name in ("train", "validation", "test"):
        split_validate.add_argument(f"--{name}", type=Path, required=True)

    training = subparsers.add_parser("train", help="train and export an ATMLART1 ensemble")
    for name in ("train", "validation", "test"):
        training.add_argument(f"--{name}", type=Path, required=True)
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)

    member = subparsers.add_parser(
        "train-member", help="train one deterministic ensemble member"
    )
    for name in ("train", "validation", "test"):
        member.add_argument(f"--{name}", type=Path, required=True)
    member.add_argument("--config", type=Path, required=True)
    member.add_argument("--member-index", type=int, required=True)
    member.add_argument("--output", type=Path, required=True)

    merge = subparsers.add_parser(
        "merge-members", help="validate and merge independently trained members"
    )
    for name in ("train", "validation", "test"):
        merge.add_argument(f"--{name}", type=Path, required=True)
    merge.add_argument("--config", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("members", nargs="+", type=Path)

    evaluation = subparsers.add_parser("evaluate", help="evaluate zero-shot and adapted ranking")
    evaluation.add_argument("--artifact", type=Path, required=True)
    evaluation.add_argument("--split", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument(
        "--adaptation-budgets", nargs="+", type=int, default=(16, 32, 64, 128, 256, 512, 1024)
    )

    plotting = subparsers.add_parser("plot", help="plot all exhaustive candidates by surface")
    plotting.add_argument("--split", type=Path, required=True)
    plotting.add_argument("--output", type=Path, required=True)
    plotting.add_argument("--artifact", type=Path)

    search = subparsers.add_parser(
        "evaluate-search", help="compare search histories with exhaustive oracle labels"
    )
    search.add_argument("inputs", nargs="+", type=Path)
    search.add_argument("--oracle", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--budgets", nargs="+", type=int, default=(16, 32, 64, 128, 256, 512, 1024))

    info = subparsers.add_parser("artifact-info", help="inspect and validate an artifact")
    info.add_argument("artifact", type=Path)

    benchmark = subparsers.add_parser(
        "benchmark-artifact", help="run the NumPy reference benchmark hook"
    )
    benchmark.add_argument("--artifact", type=Path, required=True)
    benchmark.add_argument("--split", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--candidates", type=int, default=256)
    benchmark.add_argument("--iterations", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "collect":
            summary = collect_campaign(
                arguments.campaign,
                arguments.output,
                resume=arguments.resume,
                dry_run=arguments.dry_run,
            )
            print(json.dumps(summary, indent=2))
            return 0 if summary["status"] in {"completed", "dry_run"} else 1
        if arguments.command == "validate-histories":
            count = 0
            labels = 0
            for path in discover_histories(arguments.inputs):
                for surface in load_history(path):
                    surface.validate_full_exhaustive()
                    count += 1
                    labels += sum(1 for _ in surface.rows())
            print(f"validated {count} complete surfaces with {labels} legal candidate labels")
            return 0
        if arguments.command == "build-dataset":
            manifest = build_dataset(arguments.inputs, arguments.splits, arguments.output)
            print(json.dumps(manifest, indent=2))
            return 0
        paths = [arguments.train, arguments.validation, arguments.test] if hasattr(arguments, "train") else []
        if arguments.command == "validate-splits":
            splits = validate_split_set(paths)
            print("validated " + ", ".join(f"{name}={len(rows)}" for name, rows in splits.items()))
            return 0
        if arguments.command == "train":
            card = train(paths, arguments.config, arguments.output)
            print(json.dumps(card, indent=2))
            return 0
        if arguments.command == "train-member":
            card = train_member(
                paths,
                arguments.config,
                arguments.member_index,
                arguments.output,
            )
            print(json.dumps(card, indent=2))
            return 0
        if arguments.command == "merge-members":
            card = merge_member_artifacts(
                arguments.members,
                paths,
                arguments.config,
                arguments.output,
            )
            print(json.dumps(card, indent=2))
            return 0
        if arguments.command == "evaluate":
            result = evaluate(
                arguments.artifact,
                arguments.split,
                arguments.output,
                arguments.adaptation_budgets,
            )
            print(json.dumps(result, indent=2))
            return 0
        if arguments.command == "plot":
            images = plot_split(arguments.split, arguments.output, arguments.artifact)
            print(f"wrote {len(images)} surface plots and {arguments.output / 'index.html'}")
            return 0
        if arguments.command == "evaluate-search":
            result = evaluate_search_histories(
                arguments.oracle, arguments.inputs, arguments.output, arguments.budgets
            )
            print(json.dumps(result, indent=2))
            return 0
        if arguments.command == "artifact-info":
            metadata, tensors = read_artifact(arguments.artifact)
            print(json.dumps({**metadata, "tensor_payloads": len(tensors)}, indent=2))
            return 0
        if arguments.command == "benchmark-artifact":
            result = benchmark_reference(
                arguments.artifact,
                arguments.split,
                arguments.output,
                candidates=arguments.candidates,
                iterations=arguments.iterations,
            )
            print(json.dumps(result, indent=2))
            return 0
    except (ContractError, TrainingError, RuntimeError, OSError, ValueError) as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
