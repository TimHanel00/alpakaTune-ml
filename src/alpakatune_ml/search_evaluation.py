"""Compare locality/search strategies against immutable exhaustive oracle labels."""

from __future__ import annotations

from collections import defaultdict
import statistics
from pathlib import Path
from typing import Any, Sequence

from .contracts import HistorySurface, discover_histories, load_history
from .dataset import load_split_manifest
from .util import canonical_json_bytes, write_json


def evaluate_search_histories(
    oracle_manifest: Path,
    history_inputs: list[Path],
    output: Path,
    budgets: Sequence[int] = (16, 32, 64, 128, 256, 512, 1024),
) -> dict[str, Any]:
    manifest, oracle_rows = load_split_manifest(oracle_manifest)
    oracle: dict[str, dict[bytes, float]] = defaultdict(dict)
    for row in oracle_rows:
        oracle[row["surface_id"]][canonical_json_bytes(row["configuration"])] = float(
            row["runtime_seconds"]
        )
    records = []
    for history_path in discover_histories(history_inputs):
        for surface in load_history(history_path):
            strategy = str(surface.metadata.get("strategy", "unknown"))
            if strategy == "exhaustive" or surface.surface_id not in oracle:
                continue
            oracle_surface = oracle[surface.surface_id]
            oracle_best = min(oracle_surface.values())
            configurations = surface.cache.get("candidate_configurations", [])
            estimates = surface.cache.get("candidate_estimates", [])
            measured = []
            for index, estimate in enumerate(estimates):
                if estimate is None or index >= len(configurations):
                    continue
                runtime = oracle_surface.get(canonical_json_bytes(configurations[index]))
                if runtime is not None:
                    measured.append(runtime)
            if not measured:
                continue
            budget_regret = {}
            improvements = sorted(
                surface.cache.get("best_improvements", []),
                key=lambda item: item.get("retired_configuration_count", 0),
            )
            for budget in budgets:
                eligible = [
                    item
                    for item in improvements
                    if int(item.get("retired_configuration_count", 0)) <= budget
                ]
                if eligible:
                    index = int(eligible[-1]["candidate_index"])
                    if index < len(configurations):
                        runtime = oracle_surface.get(canonical_json_bytes(configurations[index]))
                        if runtime is not None:
                            budget_regret[str(budget)] = runtime / oracle_best - 1.0
            records.append(
                {
                    "strategy": strategy,
                    "surface_id": surface.surface_id,
                    "measured_candidate_count": len(measured),
                    "final_regret": min(measured) / oracle_best - 1.0,
                    "budget_regret": budget_regret,
                    "completion_reason": surface.cache.get("completion_reason"),
                }
            )
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_strategy[record["strategy"]].append(record)
    summary = {}
    for strategy, values in sorted(by_strategy.items()):
        budget_summary = {}
        for budget in budgets:
            regrets = [
                value["budget_regret"][str(budget)]
                for value in values
                if str(budget) in value["budget_regret"]
            ]
            if regrets:
                budget_summary[str(budget)] = {
                    "surface_count": len(regrets),
                    "median_regret": statistics.median(regrets),
                }
        summary[strategy] = {
            "surface_count": len(values),
            "median_final_regret": statistics.median(
                value["final_regret"] for value in values
            ),
            "median_measured_candidate_count": statistics.median(
                value["measured_candidate_count"] for value in values
            ),
            "by_candidate_budget": budget_summary,
        }
    result = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "oracle_split": manifest["split"],
        "strategies": summary,
        "surfaces": records,
        "notes": [
            "Regret uses exhaustive oracle runtime for each selected configuration.",
            "Schema-v9 budget trajectories include only persisted best-improvement events.",
        ],
    }
    write_json(output, result)
    return result

