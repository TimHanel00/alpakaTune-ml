from __future__ import annotations

import json
from pathlib import Path

from alpakatune_ml.search_evaluation import evaluate_search_histories
from alpakatune_ml.util import canonical_json_bytes, sha256_file, stable_id


def test_capped_exhaustive_is_reported_as_a_comparison_strategy(tmp_path: Path):
    device_id = "gpu-test"
    workload_id = "workload-test"
    surface_id = stable_id(
        "surface",
        {
            "workload_id": workload_id,
            "device_id": device_id,
            "context_fingerprint": "context",
        },
    )
    labels = tmp_path / "test.jsonl"
    rows = [
        {
            "schema_version": 1,
            "row_id": stable_id("row", {"surface_id": surface_id, "candidate_index": index}),
            "surface_id": surface_id,
            "workload_id": workload_id,
            "device_id": device_id,
            "device_class": "gpu",
            "candidate_index": index,
            "configuration": {"block": value},
            "dimension_feature_names": [],
            "dimension_features": [],
            "context_features": {"candidate_count_log1p": 1.0},
            "runtime_seconds": runtime,
            "samples_seconds": [runtime] * 3,
            "source": {},
        }
        for index, (value, runtime) in enumerate(((32, 2.0), (64, 1.0)))
    ]
    labels.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    manifest = tmp_path / "test.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "dataset-test",
                "split": "test",
                "labels": labels.name,
                "labels_sha256": sha256_file(labels),
                "row_count": len(rows),
                "device_ids": [device_id],
                "workload_ids": [workload_id],
                "surface_ids": [surface_id],
            }
        ),
        encoding="utf-8",
    )

    history_dir = tmp_path / "comparison" / "example" / "exhaustive"
    history_dir.mkdir(parents=True)
    (history_dir / "history.json").write_text(
        json.dumps(
            {
                "schema_version": 10,
                "contexts": {
                    "context": {
                        "candidate_count": 2,
                        "metadata": {
                            "strategy": "exhaustive",
                            "device_descriptor": {"id": device_id, "class": "gpu"},
                            "model_context": {
                                "feature_schema_version": 1,
                                "workload_id": workload_id,
                                "device_class": "gpu",
                                "context_features": {},
                                "dimensions": [],
                            },
                        },
                        "candidate_configurations": [{"block": 32}, {"block": 64}],
                        "candidate_estimates": [2.0, None],
                        "candidate_samples": [[2.0, 2.0, 2.0], []],
                        "best_improvements": [
                            {
                                "candidate_index": 0,
                                "retired_configuration_count": 1,
                            }
                        ],
                        "completion_reason": "maximum_executions",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_search_histories(
        manifest, [history_dir.parents[1]], tmp_path / "evaluation.json", budgets=(1,)
    )

    assert set(result["strategies"]) == {"exhaustive"}
    assert result["strategies"]["exhaustive"]["surface_count"] == 1
    assert result["strategies"]["exhaustive"]["median_final_regret"] == 1.0
