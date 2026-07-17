"""Reference inference benchmark hook; native alpakaTune remains the release gate."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import load_split_manifest
from .evaluation import NumpyEnsemble
from .util import write_json


def benchmark_reference(
    artifact: Path,
    split_manifest: Path,
    output: Path,
    *,
    candidates: int = 256,
    iterations: int = 10,
) -> dict[str, Any]:
    _manifest, rows = load_split_manifest(split_manifest)
    rows = rows[: min(candidates, len(rows))]
    if not rows or iterations <= 0:
        raise ValueError("benchmark needs candidates and positive iterations")
    ensemble = NumpyEnsemble(artifact)
    ensemble.predict(rows)
    scoring_samples = []
    last_predictions = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        last_predictions, _uncertainty, _embeddings = ensemble.predict(rows)
        scoring_samples.append(time.perf_counter_ns() - started)
    assert last_predictions is not None
    order = np.argsort(last_predictions).tolist()
    recommendation_samples = []
    cursor = 0
    for _ in range(max(iterations * len(rows), 1000)):
        started = time.perf_counter_ns()
        _candidate = order[cursor]
        cursor = (cursor + 1) % len(order)
        recommendation_samples.append(time.perf_counter_ns() - started)
    result = {
        "schema_version": 1,
        "implementation": "python_numpy_reference_not_native_release_gate",
        "candidate_count": len(rows),
        "iterations": iterations,
        "full_batch_scoring_median_milliseconds": statistics.median(scoring_samples) / 1.0e6,
        "scoring_median_microseconds_per_candidate": statistics.median(scoring_samples)
        / len(rows)
        / 1.0e3,
        "cached_recommend_median_microseconds": statistics.median(recommendation_samples) / 1.0e3,
        "targets": {
            "native_scoring_microseconds_per_candidate": [1.0, 5.0],
            "native_cached_recommend_microseconds_max": 10.0,
        },
    }
    write_json(output, result)
    return result

