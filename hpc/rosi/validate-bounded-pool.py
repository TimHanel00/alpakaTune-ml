#!/usr/bin/env python3
"""Validate bounded learned-pool diagnostics from a live benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TERMINAL_REASONS = {
    "all_configurations",
    "maximum_executions",
    "maximum_retired_configurations",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    arguments = parser.parse_args()

    failures: list[str] = []
    contexts: list[dict] = []
    histories = sorted(arguments.output.glob("*/learned_hybrid/history.json"))
    if not histories:
        failures.append("no learned_hybrid histories were found")

    for history_path in histories:
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            failures.append(f"cannot read {history_path}: {exception}")
            continue
        raw_contexts = history.get("contexts")
        if not isinstance(raw_contexts, dict) or not raw_contexts:
            failures.append(f"{history_path} contains no tuning contexts")
            continue
        for fingerprint, context in raw_contexts.items():
            learning = context.get("learning", {})
            metadata = context.get("metadata", {})
            kernel = metadata.get("kernel", fingerprint)
            summary = {
                "history": str(history_path),
                "fingerprint": fingerprint,
                "kernel": kernel,
                "completion_reason": context.get("completion_reason"),
                "status": learning.get("status"),
                "candidate_pool_capacity": learning.get("candidate_pool_capacity"),
                "candidate_batch_size": learning.get("candidate_batch_size"),
                "cached_candidate_count": learning.get("cached_candidate_count"),
                "peak_cached_candidate_count": learning.get(
                    "peak_cached_candidate_count"
                ),
                "scored_candidate_count": learning.get("scored_candidate_count"),
                "pool_refill_count": learning.get("pool_refill_count"),
                "candidate_stream_exhausted": learning.get(
                    "candidate_stream_exhausted"
                ),
            }
            contexts.append(summary)

            prefix = f"{history_path}: {kernel}:"
            if summary["status"] != "active":
                failures.append(f"{prefix} learned model was not active")
            if summary["candidate_pool_capacity"] != arguments.pool_size:
                failures.append(f"{prefix} pool capacity does not match")
            if summary["candidate_batch_size"] != arguments.batch_size:
                failures.append(f"{prefix} batch size does not match")
            peak = summary["peak_cached_candidate_count"]
            cached = summary["cached_candidate_count"]
            scored = summary["scored_candidate_count"]
            refills = summary["pool_refill_count"]
            if not isinstance(peak, int) or peak > arguments.pool_size:
                failures.append(f"{prefix} peak cached candidates exceeded the pool")
            if not isinstance(cached, int) or cached > arguments.pool_size:
                failures.append(f"{prefix} cached candidates exceeded the pool")
            if not isinstance(scored, int) or scored <= arguments.pool_size:
                failures.append(f"{prefix} candidate stream did not refill")
            if not isinstance(refills, int) or refills <= 1:
                failures.append(f"{prefix} fewer than two pool fills were recorded")
            if summary["completion_reason"] not in TERMINAL_REASONS:
                failures.append(f"{prefix} benchmark did not reach a terminal reason")

    report = {
        "valid": bool(contexts) and not failures,
        "pool_size": arguments.pool_size,
        "batch_size": arguments.batch_size,
        "contexts": contexts,
        "failures": failures,
    }
    report_path = arguments.output / "bounded-pool-validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for failure in failures:
        print(f"ERROR: {failure}")
    print(f"wrote {report_path}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
