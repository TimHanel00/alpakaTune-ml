"""Scheduler-neutral orchestration for unbiased exhaustive campaigns."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping

import yaml

from .contracts import ContractError, load_history
from .util import sha256_file, slug, write_json


CAMPAIGN_SCHEMA_VERSION = 1
MEASUREMENT_POLICY = {
    "strategy": "exhaustive",
    "warmup_runs": 1,
    "runs_per_candidate": 3,
    "minimum_runs_per_candidate": 3,
    "mann_whitney_early_stop": False,
    "max_consecutive_runs": 4,
}


def load_campaign(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exception:
        raise ContractError(f"cannot load campaign {path}: {exception}") from exception
    if not isinstance(document, Mapping) or document.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ContractError(f"{path}: schema_version must be {CAMPAIGN_SCHEMA_VERSION}")
    for name in ("name", "base_tuning_config", "revisions", "platform", "runs"):
        if name not in document:
            raise ContractError(f"{path}: missing required field {name}")
    revisions = document["revisions"]
    for name in ("alpakatune", "alpaka", "collector"):
        revision = str(revisions.get(name, "")) if isinstance(revisions, Mapping) else ""
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
            raise ContractError(f"{path}: revisions.{name} must be an immutable hexadecimal commit")
    platform = document["platform"]
    if not isinstance(platform, Mapping) or not platform.get("device_id") or platform.get("device_class") not in {"cpu", "gpu"}:
        raise ContractError(f"{path}: platform needs device_id and device_class cpu|gpu")
    runs = document["runs"]
    if not isinstance(runs, list) or not runs:
        raise ContractError(f"{path}: runs must be a non-empty list")
    names: set[str] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping) or not run.get("name") or not run.get("workload_id"):
            raise ContractError(f"{path}: runs[{index}] needs name and workload_id")
        if not isinstance(run.get("command"), list) or not run["command"]:
            raise ContractError(f"{path}: runs[{index}].command must be a non-empty argv list")
        if run["name"] in names:
            raise ContractError(f"{path}: duplicate run name {run['name']!r}")
        names.add(str(run["name"]))
    return dict(document)


def exhaustive_config(base_path: Path, history: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exception:
        raise ContractError(f"cannot load base tuner config {base_path}: {exception}") from exception
    if not isinstance(document, dict) or not isinstance(document.get("tuning"), dict):
        raise ContractError(f"{base_path}: expected a tuning map")
    tuning = document["tuning"]
    tuning.update(MEASUREMENT_POLICY)
    # Full coverage has no launch/retirement budget. The executable must keep
    # enqueueing until the tuner reports all_configurations.
    tuning.pop("maximum_executions", None)
    tuning.pop("maximum_retired_configurations", None)
    document["persistence"] = {"file": str(history.resolve())}
    return document


def _expand(value: Any) -> str:
    return os.path.expandvars(os.path.expanduser(str(value)))


def _successful_run(path: Path) -> bool:
    run_path = path / "run.json"
    history = path / "history.json"
    if not run_path.exists() or not history.exists():
        return False
    try:
        metadata = json.loads(run_path.read_text(encoding="utf-8"))
        return metadata.get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def resolve_device_id(expected: str, resolved: str | None, observed: str) -> str:
    """Resolve an auto device ID once and keep every run on the same device."""
    if expected != "auto" and observed != expected:
        raise ContractError(
            f"history device_id {observed!r} does not match campaign platform {expected!r}"
        )
    if resolved is not None and observed != resolved:
        raise ContractError(
            f"campaign moved between devices {resolved!r} and {observed!r}"
        )
    return observed


def validate_campaign_history(
    history: Path,
    platform: Mapping[str, Any],
    resolved_device_id: str | None,
) -> str:
    """Validate one run and atomically extend the campaign device binding."""
    expected_device_id = str(platform["device_id"])
    candidate_device_id = resolved_device_id
    surfaces = load_history(history)
    for surface in surfaces:
        surface.validate_full_exhaustive()
        candidate_device_id = resolve_device_id(
            expected_device_id, candidate_device_id, surface.device_id
        )
        if surface.device_class != platform["device_class"]:
            raise ContractError(
                f"history device_class {surface.device_class!r} does not match campaign platform"
            )
    if candidate_device_id is None:  # load_history already rejects an empty context map.
        raise ContractError(f"{history}: history did not resolve a device_id")
    return candidate_device_id


def collect_campaign(
    campaign_path: Path,
    output: Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    campaign = load_campaign(campaign_path)
    base_config = Path(_expand(campaign["base_tuning_config"]))
    environment_additions = {
        str(name): _expand(value)
        for name, value in dict(campaign.get("environment", {})).items()
    }
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "name": campaign["name"],
        "campaign_source": str(campaign_path.resolve()),
        "campaign_source_sha256": sha256_file(campaign_path),
        "revisions": campaign["revisions"],
        "platform": campaign["platform"],
        "measurement_policy": MEASUREMENT_POLICY,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "dry_run" if dry_run else "running",
        "runs": [],
    }
    failures = []
    expected_device_id = str(campaign["platform"]["device_id"])
    resolved_device_id = None if expected_device_id == "auto" else expected_device_id
    for specification in campaign["runs"]:
        name = str(specification["name"])
        directory = output / "runs" / slug(name)
        history = directory / "history.json"
        tuning_path = directory / "tuning.yaml"
        command = [_expand(item) for item in specification["command"]]
        working_directory = Path(_expand(specification.get("working_directory", directory)))
        if dry_run:
            summary["runs"].append(
                {"name": name, "workload_id": specification["workload_id"], "command": command}
            )
            continue
        if resume and _successful_run(directory):
            try:
                resolved_device_id = validate_campaign_history(
                    history, campaign["platform"], resolved_device_id
                )
                previous = json.loads(
                    (directory / "run.json").read_text(encoding="utf-8")
                )
                summary["runs"].append(
                    {
                        **previous,
                        "name": name,
                        "workload_id": specification["workload_id"],
                        "status": "skipped",
                        "previous_status": "completed",
                    }
                )
            except (ContractError, OSError, json.JSONDecodeError) as exception:
                failures.append(name)
                summary["runs"].append(
                    {
                        "name": name,
                        "workload_id": specification["workload_id"],
                        "status": "failed",
                        "validation_errors": [
                            f"completed resume history is invalid: {exception}"
                        ],
                    }
                )
            continue
        directory.mkdir(parents=True, exist_ok=True)
        tuning = exhaustive_config(base_config, history)
        tuning_path.write_text(yaml.safe_dump(tuning, sort_keys=False), encoding="utf-8")
        environment = os.environ.copy()
        environment.update(environment_additions)
        environment["ALPAKA_TUNE_CONFIG"] = str(tuning_path.resolve())
        started = dt.datetime.now(dt.timezone.utc)
        started_monotonic = time.monotonic()
        run_metadata = {
            "name": name,
            "workload_id": specification["workload_id"],
            "command": command,
            "working_directory": str(working_directory),
            "started_at": started.isoformat(),
            "status": "running",
        }
        write_json(directory / "run.json", run_metadata)
        try:
            with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, (
                directory / "stderr.log"
            ).open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=working_directory,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            return_code = completed.returncode
            error = None
        except OSError as exception:
            return_code = None
            error = str(exception)
        validation_errors = []
        if return_code == 0 and history.exists():
            try:
                resolved_device_id = validate_campaign_history(
                    history, campaign["platform"], resolved_device_id
                )
            except ContractError as exception:
                validation_errors.append(str(exception))
        else:
            validation_errors.append("executable failed or did not write history.json")
        status = "completed" if not validation_errors else "failed"
        run_metadata.update(
            {
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - started_monotonic,
                "return_code": return_code,
                "history_sha256": sha256_file(history) if history.exists() else None,
                "validation_errors": validation_errors,
                "status": status,
            }
        )
        if error is not None:
            run_metadata["error"] = error
        write_json(directory / "run.json", run_metadata)
        summary["runs"].append(run_metadata)
        if status != "completed":
            failures.append(name)
    summary["status"] = "failed" if failures else ("dry_run" if dry_run else "completed")
    summary["failures"] = failures
    if resolved_device_id is not None:
        summary["platform"] = {
            **campaign["platform"],
            "requested_device_id": expected_device_id,
            "device_id": resolved_device_id,
        }
    if not dry_run:
        summary["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(output / "campaign.json", summary)
    return summary
