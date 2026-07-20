"""Build immutable, leakage-checked device or configuration dataset splits."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

import yaml

from .contracts import ContractError, discover_histories, load_history
from .util import canonical_json_bytes, sha256_file, stable_id, write_json


SPLIT_SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "validation", "test")


def _load_split_document(path: Path) -> Mapping[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exception:
        raise ContractError(f"cannot load split config {path}: {exception}") from exception
    if not isinstance(document, Mapping) or document.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ContractError(f"{path}: split schema_version must be {SPLIT_SCHEMA_VERSION}")
    return document


def load_split_config(path: Path) -> dict[str, tuple[str, ...]]:
    document = _load_split_document(path)
    if document.get("policy", "whole_device") != "whole_device":
        raise ContractError(f"{path}: expected a whole_device split policy")
    result: dict[str, tuple[str, ...]] = {}
    for name in SPLIT_NAMES:
        value = document.get(name)
        if not isinstance(value, Mapping) or not isinstance(value.get("devices"), list):
            raise ContractError(f"{path}: {name}.devices must be a list")
        devices = tuple(map(str, value["devices"]))
        if not devices or len(set(devices)) != len(devices):
            raise ContractError(f"{path}: {name}.devices must be non-empty and unique")
        result[name] = devices
    owners: dict[str, str] = {}
    for split, devices in result.items():
        for device in devices:
            previous = owners.get(device)
            if previous is not None:
                raise ContractError(
                    f"{path}: device {device!r} leaks across {previous} and {split}"
                )
            owners[device] = split
    return result


def _configuration_policy(path: Path) -> tuple[int, dict[str, float]]:
    document = _load_split_document(path)
    if document.get("policy") != "configuration":
        raise ContractError(f"{path}: expected a configuration split policy")
    seed = document.get("seed")
    fractions = document.get("fractions")
    if not isinstance(seed, int) or not isinstance(fractions, Mapping):
        raise ContractError(f"{path}: configuration policy needs integer seed and fractions")
    parsed = {name: float(fractions.get(name, 0.0)) for name in SPLIT_NAMES}
    if any(value <= 0.0 for value in parsed.values()) or abs(sum(parsed.values()) - 1.0) > 1.0e-9:
        raise ContractError(f"{path}: configuration fractions must be positive and sum to one")
    return seed, parsed


def _ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ContractError(f"dataset output must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def build_dataset(
    history_inputs: list[Path], split_config: Path, output: Path
) -> dict[str, Any]:
    """Validate exhaustive histories and write one immutable JSONL shard per split."""
    document = _load_split_document(split_config)
    if document.get("policy", "whole_device") == "configuration":
        return _build_configuration_dataset(history_inputs, split_config, output)
    return _build_whole_device_dataset(history_inputs, split_config, output)


def _build_whole_device_dataset(
    history_inputs: list[Path], split_config: Path, output: Path
) -> dict[str, Any]:
    splits = load_split_config(split_config)
    device_to_split = {
        device: split for split, devices in splits.items() for device in devices
    }
    surfaces = []
    for history in discover_histories(history_inputs):
        for surface in load_history(history):
            surface.validate_full_exhaustive()
            surfaces.append(surface)
    observed_devices = {surface.device_id for surface in surfaces}
    configured_devices = set(device_to_split)
    missing = observed_devices - configured_devices
    unused = configured_devices - observed_devices
    if missing or unused:
        details = []
        if missing:
            details.append(f"unassigned observed devices: {sorted(missing)}")
        if unused:
            details.append(f"configured devices without data: {sorted(unused)}")
        raise ContractError("split/device mismatch: " + "; ".join(details))

    _ensure_empty_output(output)
    streams = {
        name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in SPLIT_NAMES
    }
    seen_rows: dict[str, str] = {}
    split_state: dict[str, dict[str, Any]] = {
        name: {
            "row_count": 0,
            "surface_ids": set(),
            "device_ids": set(),
            "workload_ids": set(),
        }
        for name in SPLIT_NAMES
    }
    surface_summaries = []
    try:
        for surface in sorted(surfaces, key=lambda item: (item.device_id, item.workload_id, item.fingerprint)):
            split = device_to_split[surface.device_id]
            row_count = 0
            for row in surface.rows():
                previous = seen_rows.get(row["row_id"])
                if previous is not None:
                    raise ContractError(
                        f"duplicate row_id {row['row_id']} in {surface.source}; first seen in {previous}"
                    )
                seen_rows[row["row_id"]] = str(surface.source)
                streams[split].write(canonical_json_bytes(row).decode("utf-8") + "\n")
                row_count += 1
            state = split_state[split]
            state["row_count"] += row_count
            state["surface_ids"].add(surface.surface_id)
            state["device_ids"].add(surface.device_id)
            state["workload_ids"].add(surface.workload_id)
            surface_summaries.append(
                {
                    "surface_id": surface.surface_id,
                    "workload_id": surface.workload_id,
                    "device_id": surface.device_id,
                    "device_class": surface.device_class,
                    "candidate_count": surface.candidate_count,
                    "label_count": row_count,
                    "split": split,
                    "history_sha256": surface.source_sha256,
                    "history_schema_version": surface.history_schema_version,
                    "context_fingerprint": surface.fingerprint,
                }
            )
    finally:
        for stream in streams.values():
            stream.close()

    for name, state in split_state.items():
        if state["row_count"] == 0:
            raise ContractError(f"split {name} has no rows")

    dataset_id = stable_id(
        "dataset",
        {
            "surfaces": surface_summaries,
            "splits": {name: list(devices) for name, devices in splits.items()},
        },
    )
    split_manifests: dict[str, str] = {}
    for name, state in split_state.items():
        labels = output / f"{name}.jsonl"
        manifest = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "split": name,
            "split_policy": "whole_device",
            "device_ids": sorted(state["device_ids"]),
            "surface_ids": sorted(state["surface_ids"]),
            "workload_ids": sorted(state["workload_ids"]),
            "row_count": state["row_count"],
            "labels": labels.name,
            "labels_sha256": sha256_file(labels),
        }
        path = output / f"{name}.manifest.json"
        write_json(path, manifest)
        split_manifests[name] = path.name

    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split_policy": "whole_device",
        "split_config_sha256": sha256_file(split_config),
        "split_manifests": split_manifests,
        "row_count": len(seen_rows),
        "surfaces": surface_summaries,
    }
    write_json(output / "dataset.manifest.json", manifest)
    return manifest


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    identity = {key: value for key, value in first.items() if key not in {"runtime_seconds", "samples_seconds", "source"}}
    for row in rows[1:]:
        other = {key: value for key, value in row.items() if key not in {"runtime_seconds", "samples_seconds", "source"}}
        if canonical_json_bytes(other) != canonical_json_bytes(identity):
            raise ContractError(f"repeated row {first['row_id']} has inconsistent features")
    samples = [float(value) for row in rows for value in row.get("samples_seconds", ())]
    if not samples:
        samples = [float(row["runtime_seconds"]) for row in rows]
    result = dict(first)
    result["runtime_seconds"] = float(statistics.median(samples))
    result["samples_seconds"] = samples
    result["source"] = {
        "replicate_count": len(rows),
        "histories": [row.get("source", {}) for row in rows],
    }
    return result


def _build_configuration_dataset(
    history_inputs: list[Path], split_config: Path, output: Path
) -> dict[str, Any]:
    seed, fractions = _configuration_policy(split_config)
    replicas: dict[str, list[dict[str, Any]]] = {}
    for history in discover_histories(history_inputs):
        for surface in load_history(history):
            surface.validate_full_exhaustive()
            for row in surface.rows():
                replicas.setdefault(row["row_id"], []).append(row)
    rows = [_aggregate_rows(group) for group in replicas.values()]
    by_surface: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_surface.setdefault(row["surface_id"], []).append(row)

    assigned = {name: [] for name in SPLIT_NAMES}
    for surface_id, surface_rows in sorted(by_surface.items()):
        if len(surface_rows) < len(SPLIT_NAMES):
            raise ContractError(f"surface {surface_id} has fewer than three configurations")
        ordered = sorted(
            surface_rows,
            key=lambda row: hashlib.sha256(f"{seed}:{row['row_id']}".encode()).digest(),
        )
        validation_count = max(1, round(len(ordered) * fractions["validation"]))
        test_count = max(1, round(len(ordered) * fractions["test"]))
        if validation_count + test_count >= len(ordered):
            validation_count = test_count = 1
        assigned["validation"].extend(ordered[:validation_count])
        assigned["test"].extend(ordered[validation_count : validation_count + test_count])
        assigned["train"].extend(ordered[validation_count + test_count :])

    _ensure_empty_output(output)
    dataset_id = stable_id(
        "dataset",
        {
            "policy": "configuration",
            "seed": seed,
            "rows": sorted(
                hashlib.sha256(canonical_json_bytes(row)).hexdigest() for row in rows
            ),
            "fractions": fractions,
        },
    )
    split_manifests: dict[str, str] = {}
    for name in SPLIT_NAMES:
        split_rows = sorted(assigned[name], key=lambda row: (row["surface_id"], row["candidate_index"]))
        labels = output / f"{name}.jsonl"
        labels.write_text(
            "".join(canonical_json_bytes(row).decode("utf-8") + "\n" for row in split_rows),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "split": name,
            "split_policy": "configuration",
            "device_ids": sorted({row["device_id"] for row in split_rows}),
            "surface_ids": sorted({row["surface_id"] for row in split_rows}),
            "workload_ids": sorted({row["workload_id"] for row in split_rows}),
            "row_count": len(split_rows),
            "labels": labels.name,
            "labels_sha256": sha256_file(labels),
        }
        path = output / f"{name}.manifest.json"
        write_json(path, manifest)
        split_manifests[name] = path.name
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split_policy": "configuration",
        "split_config_sha256": sha256_file(split_config),
        "split_manifests": split_manifests,
        "row_count": len(rows),
        "replicate_label_count": sum(len(group) for group in replicas.values()),
    }
    write_json(output / "dataset.manifest.json", manifest)
    return manifest


def load_split_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise ContractError(f"cannot load split manifest {path}: {exception}") from exception
    labels = path.parent / str(manifest.get("labels", ""))
    if sha256_file(labels) != manifest.get("labels_sha256"):
        raise ContractError(f"split labels checksum mismatch: {labels}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with labels.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exception:
                raise ContractError(f"{labels}:{line_number}: invalid JSON: {exception}") from exception
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or row_id in seen:
                raise ContractError(f"{labels}:{line_number}: missing or duplicate row_id")
            seen.add(row_id)
            if row.get("device_id") not in manifest.get("device_ids", []):
                raise ContractError(f"{labels}:{line_number}: device is outside split manifest")
            rows.append(row)
    if len(rows) != manifest.get("row_count"):
        raise ContractError(f"{labels}: row_count does not match manifest")
    return manifest, rows


def validate_split_set(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Load train/validation/test and reject device, surface, or row leakage."""
    if len(paths) != 3:
        raise ContractError("exactly three split manifests are required")
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for path in paths:
        manifest, rows = load_split_manifest(path)
        name = manifest.get("split")
        if name not in SPLIT_NAMES or name in loaded:
            raise ContractError(f"expected one manifest for each of {SPLIT_NAMES}")
        loaded[name] = (manifest, rows)
    if set(loaded) != set(SPLIT_NAMES):
        raise ContractError(f"expected one manifest for each of {SPLIT_NAMES}")
    dataset_ids = {manifest["dataset_id"] for manifest, _ in loaded.values()}
    if len(dataset_ids) != 1:
        raise ContractError("split manifests do not belong to the same dataset")
    policies = {manifest.get("split_policy", "whole_device") for manifest, _ in loaded.values()}
    if len(policies) != 1:
        raise ContractError("split manifests use different split policies")
    policy = policies.pop()
    owners: dict[tuple[str, str], str] = {}
    for split, (manifest, rows) in loaded.items():
        for kind, values in (
            ("device", manifest["device_ids"]),
            ("surface", manifest["surface_ids"]),
            ("row", (row["row_id"] for row in rows)),
        ):
            if policy == "configuration" and kind != "row":
                continue
            for value in values:
                key = kind, value
                if key in owners:
                    raise ContractError(
                        f"{kind} {value!r} leaks across {owners[key]} and {split}"
                    )
                owners[key] = split
    return {name: rows for name, (_manifest, rows) in loaded.items()}
