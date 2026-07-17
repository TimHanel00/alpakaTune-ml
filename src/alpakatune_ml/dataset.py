"""Build immutable, leakage-checked whole-device dataset splits."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import ContractError, discover_histories, load_history
from .util import canonical_json_bytes, sha256_bytes, sha256_file, stable_id, write_json


SPLIT_SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "validation", "test")


def load_split_config(path: Path) -> dict[str, tuple[str, ...]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exception:
        raise ContractError(f"cannot load split config {path}: {exception}") from exception
    if not isinstance(document, Mapping) or document.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ContractError(f"{path}: split schema_version must be {SPLIT_SCHEMA_VERSION}")
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


def _ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ContractError(f"dataset output must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def build_dataset(
    history_inputs: list[Path], split_config: Path, output: Path
) -> dict[str, Any]:
    """Validate exhaustive histories and write one immutable JSONL shard per split."""
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
    owners: dict[tuple[str, str], str] = {}
    for split, (manifest, rows) in loaded.items():
        for kind, values in (
            ("device", manifest["device_ids"]),
            ("surface", manifest["surface_ids"]),
            ("row", (row["row_id"] for row in rows)),
        ):
            for value in values:
                key = kind, value
                if key in owners:
                    raise ContractError(
                        f"{kind} {value!r} leaks across {owners[key]} and {split}"
                    )
                owners[key] = split
    return {name: rows for name, (_manifest, rows) in loaded.items()}

