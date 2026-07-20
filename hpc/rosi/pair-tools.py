#!/usr/bin/env python3
"""Validate paired Rosi manifests and derive allocation/split metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import yaml


EXPECTED_PAIRS = 3
EXPECTED_RUNS = {
    "gpu": ("cuda:nvidiaGpu", "gpuCuda"),
    "cpu": ("host:cpu", "cpuOmpBlocks"),
}


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def manifest_rows(path: Path) -> list[tuple[Path, Path, Path]]:
    if not path.is_absolute() or not path.is_file():
        fail(f"pair manifest must be an absolute readable file: {path}")
    rows: list[tuple[Path, Path, Path]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 3:
            fail(
                f"{path}:{line_number}: expected GPU_CONFIG CPU_CONFIG PAIR_OUTPUT "
                "without whitespace in paths"
            )
        gpu, cpu, output = map(Path, fields)
        if not gpu.is_absolute() or not gpu.is_file():
            fail(f"{path}:{line_number}: GPU config is not an absolute file: {gpu}")
        if not cpu.is_absolute() or not cpu.is_file():
            fail(f"{path}:{line_number}: CPU config is not an absolute file: {cpu}")
        if not output.is_absolute() or output == Path("/"):
            fail(f"{path}:{line_number}: pair output must be an absolute path other than /: {output}")
        rows.append((gpu.resolve(), cpu.resolve(), output.resolve()))
    if len(rows) != EXPECTED_PAIRS:
        fail(f"pair manifest must contain exactly {EXPECTED_PAIRS} entries, found {len(rows)}")
    outputs = [row[2] for row in rows]
    for index, output in enumerate(outputs):
        for other in outputs[:index]:
            if output == other or output.is_relative_to(other) or other.is_relative_to(output):
                fail(f"pair outputs must be distinct and non-overlapping: {other} and {output}")
    for index, (gpu, cpu, _output) in enumerate(rows):
        gpu_doc = validate_campaign(gpu, "gpu")
        cpu_doc = validate_campaign(cpu, "cpu")
        gpu_runs = [(run["name"], run["workload_id"]) for run in gpu_doc["runs"]]
        cpu_runs = [(run["name"], run["workload_id"]) for run in cpu_doc["runs"]]
        if gpu_runs != cpu_runs:
            fail(f"pair {index}: GPU and CPU campaigns must list identical workloads in order")
        if gpu_doc["revisions"] != cpu_doc["revisions"]:
            fail(f"pair {index}: GPU and CPU campaign revisions differ")
    return rows


def option_value(argv: list[str], name: str) -> str | None:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        return None
    return str(argv[positions[0] + 1])


def validate_campaign(path: Path, kind: str) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exception:
        fail(f"cannot load {kind} campaign {path}: {exception}")
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        fail(f"{path}: campaign schema_version must be 1")
    platform_value = document.get("platform")
    if not isinstance(platform_value, dict) or platform_value.get("device_class") != kind:
        fail(f"{path}: platform.device_class must be {kind}")
    revisions = document.get("revisions")
    if not isinstance(revisions, dict) or set(revisions) < {"alpakatune", "alpaka", "collector"}:
        fail(f"{path}: campaign revisions are incomplete")
    runs = document.get("runs")
    if not isinstance(runs, list) or not runs:
        fail(f"{path}: campaign runs must be a non-empty list")
    expected_backend, expected_executor = EXPECTED_RUNS[kind]
    for run in runs:
        if not isinstance(run, dict) or not run.get("name") or not run.get("workload_id"):
            fail(f"{path}: every run needs name and workload_id")
        argv = run.get("command")
        if not isinstance(argv, list):
            fail(f"{path}: {run.get('name', '<unknown>')} command must be an argv list")
        values = list(map(str, argv))
        if option_value(values, "--backend") != expected_backend:
            fail(f"{path}: {run['name']} must use --backend {expected_backend}")
        if option_value(values, "--executor") != expected_executor:
            fail(f"{path}: {run['name']} must use --executor {expected_executor}")
    return document


def command_lines(command: list[str]) -> list[str]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exception:
        fail(f"cannot inspect allocated hardware with {' '.join(command)}: {exception}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def hardware() -> dict[str, Any]:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu_rows = command_lines(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not gpu_rows:
        fail("the exclusive GPU allocation exposes no NVIDIA GPU")
    gpus = []
    for row in gpu_rows:
        fields = [value.strip() for value in row.split(",")]
        if len(fields) != 7:
            fail(f"unexpected nvidia-smi row: {row}")
        gpus.append(
            dict(
                zip(
                    ("index", "uuid", "name", "pci_bus_id", "driver_version", "memory_mib", "compute_capability"),
                    fields,
                    strict=True,
                )
            )
        )
    node = os.environ.get("SLURMD_NODENAME") or platform.node()
    identity = {
        "node": node,
        "cpu_model": cpu_model,
        "gpu_uuids": [gpu["uuid"] for gpu in gpus],
    }
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "identity_sha256": identity_digest,
        "node": node,
        "hostname": platform.node(),
        "cpu": {
            "model": cpu_model,
            "logical_cpus": os.cpu_count(),
            "machine": platform.machine(),
        },
        "gpus": gpus,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "module_stack": [
            module
            for module in os.environ.get("LOADEDMODULES", "").split(":")
            if module
        ],
    }


def record_allocation(output: Path, pair_index: int | None, kind: str) -> None:
    metadata_path = output / "allocation.json"
    current = hardware()
    attempt = {
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("hardware", {}).get("identity_sha256") != current["identity_sha256"]:
            fail(
                f"resume allocation differs from {metadata_path}; use a new output root "
                "instead of mixing nodes or devices"
            )
        if previous.get("pair_index") != pair_index or previous.get("kind") != kind:
            fail(f"allocation metadata contract differs: {metadata_path}")
        attempts = list(previous.get("attempts", []))
    else:
        attempts = []
    if attempt not in attempts:
        attempts.append(attempt)
    atomic_json(
        metadata_path,
        {
            "schema_version": 1,
            "kind": kind,
            "pair_index": pair_index,
            "hardware": current,
            "attempts": attempts,
        },
    )


def completed_campaign(path: Path, expected_class: str) -> str:
    try:
        campaign = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        fail(f"cannot read completed campaign {path}: {exception}")
    if campaign.get("status") != "completed":
        fail(f"campaign is not completed: {path}")
    platform_value = campaign.get("platform")
    if not isinstance(platform_value, dict) or platform_value.get("device_class") != expected_class:
        fail(f"campaign device class is not {expected_class}: {path}")
    device_id = platform_value.get("device_id")
    if not isinstance(device_id, str) or not device_id or device_id == "auto":
        fail(f"campaign has no resolved device_id: {path}")
    return device_id


def prepare_splits(manifest: Path, output: Path) -> None:
    rows = manifest_rows(manifest)
    split_names = ("train", "validation", "test")
    split_document: dict[str, Any] = {"schema_version": 1}
    all_devices: list[str] = []
    pair_metadata = []
    for index, (_gpu_config, _cpu_config, pair_root) in enumerate(rows):
        allocation_path = pair_root / "allocation.json"
        try:
            allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            fail(f"cannot read pair allocation {allocation_path}: {exception}")
        if allocation.get("kind") != "collection_pair" or allocation.get("pair_index") != index:
            fail(f"pair allocation index/contract mismatch: {allocation_path}")
        gpu_id = completed_campaign(pair_root / "gpu/campaign.json", "gpu")
        cpu_id = completed_campaign(pair_root / "cpu/campaign.json", "cpu")
        devices = [cpu_id, gpu_id]
        split_document[split_names[index]] = {"devices": devices}
        all_devices.extend(devices)
        pair_metadata.append(
            {
                "pair_index": index,
                "split": split_names[index],
                "node": allocation["hardware"]["node"],
                "cpu_device_id": cpu_id,
                "gpu_device_id": gpu_id,
            }
        )
    if len(set(all_devices)) != 2 * EXPECTED_PAIRS:
        fail(f"three disjoint CPU/GPU pairs require six unique device IDs, observed {all_devices}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(split_document, sort_keys=False), encoding="utf-8")
    temporary.replace(output)
    atomic_json(output.with_suffix(output.suffix + ".pairs.json"), pair_metadata)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-manifest")
    validate.add_argument("manifest", type=Path)
    roots = commands.add_parser("roots")
    roots.add_argument("manifest", type=Path)
    record = commands.add_parser("record-allocation")
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--pair-index", type=int)
    record.add_argument("--kind", required=True)
    splits = commands.add_parser("prepare-splits")
    splits.add_argument("manifest", type=Path)
    splits.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    try:
        if arguments.command == "validate-manifest":
            rows = manifest_rows(arguments.manifest)
            print(f"validated {len(rows)} paired collection tasks")
        elif arguments.command == "roots":
            for _gpu, _cpu, output in manifest_rows(arguments.manifest):
                print(output / "gpu")
                print(output / "cpu")
        elif arguments.command == "record-allocation":
            if arguments.pair_index is not None and arguments.pair_index not in range(EXPECTED_PAIRS):
                fail("pair index must be 0, 1, or 2")
            record_allocation(arguments.output, arguments.pair_index, arguments.kind)
        elif arguments.command == "prepare-splits":
            prepare_splits(arguments.manifest, arguments.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
