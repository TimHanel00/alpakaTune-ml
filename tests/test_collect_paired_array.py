from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "hpc/rosi/collect-paired-array.sbatch"
PAIR_TOOLS = ROOT / "hpc/rosi/pair-tools.py"


def _campaign(path: Path, kind: str) -> None:
    backend, executor = {
        "gpu": ("cuda:nvidiaGpu", "gpuCuda"),
        "cpu": ("host:cpu", "cpuOmpBlocks"),
    }[kind]
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": f"test-{kind}",
                "base_tuning_config": "/source/config.yaml",
                "revisions": {
                    "alpakatune": "1234567",
                    "alpaka": "2345678",
                    "collector": "3456789",
                },
                "platform": {"device_id": "auto", "device_class": kind},
                "runs": [
                    {
                        "name": "vectorAdd",
                        "workload_id": "vectorAdd/default",
                        "command": [
                            "/build/vectorAdd",
                            "--backend",
                            backend,
                            "--executor",
                            executor,
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _paired_environment(tmp_path: Path, *, task_id: int = 1):
    source = tmp_path / "alpakaTune-ml"
    fake_python = source / ".venv/bin/python"
    fake_python.parent.mkdir(parents=True)
    capture = tmp_path / "calls.txt"
    fake_python.write_text(
        """#!/usr/bin/env bash
if [[ "$1" == */pair-tools.py ]]; then
    exec "${SYSTEM_PYTHON}" "$@"
fi
printf 'CALL\\n' >> "${PAIRED_CAPTURE}"
printf '%s\\n' "$@" >> "${PAIRED_CAPTURE}"
printf 'END\\n' >> "${PAIRED_CAPTURE}"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    bash_environment = tmp_path / "bash-environment.sh"
    bash_environment.write_text("module() { return 0; }\n", encoding="utf-8")
    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    nvidia_smi = binary_directory / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\necho '0, GPU-test-uuid, Test GPU, 00000000:01:00.0, 999.1, 81920, 9.0'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)

    gpu = tmp_path / "gpu.yaml"
    cpu = tmp_path / "cpu.yaml"
    _campaign(gpu, "gpu")
    _campaign(cpu, "cpu")
    outputs = [tmp_path / f"pair-{index}" for index in range(3)]
    manifest = tmp_path / "pairs.txt"
    manifest.write_text(
        "".join(f"{gpu} {cpu} {output}\n" for output in outputs), encoding="utf-8"
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    environment = {
        **os.environ,
        "PATH": f"{binary_directory}:{os.environ['PATH']}",
        "BASH_ENV": str(bash_environment),
        "SYSTEM_PYTHON": sys.executable,
        "PAIRED_CAPTURE": str(capture),
        "SLURM_ARRAY_TASK_ID": str(task_id),
        "SLURM_ARRAY_TASK_COUNT": "3",
        "SLURM_ARRAY_TASK_MIN": "0",
        "SLURM_ARRAY_TASK_MAX": "2",
        "SLURM_JOB_ID": "22",
        "SLURM_ARRAY_JOB_ID": "20",
        "SLURMD_NODENAME": "test-node",
        "PAIR_MANIFEST": str(manifest),
        "PAIR_MANIFEST_SHA256": digest,
        "ALPAKATUNE_ML_SOURCE": str(source),
    }
    return environment, capture, manifest, outputs, gpu, cpu


def _captured_calls(path: Path) -> list[list[str]]:
    calls = []
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "CALL":
            current = []
        elif line == "END":
            calls.append(current)
            current = None
        else:
            current.append(line)
    return calls


def test_paired_task_runs_gpu_then_cpu_and_records_allocation(tmp_path):
    environment, capture, _manifest, outputs, gpu, cpu = _paired_environment(tmp_path)
    completed = subprocess.run(
        ["bash", str(SCRIPT)], env=environment, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert _captured_calls(capture) == [
        ["-m", "alpakatune_ml", "collect", str(gpu), "--output", str(outputs[1] / "gpu"), "--resume"],
        ["-m", "alpakatune_ml", "collect", str(cpu), "--output", str(outputs[1] / "cpu"), "--resume"],
    ]
    allocation = json.loads((outputs[1] / "allocation.json").read_text(encoding="utf-8"))
    assert allocation["pair_index"] == 1
    assert allocation["hardware"]["node"] == "test-node"
    assert allocation["hardware"]["gpus"][0]["uuid"] == "GPU-test-uuid"


def test_paired_task_requires_exact_three_task_array(tmp_path):
    environment, _capture, _manifest, _outputs, _gpu, _cpu = _paired_environment(tmp_path)
    environment["SLURM_ARRAY_TASK_COUNT"] = "2"
    environment["SLURM_ARRAY_TASK_MAX"] = "1"
    completed = subprocess.run(
        ["bash", str(SCRIPT)], env=environment, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 2
    assert "exactly tasks 0-2" in completed.stderr


def test_pair_tools_assign_task_order_to_splits(tmp_path):
    _environment, _capture, manifest, outputs, _gpu, _cpu = _paired_environment(tmp_path)
    for index, output in enumerate(outputs):
        (output / "gpu").mkdir(parents=True)
        (output / "cpu").mkdir()
        (output / "allocation.json").write_text(
            json.dumps(
                {
                    "kind": "collection_pair",
                    "pair_index": index,
                    "hardware": {"node": f"node-{index}"},
                }
            ),
            encoding="utf-8",
        )
        for kind in ("gpu", "cpu"):
            (output / kind / "campaign.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "platform": {
                            "device_class": kind,
                            "device_id": f"{kind}-{index}",
                        },
                    }
                ),
                encoding="utf-8",
            )
    splits = tmp_path / "derived-splits.yaml"
    completed = subprocess.run(
        [sys.executable, str(PAIR_TOOLS), "prepare-splits", str(manifest), "--output", str(splits)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    document = yaml.safe_load(splits.read_text(encoding="utf-8"))
    assert document == {
        "schema_version": 1,
        "train": {"devices": ["cpu-0", "gpu-0"]},
        "validation": {"devices": ["cpu-1", "gpu-1"]},
        "test": {"devices": ["cpu-2", "gpu-2"]},
    }


def test_pair_tools_reject_reused_device_across_splits(tmp_path):
    _environment, _capture, manifest, outputs, _gpu, _cpu = _paired_environment(tmp_path)
    for index, output in enumerate(outputs):
        (output / "gpu").mkdir(parents=True)
        (output / "cpu").mkdir()
        (output / "allocation.json").write_text(
            json.dumps({"kind": "collection_pair", "pair_index": index, "hardware": {"node": f"node-{index}"}}),
            encoding="utf-8",
        )
        for kind in ("gpu", "cpu"):
            device_id = "same-gpu" if kind == "gpu" else f"cpu-{index}"
            (output / kind / "campaign.json").write_text(
                json.dumps({"status": "completed", "platform": {"device_class": kind, "device_id": device_id}}),
                encoding="utf-8",
            )
    completed = subprocess.run(
        [sys.executable, str(PAIR_TOOLS), "prepare-splits", str(manifest), "--output", str(tmp_path / "splits.yaml")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "six unique device IDs" in completed.stderr
    configuration_splits = tmp_path / "configuration-splits.yaml"
    completed = subprocess.run(
        [
            sys.executable,
            str(PAIR_TOOLS),
            "prepare-splits",
            str(manifest),
            "--output",
            str(configuration_splits),
            "--policy",
            "configuration",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    document = yaml.safe_load(configuration_splits.read_text(encoding="utf-8"))
    assert document["policy"] == "configuration"
    assert document["fractions"] == {"train": 0.8, "validation": 0.1, "test": 0.1}
