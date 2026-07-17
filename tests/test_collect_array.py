from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).parents[1] / "hpc/rosi/collect-array.sbatch"


def _array_environment(
    tmp_path: Path,
    entries: list[tuple[Path, Path]],
    *,
    task_id: int = 0,
) -> tuple[dict[str, str], Path, Path]:
    source = tmp_path / "alpakaTune-ml"
    fake_python = source / ".venv/bin/python"
    fake_python.parent.mkdir(parents=True)
    capture = tmp_path / "python-arguments.txt"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"${ARRAY_CAPTURE}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    bash_environment = tmp_path / "bash-environment.sh"
    bash_environment.write_text("module() { return 0; }\n", encoding="utf-8")
    manifest = tmp_path / "campaign-array.txt"
    manifest.write_text(
        "".join(f"{config} {output}\n" for config, output in entries),
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    environment = {
        **os.environ,
        "BASH_ENV": str(bash_environment),
        "ARRAY_CAPTURE": str(capture),
        "SLURM_ARRAY_TASK_ID": str(task_id),
        "SLURM_ARRAY_TASK_COUNT": str(len(entries)),
        "SLURM_ARRAY_TASK_MIN": "0",
        "SLURM_ARRAY_TASK_MAX": str(len(entries) - 1),
        "SLURMD_NODENAME": "test-node",
        "CAMPAIGN_ARRAY_FILE": str(manifest),
        "CAMPAIGN_ARRAY_SHA256": digest,
        "ALPAKATUNE_ML_SOURCE": str(source),
        "ALPAKATUNE_SOURCE": str(tmp_path / "alpakaTune"),
        "ALPAKATUNE_BUILD": str(tmp_path / "alpakaTune-build"),
    }
    return environment, capture, manifest


def _entries(tmp_path: Path, count: int = 2) -> list[tuple[Path, Path]]:
    entries = []
    for index in range(count):
        config = tmp_path / f"campaign-{index}.yaml"
        config.write_text("schema_version: 1\n", encoding="utf-8")
        entries.append((config, tmp_path / f"output-{index}"))
    return entries


def test_array_task_selects_exact_manifest_entry(tmp_path):
    entries = _entries(tmp_path)
    environment, capture, _manifest = _array_environment(
        tmp_path, entries, task_id=1
    )

    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "alpakatune_ml",
        "collect",
        str(entries[1][0].resolve()),
        "--output",
        str(entries[1][1].resolve()),
        "--resume",
    ]
    assert "array_task=1/2" in completed.stdout
    assert "node=test-node" in completed.stdout


def test_array_shape_must_cover_every_manifest_entry(tmp_path):
    entries = _entries(tmp_path)
    environment, _capture, _manifest = _array_environment(tmp_path, entries)
    environment["SLURM_ARRAY_TASK_COUNT"] = "1"
    environment["SLURM_ARRAY_TASK_MAX"] = "0"

    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "array shape must be exactly 0-1 (2 tasks)" in completed.stderr


def test_array_rejects_changed_manifest(tmp_path):
    entries = _entries(tmp_path, count=1)
    environment, _capture, manifest = _array_environment(tmp_path, entries)
    manifest.write_text(manifest.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "campaign array checksum mismatch" in completed.stderr


def test_array_rejects_nested_output_roots(tmp_path):
    entries = _entries(tmp_path)
    entries[1] = (entries[1][0], entries[0][1] / "nested")
    environment, _capture, _manifest = _array_environment(tmp_path, entries)

    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "campaign outputs must be distinct and non-overlapping" in completed.stderr
