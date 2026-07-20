from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).parents[1]
ROSI = ROOT / "hpc/rosi"


def test_setup_is_git_free_and_fully_disconnected():
    script = (ROSI / "setup-login.sh").read_text(encoding="utf-8")
    commands = [line.strip() for line in script.splitlines() if not line.lstrip().startswith("#")]
    assert not any(line == "git" or line.startswith("git ") for line in commands)
    assert "-DFETCHCONTENT_FULLY_DISCONNECTED=ON" in script
    assert '-DFETCHCONTENT_SOURCE_DIR_ALPAKA3="${alpaka_source}"' in script
    assert '-DFETCHCONTENT_SOURCE_DIR_YAML_CPP="${yaml_cpp_source}"' in script
    assert '-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="${nlohmann_json_source}"' in script
    assert "-Dalpaka_EXEC_CpuSerial=OFF" in script
    assert "-DalpakaTune_BUILD_TESTING=OFF" in script
    assert 'pip install -e "${ALPAKATUNE_ML_SOURCE}[plot]"' in script
    assert "import torch" in script


def test_allocation_metadata_records_loaded_modules():
    script = (ROSI / "pair-tools.py").read_text(encoding="utf-8")
    assert 'os.environ.get("LOADEDMODULES", "")' in script
    assert '"module_stack"' in script


def test_module_setup_accepts_optional_experiment_defaults():
    script = (ROSI / "modules.sh").read_text(encoding="utf-8")
    assert "declare -F module" in script
    assert "/etc/profile.d/lmod.sh" in script
    assert "/rosi/shared/lmod/lmod/init/bash" in script
    assert 'experiment-01.env.sh' in script
    assert 'source "${experiment_defaults}"' in script


def test_sbatch_copies_fall_back_to_the_synchronized_script_directory():
    for path in ROSI.glob("*.sbatch"):
        script = path.read_text(encoding="utf-8")
        if "BASH_SOURCE[0]" in script:
            assert "ALPAKATUNE_ROSI_SCRIPT_DIR" in script, path.name
            assert "/home/th168408/workspace/alpakaTune-ml/hpc/rosi" in script, path.name


def test_experiment_01_is_concrete_and_checksum_pinned():
    experiment = ROOT / "configs/rosi/experiment-01"
    gpu = yaml.safe_load((experiment / "campaign-gpu.yaml").read_text(encoding="utf-8"))
    cpu = yaml.safe_load((experiment / "campaign-cpu.yaml").read_text(encoding="utf-8"))
    assert len(gpu["runs"]) == len(cpu["runs"]) == 6
    assert all("cuda:nvidiaGpu" in run["command"] for run in gpu["runs"])
    assert all("gpuCuda" in run["command"] for run in gpu["runs"])
    assert all("host:cpu" in run["command"] for run in cpu["runs"])
    assert all("cpuOmpBlocks" in run["command"] for run in cpu["runs"])
    assert "CpuSerial" not in (experiment / "campaign-gpu.yaml").read_text()
    assert "CpuSerial" not in (experiment / "campaign-cpu.yaml").read_text()

    manifest = experiment / "campaign-pairs.txt"
    rows = [
        line
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert len(rows) == 3
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    defaults = (ROSI / "experiment-01.env.sh").read_text(encoding="utf-8")
    assert digest in defaults
    assert "/home/th168408/workspace/alpakaTune-ml-runs/experiment-01" in defaults


def test_gpu_jobs_are_exclusive_across_all_gpu_partitions_without_srun():
    all_gpu_partitions = (
        "#SBATCH --partition=gpu-a100,gpu-a100-haicu,gpu-b200-casus,"
        "gpu-b200-haicore,gpu-b200-haicore-mig,gpu-h100,gpu-h100-casus,"
        "gpu-v100"
    )
    for name in (
        "collect-paired-array.sbatch",
        "compare-strategies.sbatch",
        "train-member-array.sbatch",
    ):
        script = (ROSI / name).read_text(encoding="utf-8")
        assert "#SBATCH --gres=gpu:1" in script
        assert "#SBATCH --exclusive" in script
        assert all_gpu_partitions in script
        assert "srun" not in script


def test_training_checks_pytorch_gpu_inside_the_allocation():
    script = (ROSI / "train-member-array.sbatch").read_text(encoding="utf-8")
    assert "torch.cuda.is_available()" in script
    assert "torch.cuda.get_device_name(0)" in script


def test_comparison_uses_core_five_strategy_terminal_interface():
    script = (ROSI / "compare-strategies.sbatch").read_text(encoding="utf-8")
    assert (
        "--strategies exhaustive random simulated_annealing "
        "bayesian_optimization learned_hybrid"
    ) in script
    assert '--model "${MODEL_ARTIFACT}"' in script
    assert "--maximum-executions 40000" in script
    assert "--maximum-retired-configurations 100000" in script
    assert "--tune-until-terminal" in script
    assert "--resume" in script
    assert "--no-plot" in script
    assert "matrixMultiplication" in script
    assert "baseline_examples=(boundaryIter grayScale heatEquation2D nBody vectorAdd)" in script


def test_submit_comparison_supports_held_out_node_and_merge_dependency(tmp_path):
    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    capture = tmp_path / "sbatch-args.txt"
    sbatch = binary_directory / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"${SBATCH_CAPTURE}\"\necho 12345\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    oracle = tmp_path / "test.manifest.json"
    oracle.write_text("{}\n", encoding="utf-8")
    environment = {
        **os.environ,
        "PATH": f"{binary_directory}:{os.environ['PATH']}",
        "SBATCH_CAPTURE": str(capture),
        "ALPAKATUNE_ML_SOURCE": str(ROOT),
        "COMPARE_OUTPUT": str(tmp_path / "comparison"),
        "MODEL_ARTIFACT": str(tmp_path / "future-model.atml"),
        "ORACLE_MANIFEST": str(oracle),
        "COMPARE_NODE": "held-out-node-17",
        "MERGE_JOB_ID": "9988",
    }
    completed = subprocess.run(
        ["bash", str(ROSI / "submit-comparison.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert "--nodelist=held-out-node-17" in arguments
    assert "--dependency=afterok:9988" in arguments
    assert "--partition" not in " ".join(arguments)
    assert "comparison_job=12345" in completed.stdout
