#!/usr/bin/env bash
#SBATCH --job-name=alpakatune-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --partition=gpu-a100,gpu-a100-haicu,gpu-b200-casus,gpu-b200-haicore,gpu-b200-haicore-mig,gpu-h100,gpu-h100-casus,gpu-v100
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/modules.sh"

: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE}"
: "${COMPARE_OUTPUT:?set COMPARE_OUTPUT to shared storage outside Git}"
: "${MODEL_ARTIFACT:?set MODEL_ARTIFACT to the merged .atml model}"
: "${ORACLE_MANIFEST:?set ORACLE_MANIFEST to the matching dataset split manifest}"

ml_venv="${ALPAKATUNE_ML_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv}"
if [[ ! -x "${ml_venv}/bin/python" ]]; then
    printf 'missing prepared collection environment: %s\n' "${ml_venv}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_ARTIFACT}" || ! -f "${ORACLE_MANIFEST}" ]]; then
    printf 'MODEL_ARTIFACT and ORACLE_MANIFEST must be readable files\n' >&2
    exit 2
fi

alpakatune_source="${ALPAKATUNE_ML_SOURCE}/alpaka3-tuner"
build_root="${ALPAKATUNE_ML_BUILD:-${ALPAKATUNE_ML_SOURCE}/build}"
alpakatune_build="${build_root}/alpaka3-tuner"
baseline_build="${build_root}/alpaka-baseline"
base_config="${alpakatune_source}/config/alpakaTune.yaml"
benchmark_runner="${alpakatune_source}/benchmarks/run.py"
baseline_runner="${alpakatune_source}/benchmarks/run_baseline.py"
visualizer="${alpakatune_source}/benchmarks/visualize.py"

read -r -a examples <<< "${COMPARE_EXAMPLES:-boundaryIter grayScale heatEquation2D matrixMultiplication nBody vectorAdd}"
# Upstream Alpaka has no matrixMultiplication example. It remains in every
# tuned strategy comparison, but there is no artificial untuned baseline for it.
baseline_examples=(boundaryIter grayScale heatEquation2D nBody vectorAdd)
baseline_output="${COMPARE_OUTPUT}/baseline"
gpu_output="${COMPARE_OUTPUT}/gpu"
cpu_output="${COMPARE_OUTPUT}/cpu"

"${ml_venv}/bin/python" "${script_directory}/pair-tools.py" record-allocation \
    --output "${COMPARE_OUTPUT}" --kind strategy_comparison

"${ml_venv}/bin/python" "${baseline_runner}" \
    --build-dir "${baseline_build}" \
    --output "${baseline_output}" \
    --examples "${baseline_examples[@]}" \
    --repetitions 10 \
    --resume

for backend_kind in gpu cpu; do
    if [[ "${backend_kind}" == gpu ]]; then
        backend="cuda:nvidiaGpu"
        executor="gpuCuda"
        output="${gpu_output}"
    else
        backend="host:cpu"
        executor="cpuOmpBlocks"
        output="${cpu_output}"
    fi
    "${ml_venv}/bin/python" "${benchmark_runner}" \
        --build-dir "${alpakatune_build}" \
        --config "${base_config}" \
        --output "${output}" \
        --examples "${examples[@]}" \
        --strategies exhaustive random simulated_annealing bayesian_optimization learned_hybrid \
        --model "${MODEL_ARTIFACT}" \
        --backend "${backend}" \
        --executor "${executor}" \
        --maximum-executions 40000 \
        --maximum-retired-configurations 100000 \
        --tune-until-terminal \
        --resume \
        --no-plot
    "${ml_venv}/bin/python" "${visualizer}" "${output}" \
        --output "${output}/plots" \
        --baseline "${baseline_output}"
    "${ml_venv}/bin/python" -m alpakatune_ml evaluate-search "${output}" \
        --oracle "${ORACLE_MANIFEST}" \
        --output "${output}/search-evaluation.json"
done
