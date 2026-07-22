#!/usr/bin/env bash
#SBATCH --job-name=alpakatune-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --partition=gpu-a100,gpu-a100-haicu,gpu-b200-casus,gpu-b200-haicore,gpu-b200-haicore-mig,gpu-h100,gpu-h100-casus,gpu-v100
#SBATCH --time=24:00:00
#SBATCH --output=/home/th168408/workspace/alpakaTune-ml-runs/experiment-02/slurm-%x-%j.out
#SBATCH --error=/home/th168408/workspace/alpakaTune-ml-runs/experiment-02/slurm-%x-%j.err

set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -r "${script_directory}/modules.sh" ]]; then
    script_directory="${ALPAKATUNE_ROSI_SCRIPT_DIR:-/home/th168408/workspace/alpakaTune-ml/hpc/rosi}"
fi
source "${script_directory}/modules.sh"

: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE}"
validation_output="${BOUNDED_POOL_OUTPUT:-/home/th168408/workspace/alpakaTune-ml-runs/experiment-02/evaluations/bounded-pool}"
model_artifact="${BOUNDED_POOL_MODEL:-/home/th168408/workspace/alpakaTune-ml-runs/experiment-01/artifacts/rosi/experiment-01/model.atml}"
pool_size="${BOUNDED_POOL_SIZE:-64}"
batch_size="${BOUNDED_POOL_BATCH_SIZE:-16}"

ml_python="${ALPAKATUNE_ML_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv}/bin/python"
alpakatune_source="${ALPAKATUNE_ML_SOURCE}/alpaka3-tuner"
build_root="${ALPAKATUNE_ML_BUILD:-${ALPAKATUNE_ML_SOURCE}/build}"
alpakatune_build="${build_root}/alpaka3-tuner"
baseline_build="${build_root}/alpaka-baseline-gpu"
base_config="${alpakatune_source}/config/alpakaTune.yaml"

if [[ ! -x "${ml_python}" || ! -f "${model_artifact}" ]]; then
    printf 'missing Python environment or experiment-01 model artifact\n' >&2
    exit 2
fi
mkdir -p "${validation_output}"

"${ml_python}" "${script_directory}/pair-tools.py" record-allocation \
    --output "${validation_output}" --kind bounded_pool_validation

"${ml_python}" "${alpakatune_source}/benchmarks/run_baseline.py" \
    --build-dir "${baseline_build}" \
    --output "${validation_output}/baseline" \
    --examples vectorAdd \
    --repetitions 10 \
    --resume

"${ml_python}" "${alpakatune_source}/benchmarks/run.py" \
    --build-dir "${alpakatune_build}" \
    --config "${base_config}" \
    --output "${validation_output}/gpu" \
    --examples matrixMultiplication vectorAdd \
    --strategies learned_hybrid exhaustive random simulated_annealing bayesian_optimization \
    --model "${model_artifact}" \
    --learned-candidate-pool-size "${pool_size}" \
    --learned-candidate-batch-size "${batch_size}" \
    --backend cuda:nvidiaGpu \
    --executor gpuCuda \
    --maximum-executions 4000 \
    --maximum-retired-configurations 100000 \
    --resume \
    --no-plot

"${ml_python}" "${script_directory}/validate-bounded-pool.py" \
    "${validation_output}/gpu" \
    --pool-size "${pool_size}" \
    --batch-size "${batch_size}"

"${ml_python}" "${alpakatune_source}/benchmarks/visualize.py" \
    "${validation_output}/gpu" \
    --output "${validation_output}/gpu/plots" \
    --baseline "${validation_output}/baseline"
