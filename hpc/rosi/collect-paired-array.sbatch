#!/usr/bin/env bash
#SBATCH --job-name=alpakatune-paired-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --partition=gpu-a100,gpu-a100-haicu,gpu-b200-casus,gpu-b200-haicore,gpu-b200-haicore-mig,gpu-h100,gpu-h100-casus,gpu-v100
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/modules.sh"

: "${SLURM_ARRAY_TASK_ID:?submit this script as the exact 0-2 array}"
: "${SLURM_ARRAY_TASK_COUNT:?SLURM did not provide the array task count}"
: "${SLURM_ARRAY_TASK_MIN:?SLURM did not provide the array minimum}"
: "${SLURM_ARRAY_TASK_MAX:?SLURM did not provide the array maximum}"
: "${PAIR_MANIFEST:?set PAIR_MANIFEST to the three-row paired campaign manifest}"
: "${PAIR_MANIFEST_SHA256:?set PAIR_MANIFEST_SHA256 to its checksum}"
: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE to the synced checkout}"

if [[ "${PAIR_MANIFEST}" != /* || ! -r "${PAIR_MANIFEST}" ]]; then
    printf 'PAIR_MANIFEST must be an absolute readable file: %s\n' "${PAIR_MANIFEST}" >&2
    exit 2
fi
if [[ ! "${PAIR_MANIFEST_SHA256}" =~ ^[[:xdigit:]]{64}$ ]]; then
    printf 'PAIR_MANIFEST_SHA256 must contain 64 hexadecimal characters\n' >&2
    exit 2
fi
observed_sha256="$(sha256sum -- "${PAIR_MANIFEST}")"
observed_sha256="${observed_sha256%% *}"
if [[ "${observed_sha256,,}" != "${PAIR_MANIFEST_SHA256,,}" ]]; then
    printf 'pair manifest checksum mismatch: expected %s, observed %s\n' \
        "${PAIR_MANIFEST_SHA256}" "${observed_sha256}" >&2
    exit 2
fi
for value in "${SLURM_ARRAY_TASK_ID}" "${SLURM_ARRAY_TASK_COUNT}" \
    "${SLURM_ARRAY_TASK_MIN}" "${SLURM_ARRAY_TASK_MAX}"; do
    if [[ ! "${value}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        printf 'invalid Slurm array value: %s\n' "${value}" >&2
        exit 2
    fi
done
task_index="$((10#${SLURM_ARRAY_TASK_ID}))"
if (( 10#${SLURM_ARRAY_TASK_COUNT} != 3 || 10#${SLURM_ARRAY_TASK_MIN} != 0 \
      || 10#${SLURM_ARRAY_TASK_MAX} != 2 || task_index > 2 )); then
    printf 'paired collection array must be exactly tasks 0-2\n' >&2
    exit 2
fi

ml_venv="${ALPAKATUNE_ML_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv}"
if [[ ! -x "${ml_venv}/bin/python" ]]; then
    printf 'missing prepared collection environment: %s\n' "${ml_venv}" >&2
    exit 2
fi
pair_tools="${script_directory}/pair-tools.py"
"${ml_venv}/bin/python" "${pair_tools}" validate-manifest "${PAIR_MANIFEST}"

mapfile -t pair_entries < <(awk 'NF && $1 !~ /^#/ {print}' "${PAIR_MANIFEST}")
read -r gpu_config cpu_config pair_output extra <<< "${pair_entries[task_index]}"
if [[ -n "${extra:-}" ]]; then
    printf 'invalid paired manifest entry %d\n' "${task_index}" >&2
    exit 2
fi
gpu_config="$(realpath -- "${gpu_config}")"
cpu_config="$(realpath -- "${cpu_config}")"
pair_output="$(realpath -m -- "${pair_output}")"

export ALPAKATUNE_SOURCE="${ALPAKATUNE_ML_SOURCE}/alpaka3-tuner"
export ALPAKATUNE_BUILD="${ALPAKATUNE_ML_BUILD:-${ALPAKATUNE_ML_SOURCE}/build}/alpaka3-tuner"

"${ml_venv}/bin/python" "${pair_tools}" record-allocation \
    --output "${pair_output}" \
    --pair-index "${task_index}" \
    --kind collection_pair

printf 'pair_task=%d/3 node=%s manifest_sha256=%s output=%s\n' \
    "${task_index}" "${SLURMD_NODENAME:-unknown}" "${observed_sha256}" "${pair_output}"

# Both campaigns execute serially in this allocation. --resume validates and
# skips completed histories while the allocation metadata prevents cross-node mixing.
"${ml_venv}/bin/python" -m alpakatune_ml collect \
    "${gpu_config}" --output "${pair_output}/gpu" --resume
"${ml_venv}/bin/python" -m alpakatune_ml collect \
    "${cpu_config}" --output "${pair_output}/cpu" --resume
