#!/usr/bin/env bash
#SBATCH --job-name=alpakatune-collect-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/modules.sh"

: "${SLURM_ARRAY_TASK_ID:?submit this script with sbatch --array=...}"
: "${SLURM_ARRAY_TASK_COUNT:?SLURM did not provide the array task count}"
: "${SLURM_ARRAY_TASK_MIN:?SLURM did not provide the array minimum}"
: "${SLURM_ARRAY_TASK_MAX:?SLURM did not provide the array maximum}"
: "${CAMPAIGN_ARRAY_FILE:?set CAMPAIGN_ARRAY_FILE to the absolute campaign list}"
: "${CAMPAIGN_ARRAY_SHA256:?set CAMPAIGN_ARRAY_SHA256 to the campaign list checksum}"
: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE to the synced alpakaTune-ml checkout}"
: "${ALPAKATUNE_SOURCE:?set ALPAKATUNE_SOURCE to the pinned alpakaTune checkout}"
: "${ALPAKATUNE_BUILD:?set ALPAKATUNE_BUILD to its prepared build directory}"

if [[ "${CAMPAIGN_ARRAY_FILE}" != /* || ! -f "${CAMPAIGN_ARRAY_FILE}" || ! -r "${CAMPAIGN_ARRAY_FILE}" ]]; then
    printf 'campaign array file must be an absolute, readable regular file: %s\n' "${CAMPAIGN_ARRAY_FILE}" >&2
    exit 2
fi
if [[ ! "${CAMPAIGN_ARRAY_SHA256}" =~ ^[[:xdigit:]]{64}$ ]]; then
    printf 'CAMPAIGN_ARRAY_SHA256 must contain exactly 64 hexadecimal characters\n' >&2
    exit 2
fi
manifest_sha256="$(sha256sum -- "${CAMPAIGN_ARRAY_FILE}")"
manifest_sha256="${manifest_sha256%% *}"
if [[ "${manifest_sha256,,}" != "${CAMPAIGN_ARRAY_SHA256,,}" ]]; then
    printf 'campaign array checksum mismatch: expected %s, observed %s\n' \
        "${CAMPAIGN_ARRAY_SHA256}" "${manifest_sha256}" >&2
    exit 2
fi

mapfile -t campaign_entries < <(
    awk 'NF && $1 !~ /^#/ {print}' "${CAMPAIGN_ARRAY_FILE}"
)
entry_count="${#campaign_entries[@]}"
if (( entry_count == 0 )); then
    printf 'campaign array file contains no entries: %s\n' "${CAMPAIGN_ARRAY_FILE}" >&2
    exit 2
fi

for value in \
    "${SLURM_ARRAY_TASK_ID}" \
    "${SLURM_ARRAY_TASK_COUNT}" \
    "${SLURM_ARRAY_TASK_MIN}" \
    "${SLURM_ARRAY_TASK_MAX}"; do
    if [[ ! "${value}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        printf 'invalid non-negative integer in Slurm array metadata: %s\n' "${value}" >&2
        exit 2
    fi
done

task_index="$((10#${SLURM_ARRAY_TASK_ID}))"
array_task_count="$((10#${SLURM_ARRAY_TASK_COUNT}))"
array_task_min="$((10#${SLURM_ARRAY_TASK_MIN}))"
array_task_max="$((10#${SLURM_ARRAY_TASK_MAX}))"
if (( array_task_count != entry_count || array_task_min != 0 || array_task_max != entry_count - 1 )); then
    printf 'array shape must be exactly 0-%d (%d tasks); received min=%d max=%d count=%d\n' \
        "$((entry_count - 1))" "${entry_count}" \
        "${array_task_min}" "${array_task_max}" "${array_task_count}" >&2
    exit 2
fi
if (( task_index >= entry_count )); then
    printf 'array index %d is outside %d campaign entries\n' "${task_index}" "${entry_count}" >&2
    exit 2
fi

declare -a campaign_configs=()
declare -a campaign_outputs=()
for entry_index in "${!campaign_entries[@]}"; do
    read -r campaign_config campaign_output extra <<< "${campaign_entries[entry_index]}"
    if [[ -z "${campaign_config}" || -z "${campaign_output}" || -n "${extra:-}" ]]; then
        printf 'campaign-array entry %d must contain exactly CONFIG_PATH OUTPUT_PATH without whitespace in paths\n' \
            "$((entry_index + 1))" >&2
        exit 2
    fi
    if [[ "${campaign_config}" != /* || ! -f "${campaign_config}" || ! -r "${campaign_config}" ]]; then
        printf 'campaign config on entry %d must be an absolute, readable file: %s\n' \
            "$((entry_index + 1))" "${campaign_config}" >&2
        exit 2
    fi
    if [[ "${campaign_output}" != /* || "${campaign_output}" == / ]]; then
        printf 'campaign output on entry %d must be an absolute path other than /: %s\n' \
            "$((entry_index + 1))" "${campaign_output}" >&2
        exit 2
    fi

    campaign_config="$(realpath -- "${campaign_config}")"
    campaign_output="$(realpath -m -- "${campaign_output}")"
    for previous_output in "${campaign_outputs[@]}"; do
        if [[ "${campaign_output}" == "${previous_output}" \
              || "${campaign_output}" == "${previous_output}"/* \
              || "${previous_output}" == "${campaign_output}"/* ]]; then
            printf 'campaign outputs must be distinct and non-overlapping: %s and %s\n' \
                "${previous_output}" "${campaign_output}" >&2
            exit 2
        fi
    done
    campaign_configs+=("${campaign_config}")
    campaign_outputs+=("${campaign_output}")
done

campaign_config="${campaign_configs[task_index]}"
campaign_output="${campaign_outputs[task_index]}"

ml_venv="${ALPAKATUNE_ML_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv}"
if [[ ! -x "${ml_venv}/bin/python" ]]; then
    printf 'missing prepared collection environment: %s\n' "${ml_venv}" >&2
    printf 'run hpc/rosi/setup-login.sh before submitting this array\n' >&2
    exit 2
fi

printf 'array_task=%s/%s node=%s manifest_sha256=%s config=%s output=%s\n' \
    "${SLURM_ARRAY_TASK_ID}" "${SLURM_ARRAY_TASK_COUNT}" \
    "${SLURMD_NODENAME:-unknown}" "${manifest_sha256}" \
    "${campaign_config}" "${campaign_output}"
"${ml_venv}/bin/python" -m alpakatune_ml collect \
    "${campaign_config}" \
    --output "${campaign_output}" \
    --resume
