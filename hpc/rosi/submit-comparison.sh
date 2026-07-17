#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE}"
: "${COMPARE_OUTPUT:?set COMPARE_OUTPUT}"
: "${MODEL_ARTIFACT:?set MODEL_ARTIFACT}"
: "${ORACLE_MANIFEST:?set ORACLE_MANIFEST}"
: "${COMPARE_NODE:?set COMPARE_NODE to the held-out task-2 allocation node}"

COMPARE_OUTPUT="$(realpath -m -- "${COMPARE_OUTPUT}")"
MODEL_ARTIFACT="$(realpath -m -- "${MODEL_ARTIFACT}")"
ORACLE_MANIFEST="$(realpath -- "${ORACLE_MANIFEST}")"
if [[ ! "${COMPARE_NODE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'COMPARE_NODE contains invalid characters: %s\n' "${COMPARE_NODE}" >&2
    exit 2
fi
export ALPAKATUNE_ML_SOURCE COMPARE_OUTPUT MODEL_ARTIFACT ORACLE_MANIFEST COMPARE_NODE
if [[ -n "${ALPAKATUNE_ML_BUILD:-}" ]]; then export ALPAKATUNE_ML_BUILD; fi
if [[ -n "${ALPAKATUNE_ML_VENV:-}" ]]; then export ALPAKATUNE_ML_VENV; fi
if [[ -n "${COMPARE_EXAMPLES:-}" ]]; then export COMPARE_EXAMPLES; fi

sbatch_arguments=(--parsable "--nodelist=${COMPARE_NODE}")
if [[ -n "${MERGE_JOB_ID:-}" ]]; then
    if [[ ! "${MERGE_JOB_ID}" =~ ^[0-9]+$ ]]; then
        printf 'MERGE_JOB_ID must be a numeric Slurm job ID: %s\n' "${MERGE_JOB_ID}" >&2
        exit 2
    fi
    sbatch_arguments+=("--dependency=afterok:${MERGE_JOB_ID}")
fi
job_id="$({ sbatch "${sbatch_arguments[@]}" \
    "${script_directory}/compare-strategies.sbatch"; } | cut -d';' -f1)"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    printf 'could not parse comparison job id: %s\n' "${job_id}" >&2
    exit 2
fi
printf 'comparison_job=%s\n' "${job_id}"
