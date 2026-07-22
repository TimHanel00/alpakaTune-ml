#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE}"

bounded_pool_output="${BOUNDED_POOL_OUTPUT:-/home/th168408/workspace/alpakaTune-ml-runs/experiment-02/evaluations/bounded-pool}"
mkdir -p "${bounded_pool_output}" \
    /home/th168408/workspace/alpakaTune-ml-runs/experiment-02
export ALPAKATUNE_ML_SOURCE BOUNDED_POOL_OUTPUT="${bounded_pool_output}"
if [[ -n "${ALPAKATUNE_ML_BUILD:-}" ]]; then export ALPAKATUNE_ML_BUILD; fi
if [[ -n "${ALPAKATUNE_ML_VENV:-}" ]]; then export ALPAKATUNE_ML_VENV; fi
if [[ -n "${BOUNDED_POOL_MODEL:-}" ]]; then export BOUNDED_POOL_MODEL; fi
if [[ -n "${BOUNDED_POOL_SIZE:-}" ]]; then export BOUNDED_POOL_SIZE; fi
if [[ -n "${BOUNDED_POOL_BATCH_SIZE:-}" ]]; then export BOUNDED_POOL_BATCH_SIZE; fi

job_id="$({ sbatch --parsable "${script_directory}/validate-bounded-pool.sbatch"; } | cut -d';' -f1)"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    printf 'could not parse bounded-pool job id: %s\n' "${job_id}" >&2
    exit 2
fi
printf 'bounded_pool_job=%s\n' "${job_id}"
