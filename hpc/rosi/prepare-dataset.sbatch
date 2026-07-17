#!/usr/bin/env bash
#SBATCH --job-name=alpakatune-dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/modules.sh"

: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE}"
: "${HISTORY_ROOTS:?set HISTORY_ROOTS to a colon-separated list of campaign roots}"
: "${SPLIT_CONFIG:?set SPLIT_CONFIG to the whole-device split YAML}"
: "${DATASET_OUTPUT:?set DATASET_OUTPUT to bulk storage outside Git}"

ml_venv="${ALPAKATUNE_ML_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv}"
if [[ ! -x "${ml_venv}/bin/python" ]]; then
    printf 'missing prepared collection environment: %s\n' "${ml_venv}" >&2
    printf 'run hpc/rosi/setup-login.sh before submitting this job\n' >&2
    exit 2
fi

IFS=: read -r -a history_roots <<< "${HISTORY_ROOTS}"
"${ml_venv}/bin/python" -m alpakatune_ml build-dataset \
    "${history_roots[@]}" \
    --splits "${SPLIT_CONFIG}" \
    --output "${DATASET_OUTPUT}"

"${ml_venv}/bin/python" -m alpakatune_ml validate-splits \
    --train "${DATASET_OUTPUT}/train.manifest.json" \
    --validation "${DATASET_OUTPUT}/validation.manifest.json" \
    --test "${DATASET_OUTPUT}/test.manifest.json"
