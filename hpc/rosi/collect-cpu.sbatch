#!/usr/bin/env bash
#SBATCH --job-name=alpakatune-collect-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/modules.sh"

: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE to the synced alpakaTune-ml checkout}"
: "${ALPAKATUNE_SOURCE:?set ALPAKATUNE_SOURCE to the pinned alpakaTune checkout}"
: "${ALPAKATUNE_BUILD:?set ALPAKATUNE_BUILD to its prepared build directory}"
: "${CAMPAIGN_CONFIG:?set CAMPAIGN_CONFIG to a committed CPU campaign YAML}"
: "${CAMPAIGN_OUTPUT:?set CAMPAIGN_OUTPUT to bulk storage outside Git}"

ml_venv="${ALPAKATUNE_ML_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv}"
if [[ ! -x "${ml_venv}/bin/python" ]]; then
    echo "missing prepared collection environment: ${ml_venv}" >&2
    echo "run hpc/rosi/setup-login.sh before submitting this job" >&2
    exit 2
fi

"${ml_venv}/bin/python" -m alpakatune_ml collect \
    "${CAMPAIGN_CONFIG}" \
    --output "${CAMPAIGN_OUTPUT}" \
    --resume
