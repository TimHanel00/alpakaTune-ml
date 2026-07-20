#!/usr/bin/env bash
#SBATCH --job-name=alpakatune-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -r "${script_directory}/modules.sh" ]]; then
    script_directory="${ALPAKATUNE_ROSI_SCRIPT_DIR:-/home/th168408/workspace/alpakaTune-ml/hpc/rosi}"
fi
source "${script_directory}/modules.sh"

: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE}"
: "${DATASET_ROOT:?set DATASET_ROOT to the immutable prepared dataset}"
: "${TRAINING_CONFIG:?set TRAINING_CONFIG}"
: "${MODEL_OUTPUT:?set MODEL_OUTPUT to an .atml path outside Git}"

ml_venv="${ALPAKATUNE_TRAINING_VENV:-${ALPAKATUNE_ML_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv-train}}"
if [[ ! -x "${ml_venv}/bin/python" ]]; then
    echo "missing prepared training environment: ${ml_venv}" >&2
    echo "run hpc/rosi/setup-login.sh before submitting this job" >&2
    exit 2
fi

"${ml_venv}/bin/python" -m alpakatune_ml train \
    --train "${DATASET_ROOT}/train.manifest.json" \
    --validation "${DATASET_ROOT}/validation.manifest.json" \
    --test "${DATASET_ROOT}/test.manifest.json" \
    --config "${TRAINING_CONFIG}" \
    --output "${MODEL_OUTPUT}"
