#!/usr/bin/env bash
#SBATCH --job-name=alpakatune-train-member
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

: "${SLURM_ARRAY_TASK_ID:?submit this script with sbatch --array=0-2}"
: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE}"
: "${DATASET_ROOT:?set DATASET_ROOT to the immutable prepared dataset}"
: "${TRAINING_CONFIG:?set TRAINING_CONFIG}"
: "${MODEL_MEMBER_DIR:?set MODEL_MEMBER_DIR to shared storage outside Git}"

if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "member index must be a non-negative integer: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

ml_venv="${ALPAKATUNE_TRAINING_VENV:-${ALPAKATUNE_ML_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv-train}}"
if [[ ! -x "${ml_venv}/bin/python" ]]; then
    echo "missing prepared training environment: ${ml_venv}" >&2
    echo "run hpc/rosi/setup-login.sh before submitting this array" >&2
    exit 2
fi

ensemble_size="$("${ml_venv}/bin/python" -c \
    'import sys; from pathlib import Path; from alpakatune_ml.training import load_training_config; print(load_training_config(Path(sys.argv[1]))["ensemble_size"])' \
    "${TRAINING_CONFIG}")"
array_min="${SLURM_ARRAY_TASK_MIN:-${SLURM_ARRAY_TASK_ID}}"
array_max="${SLURM_ARRAY_TASK_MAX:-${SLURM_ARRAY_TASK_ID}}"
array_count="${SLURM_ARRAY_TASK_COUNT:-1}"
if (( array_min != 0 || array_max != ensemble_size - 1 || array_count != ensemble_size )); then
    echo "training array must contain exactly 0-$((ensemble_size - 1)) for this config" >&2
    exit 2
fi

mkdir -p "${MODEL_MEMBER_DIR}"
member_output="${MODEL_MEMBER_DIR}/member-${SLURM_ARRAY_TASK_ID}.atml"
"${ml_venv}/bin/python" -m alpakatune_ml train-member \
    --train "${DATASET_ROOT}/train.manifest.json" \
    --validation "${DATASET_ROOT}/validation.manifest.json" \
    --test "${DATASET_ROOT}/test.manifest.json" \
    --config "${TRAINING_CONFIG}" \
    --member-index "${SLURM_ARRAY_TASK_ID}" \
    --output "${member_output}"
