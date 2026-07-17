#!/usr/bin/env bash

set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE}"
: "${DATASET_ROOT:?set DATASET_ROOT to the immutable prepared dataset}"
: "${TRAINING_CONFIG:?set TRAINING_CONFIG}"
: "${MODEL_MEMBER_DIR:?set MODEL_MEMBER_DIR to shared storage outside Git}"
: "${MODEL_OUTPUT:?set MODEL_OUTPUT to the final .atml path outside Git}"

ml_venv="${ALPAKATUNE_TRAINING_VENV:-${ALPAKATUNE_ML_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv-train}}"
if [[ ! -x "${ml_venv}/bin/python" ]]; then
    echo "missing prepared training environment: ${ml_venv}" >&2
    exit 2
fi
ensemble_size="$("${ml_venv}/bin/python" -c \
    'import sys; from pathlib import Path; from alpakatune_ml.training import load_training_config; print(load_training_config(Path(sys.argv[1]))["ensemble_size"])' \
    "${TRAINING_CONFIG}")"

export ALPAKATUNE_ML_SOURCE DATASET_ROOT TRAINING_CONFIG MODEL_MEMBER_DIR MODEL_OUTPUT
if [[ -n "${ALPAKATUNE_TRAINING_VENV:-}" ]]; then
    export ALPAKATUNE_TRAINING_VENV
fi
if [[ -n "${ALPAKATUNE_ML_VENV:-}" ]]; then
    export ALPAKATUNE_ML_VENV
fi

member_job="$({ sbatch --parsable --array="0-$((ensemble_size - 1))" \
    "${script_directory}/train-member-array.sbatch"; } | cut -d';' -f1)"
if [[ ! "${member_job}" =~ ^[0-9]+$ ]]; then
    echo "could not parse member-array job id: ${member_job}" >&2
    exit 2
fi
merge_job="$({ sbatch --parsable --dependency="afterok:${member_job}" \
    "${script_directory}/merge-members.sbatch"; } | cut -d';' -f1)"
if [[ ! "${merge_job}" =~ ^[0-9]+$ ]]; then
    echo "could not parse merge job id: ${merge_job}" >&2
    exit 2
fi
echo "member_array_job=${member_job} merge_job=${merge_job}"
