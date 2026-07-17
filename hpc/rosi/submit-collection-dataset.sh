#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE}"
: "${PAIR_MANIFEST:?set PAIR_MANIFEST to the three-row paired manifest}"
: "${DATASET_OUTPUT:?set DATASET_OUTPUT}"

ml_venv="${ALPAKATUNE_ML_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv}"
if [[ ! -x "${ml_venv}/bin/python" ]]; then
    printf 'missing prepared collection environment: %s\n' "${ml_venv}" >&2
    exit 2
fi
PAIR_MANIFEST="$(realpath -- "${PAIR_MANIFEST}")"
DATASET_OUTPUT="$(realpath -m -- "${DATASET_OUTPUT}")"
PAIR_MANIFEST_SHA256="$(sha256sum -- "${PAIR_MANIFEST}")"
PAIR_MANIFEST_SHA256="${PAIR_MANIFEST_SHA256%% *}"
"${ml_venv}/bin/python" "${script_directory}/pair-tools.py" \
    validate-manifest "${PAIR_MANIFEST}"

export ALPAKATUNE_ML_SOURCE PAIR_MANIFEST PAIR_MANIFEST_SHA256 DATASET_OUTPUT
if [[ -n "${ALPAKATUNE_ML_BUILD:-}" ]]; then export ALPAKATUNE_ML_BUILD; fi
if [[ -n "${ALPAKATUNE_ML_VENV:-}" ]]; then export ALPAKATUNE_ML_VENV; fi
if [[ -n "${PAIR_SPLIT_CONFIG:-}" ]]; then export PAIR_SPLIT_CONFIG; fi

collection_job="$({ sbatch --parsable --array=0-2 \
    "${script_directory}/collect-paired-array.sbatch"; } | cut -d';' -f1)"
if [[ ! "${collection_job}" =~ ^[0-9]+$ ]]; then
    printf 'could not parse collection array job id: %s\n' "${collection_job}" >&2
    exit 2
fi
dataset_job="$({ sbatch --parsable --dependency="afterok:${collection_job}" \
    "${script_directory}/prepare-dataset.sbatch"; } | cut -d';' -f1)"
if [[ ! "${dataset_job}" =~ ^[0-9]+$ ]]; then
    printf 'could not parse dataset job id: %s\n' "${dataset_job}" >&2
    exit 2
fi
printf 'collection_array_job=%s dataset_job=%s\n' "${collection_job}" "${dataset_job}"
