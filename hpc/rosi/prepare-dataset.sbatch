#!/usr/bin/env bash
#SBATCH --job-name=alpakatune-dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -r "${script_directory}/modules.sh" ]]; then
    script_directory="${ALPAKATUNE_ROSI_SCRIPT_DIR:-/home/th168408/workspace/alpakaTune-ml/hpc/rosi}"
fi
source "${script_directory}/modules.sh"

: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE}"
: "${PAIR_MANIFEST:?set PAIR_MANIFEST to the exact three-row collection manifest}"
: "${PAIR_MANIFEST_SHA256:?set PAIR_MANIFEST_SHA256}"
: "${DATASET_OUTPUT:?set DATASET_OUTPUT to bulk storage outside Git}"

observed_sha256="$(sha256sum -- "${PAIR_MANIFEST}")"
observed_sha256="${observed_sha256%% *}"
if [[ "${observed_sha256,,}" != "${PAIR_MANIFEST_SHA256,,}" ]]; then
    printf 'pair manifest checksum mismatch: expected %s, observed %s\n' \
        "${PAIR_MANIFEST_SHA256}" "${observed_sha256}" >&2
    exit 2
fi

ml_venv="${ALPAKATUNE_ML_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv}"
if [[ ! -x "${ml_venv}/bin/python" ]]; then
    printf 'missing prepared collection environment: %s\n' "${ml_venv}" >&2
    exit 2
fi
pair_tools="${script_directory}/pair-tools.py"
split_config="${PAIR_SPLIT_CONFIG:-${DATASET_OUTPUT}.splits.yaml}"

if [[ -f "${DATASET_OUTPUT}/dataset.manifest.json" ]]; then
    "${ml_venv}/bin/python" -m alpakatune_ml validate-splits \
        --train "${DATASET_OUTPUT}/train.manifest.json" \
        --validation "${DATASET_OUTPUT}/validation.manifest.json" \
        --test "${DATASET_OUTPUT}/test.manifest.json"
    printf 'SKIP completed immutable dataset: %s\n' "${DATASET_OUTPUT}"
    exit 0
fi

"${ml_venv}/bin/python" "${pair_tools}" prepare-splits \
    "${PAIR_MANIFEST}" --output "${split_config}" \
    --policy "${DATASET_SPLIT_POLICY:-whole_device}"
mapfile -t history_roots < <(
    "${ml_venv}/bin/python" "${pair_tools}" roots "${PAIR_MANIFEST}"
)
if (( ${#history_roots[@]} != 6 )); then
    printf 'expected six CPU/GPU campaign roots, found %d\n' "${#history_roots[@]}" >&2
    exit 2
fi

"${ml_venv}/bin/python" -m alpakatune_ml validate-histories "${history_roots[@]}"
"${ml_venv}/bin/python" -m alpakatune_ml build-dataset \
    "${history_roots[@]}" \
    --splits "${split_config}" \
    --output "${DATASET_OUTPUT}"
"${ml_venv}/bin/python" -m alpakatune_ml validate-splits \
    --train "${DATASET_OUTPUT}/train.manifest.json" \
    --validation "${DATASET_OUTPUT}/validation.manifest.json" \
    --test "${DATASET_OUTPUT}/test.manifest.json"
cp -- "${split_config}" "${DATASET_OUTPUT}/splits.yaml"
cp -- "${split_config}.pairs.json" "${DATASET_OUTPUT}/pairs.json"
