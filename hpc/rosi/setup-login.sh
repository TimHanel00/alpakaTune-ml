#!/usr/bin/env bash
# Run this interactively on a rosi login node before submitting jobs.
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/modules.sh"

: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE to the synced alpakaTune-ml checkout}"
ALPAKATUNE_ML_BUILD="${ALPAKATUNE_ML_BUILD:-${ALPAKATUNE_ML_SOURCE}/build}"
export ALPAKATUNE_ML_BUILD

cmake -S "${ALPAKATUNE_ML_SOURCE}" -B "${ALPAKATUNE_ML_BUILD}" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS_RELEASE="-O2 -DNDEBUG" \
    -DALPAKATUNE_ML_ENABLE_CUDA=ON \
    -DALPAKATUNE_ML_BUILD_ALPAKATUNE_TESTS=ON
(cd "${ALPAKATUNE_ML_BUILD}" && make -j)

# These paths are generated from the pinned submodule and are consumed by the
# campaign configurations and Slurm collection jobs.
source "${ALPAKATUNE_ML_BUILD}/generated/alpakatune-paths.sh"

collection_venv="${ALPAKATUNE_COLLECTION_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv}"
training_venv="${ALPAKATUNE_TRAINING_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv-train}"

if [[ ! -x "${collection_venv}/bin/python" ]]; then
    python -m venv "${collection_venv}"
fi
"${collection_venv}/bin/python" -m pip install -e "${ALPAKATUNE_ML_SOURCE}"

if [[ ! -x "${training_venv}/bin/python" ]]; then
    python -m venv "${training_venv}"
fi
# Rosi has no PyTorch module, so the training-only venv supplies it.
"${training_venv}/bin/python" -m pip install -e "${ALPAKATUNE_ML_SOURCE}[train]"

printf 'Prepared ML build: %s\nalpakaTune source: %s\nalpakaTune build: %s\nPath metadata: %s\nCollection venv: %s\nTraining venv: %s\n' \
    "${ALPAKATUNE_ML_BUILD}" \
    "${ALPAKATUNE_SOURCE}" \
    "${ALPAKATUNE_BUILD}" \
    "${ALPAKATUNE_PATH_METADATA_DIR}/alpakatune-paths.json" \
    "${collection_venv}" \
    "${training_venv}"
