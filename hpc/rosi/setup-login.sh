#!/usr/bin/env bash
# Run this interactively on a rosi login node before submitting jobs.
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/modules.sh"

: "${ALPAKATUNE_ML_SOURCE:?set ALPAKATUNE_ML_SOURCE to the synced alpakaTune-ml checkout}"
ALPAKATUNE_ML_BUILD="${ALPAKATUNE_ML_BUILD:-${ALPAKATUNE_ML_SOURCE}/build}"
ALPAKATUNE_SOURCE="${ALPAKATUNE_ML_SOURCE}/alpaka3-tuner"
ALPAKATUNE_BUILD="${ALPAKATUNE_ML_BUILD}/alpaka3-tuner"
ALPAKA_BASELINE_BUILD="${ALPAKATUNE_ML_BUILD}/alpaka-baseline"
ALPAKATUNE_DEPENDENCY_ROOT="${ALPAKATUNE_DEPENDENCY_ROOT:-${ALPAKATUNE_ML_SOURCE}/dependencies}"
alpaka_source="${ALPAKATUNE_DEPENDENCY_ROOT}/alpaka3"
yaml_cpp_source="${ALPAKATUNE_DEPENDENCY_ROOT}/yaml_cpp"
nlohmann_json_source="${ALPAKATUNE_DEPENDENCY_ROOT}/nlohmann_json"
export ALPAKATUNE_ML_BUILD ALPAKATUNE_SOURCE ALPAKATUNE_BUILD \
    ALPAKA_BASELINE_BUILD ALPAKATUNE_DEPENDENCY_ROOT

for dependency_source in "${alpaka_source}" "${yaml_cpp_source}" "${nlohmann_json_source}"; do
    if [[ ! -f "${dependency_source}/CMakeLists.txt" ]]; then
        printf 'pre-RSYNCed dependency source is missing: %s\n' "${dependency_source}" >&2
        exit 2
    fi
done

# RSYNC deliberately excludes .git metadata. Configure the synced runtime tree
# directly; the top-level ML CMake project enforces a gitlink and is local-only.
cmake -S "${ALPAKATUNE_SOURCE}" -B "${ALPAKATUNE_BUILD}" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS_RELEASE="-O2 -DNDEBUG" \
    -Dalpaka_DEP_CUDA=ON \
    -Dalpaka_DEP_OMP=ON \
    -Dalpaka_EXEC_CpuSerial=OFF \
    -Dalpaka_EXEC_CpuOmpBlocks=ON \
    -Dalpaka_EXEC_GpuCuda=ON \
    -DalpakaTune_BUILD_EXAMPLES=ON \
    -DalpakaTune_BUILD_TESTING=OFF \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_ALPAKA3="${alpaka_source}" \
    -DFETCHCONTENT_SOURCE_DIR_YAML_CPP="${yaml_cpp_source}" \
    -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="${nlohmann_json_source}"
(cd "${ALPAKATUNE_BUILD}" && make -j)

# The comparison job needs an instrumentation-free build of the same pre-synced
# Alpaka source. Fully disconnected mode forbids CMake from invoking Git.
cmake -S "${alpaka_source}" -B "${ALPAKA_BASELINE_BUILD}" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS_RELEASE="-O2 -DNDEBUG" \
    -Dalpaka_EXAMPLES=ON \
    -Dalpaka_DEP_CUDA=ON \
    -Dalpaka_DEP_OMP=ON \
    -Dalpaka_EXEC_CpuSerial=OFF \
    -Dalpaka_EXEC_CpuOmpBlocks=ON \
    -Dalpaka_EXEC_GpuCuda=ON \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON
(cd "${ALPAKA_BASELINE_BUILD}" && make -j)

collection_venv="${ALPAKATUNE_COLLECTION_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv}"
training_venv="${ALPAKATUNE_TRAINING_VENV:-${ALPAKATUNE_ML_SOURCE}/.venv-train}"
torch_version="${ALPAKATUNE_TORCH_VERSION:-2.11.0+cu128}"
torch_index_url="${ALPAKATUNE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
export TMPDIR="${ALPAKATUNE_TMPDIR:-${ALPAKATUNE_ML_SOURCE}/.tmp}"
export PIP_CACHE_DIR="${ALPAKATUNE_PIP_CACHE_DIR:-${ALPAKATUNE_ML_SOURCE}/.pip-cache}"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

if [[ ! -x "${collection_venv}/bin/python" ]]; then
    python -m venv "${collection_venv}"
fi
"${collection_venv}/bin/python" -m pip install -e "${ALPAKATUNE_ML_SOURCE}[plot]"

if [[ ! -x "${training_venv}/bin/python" ]]; then
    python -m venv "${training_venv}"
fi
# Rosi has no PyTorch module. Pin the CUDA 12.8 wheel to the loaded module and
# cluster driver instead of allowing pip to select a newer CUDA wheel.
"${training_venv}/bin/python" -m pip install -e "${ALPAKATUNE_ML_SOURCE}"
"${training_venv}/bin/python" -m pip install \
    "torch==${torch_version}" \
    --index-url "${torch_index_url}"
"${training_venv}/bin/python" -c \
    'import torch; print(f"PyTorch {torch.__version__} import verified")'

printf 'Prepared ML build root: %s\nDependency root: %s\nalpakaTune source: %s\nalpakaTune build: %s\nAlpaka baseline build: %s\nCollection venv: %s\nTraining venv: %s\n' \
    "${ALPAKATUNE_ML_BUILD}" \
    "${ALPAKATUNE_DEPENDENCY_ROOT}" \
    "${ALPAKATUNE_SOURCE}" \
    "${ALPAKATUNE_BUILD}" \
    "${ALPAKA_BASELINE_BUILD}" \
    "${collection_venv}" \
    "${training_venv}"
