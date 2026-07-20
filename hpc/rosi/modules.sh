#!/usr/bin/env bash
# Module stack recorded from the working interactive rosi setup on 2026-07-17.
set -euo pipefail

if ! declare -F module >/dev/null; then
    lmod_init=/etc/profile.d/lmod.sh
    if [[ ! -r "${lmod_init}" && -n "${MODULEPATH:-}" ]]; then
        lmod_init="${BASH_ENV:-/rosi/shared/lmod/lmod/init/bash}"
    fi
    if [[ ! -r "${lmod_init}" ]]; then
        printf 'Lmod initialization script is not readable: %s\n' "${lmod_init}" >&2
        return 1 2>/dev/null || exit 1
    fi
    source "${lmod_init}"
fi

module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4
module list

# A synchronized campaign may provide concrete defaults while preserving all
# explicit environment overrides used by generic/local tests.
experiment_defaults="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/experiment-01.env.sh"
if [[ -r "${experiment_defaults}" ]]; then
    source "${experiment_defaults}"
fi
