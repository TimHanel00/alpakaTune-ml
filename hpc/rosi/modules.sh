#!/usr/bin/env bash
# Module stack recorded from the working interactive rosi setup on 2026-07-17.
set -euo pipefail

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
