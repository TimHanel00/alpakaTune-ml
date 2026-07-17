#!/usr/bin/env bash
# Module stack recorded from the working interactive rosi setup on 2026-07-17.
set -euo pipefail

module purge
module load gcc/14.2.0
module load cmake/4.0.3
module load cuda/12.8
module load python/3.12.4
module list

