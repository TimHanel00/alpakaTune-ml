#!/usr/bin/env bash
#SBATCH --job-name=alpakatune-cpu-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# The GPU and CPU arrays intentionally share all manifest validation and
# collection behavior. Only this allocation header differs: no GPU is requested.
set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_directory}/collect-array.sbatch"

