# Rosi workflow

Rosi preparation happens interactively on a login node. Slurm jobs collect
measurements, build datasets, or train models; they never configure/build
alpakaTune, create a virtual environment, or install Python packages.

All scripts load the module stack verified interactively on Rosi:

```text
gcc/14.2.0
cmake/4.0.3
cuda/12.8
python/3.12.4
```

No script selects a Slurm partition. The GPU array requests one GPU with
`--gres=gpu:1`, one node, and `--exclusive`. Consequently, every concurrently
running array element owns a complete node; the array concurrency limit controls
how many nodes can run at once.

## 1. Sync and prepare on the login node

Initialize the pinned dependency locally, synchronize the complete repository
with the site command, log in, and select an absolute build path:

```bash
git submodule update --init --recursive
RSYNC rosi
ssh rosi

export ALPAKATUNE_ML_SOURCE=/absolute/path/alpakaTune-ml
export ALPAKATUNE_ML_BUILD=/absolute/path/alpakaTune-ml/build
```

Run the login-node setup once:

```bash
bash "$ALPAKATUNE_ML_SOURCE/hpc/rosi/setup-login.sh"
```

The script sources `hpc/rosi/modules.sh`, verifies the nested `alpaka3-tuner`
checkout against the repository's gitlink, configures a Unix Makefiles build
with CUDA and the examples enabled, leaves Rosi's login-node resource limits
unchanged, and then runs exactly:

```bash
(cd "$ALPAKATUNE_ML_BUILD" && make -j)
```

The nested source and build paths are generated from that pinned checkout. A
script run in a child shell cannot export variables back into the login shell,
so load them once before preparing campaign files or submitting collection
jobs:

```bash
source "$ALPAKATUNE_ML_BUILD/generated/alpakatune-paths.sh"
```

It also creates two project-local environments on the login node:

- `$ALPAKATUNE_ML_SOURCE/.venv` contains collection/dataset dependencies.
- `$ALPAKATUNE_ML_SOURCE/.venv-train` contains the training extra and PyTorch.

Rosi has no PyTorch module; compiler, CMake, CUDA, and Python still come only
from modules. To place venvs elsewhere, set `ALPAKATUNE_COLLECTION_VENV` and
`ALPAKATUNE_TRAINING_VENV` while running setup, then pass the selected path as
`ALPAKATUNE_ML_VENV` to the corresponding job.

Create site campaign files from the examples and replace all three revision
placeholders with full immutable commits. Keep `platform.device_id: auto`:

```bash
cp "$ALPAKATUNE_ML_SOURCE/configs/campaign.example.yaml" \
   "$ALPAKATUNE_ML_SOURCE/configs/campaign.rosi-gpu.yaml"
cp "$ALPAKATUNE_ML_SOURCE/configs/campaign.cpu.example.yaml" \
   "$ALPAKATUNE_ML_SOURCE/configs/campaign.rosi-cpu.yaml"
```

`auto` resolves to the device ID persisted by the first validated history and
requires every later context in that campaign to report the same device. The
resolved value is written to `campaign.json`. Resuming also validates completed
histories before skipping them.

## 2. Submit the GPU collection array

Create an absolute-path manifest following
`configs/campaign-array.example.txt`. Each non-comment line contains exactly a
campaign YAML and a unique output directory. For four lines:

```bash
export CAMPAIGN_ARRAY_FILE=/absolute/path/campaign-array.rosi-gpu.txt
export CAMPAIGN_ARRAY_SHA256="$(sha256sum "$CAMPAIGN_ARRAY_FILE" | awk '{print $1}')"
export ALPAKATUNE_ML_VENV="$ALPAKATUNE_ML_SOURCE/.venv"

sbatch --array=0-3%4 \
  "$ALPAKATUNE_ML_SOURCE/hpc/rosi/collect-array.sbatch"
```

There is deliberately no `--partition`. `--array=0-3%4` allows four elements to
run concurrently; each requests `--nodes=1`, `--gres=gpu:1`, and `--exclusive`
from the script, so Slurm allocates one exclusive node per running element.

The submitted IDs must be exactly `0..N-1`, where `N` is the number of
non-comment manifest entries; `%4` only limits concurrency. Every task checks
that shape and the immutable manifest checksum before selecting its row. The
wrapper normalizes all paths and rejects duplicate or nested output roots, so
array elements cannot intentionally write the same campaign tree. Do not edit
the manifest after computing `CAMPAIGN_ARRAY_SHA256`.

Slurm may assign the same GPU model to multiple elements. After completion,
inspect each output's resolved `campaign.json` and retain/merge campaigns by
device ID deliberately; array execution alone does not guarantee four distinct
GPU architectures.

For a single campaign instead, export `CAMPAIGN_CONFIG` and `CAMPAIGN_OUTPUT`
and submit `hpc/rosi/collect.sbatch`.

## 3. Submit the analogous CPU collection array

Use `configs/campaign-cpu-array.example.txt` with the CPU campaign YAML:

```bash
export CAMPAIGN_ARRAY_FILE=/absolute/path/campaign-array.rosi-cpu.txt
export CAMPAIGN_ARRAY_SHA256="$(sha256sum "$CAMPAIGN_ARRAY_FILE" | awk '{print $1}')"
export ALPAKATUNE_ML_VENV="$ALPAKATUNE_ML_SOURCE/.venv"

sbatch --array=0-3%4 \
  "$ALPAKATUNE_ML_SOURCE/hpc/rosi/collect-cpu-array.sbatch"
```

The CPU array also requests one exclusive node per running element and no
partition, but it does not request a GPU. `device_id:auto` records the assigned
CPU model. `hpc/rosi/collect-cpu.sbatch` remains the single-campaign equivalent.

## 4. Submit dataset construction

After choosing one complete campaign per intended device and defining strict
whole-device splits, submit dataset validation/preparation to a compute node:

```bash
export HISTORY_ROOTS=/project/path/campaigns
export SPLIT_CONFIG="$ALPAKATUNE_ML_SOURCE/configs/splits.rosi.yaml"
export DATASET_OUTPUT=/project/path/datasets/cross-device-v1
export ALPAKATUNE_ML_VENV="$ALPAKATUNE_ML_SOURCE/.venv"

sbatch "$ALPAKATUNE_ML_SOURCE/hpc/rosi/prepare-dataset.sbatch"
```

Bulk histories, datasets, checkpoints, and models belong in project/scratch
storage, not in either Git checkout or the later `RSYNC rosi` transfer.

## 5. Submit training

The three ensemble members can train independently on three array nodes. A
short dependency job validates their preprocessing, feature schema, dataset,
configuration, indexes, and seeds; evaluates the untouched test split; and
emits the normal ATMLART1 ensemble:

```bash
export DATASET_ROOT=/project/path/datasets/cross-device-v1
export TRAINING_CONFIG="$ALPAKATUNE_ML_SOURCE/configs/training.rosi.example.yaml"
export MODEL_MEMBER_DIR=/project/path/models/cross-device-v1-members
export MODEL_OUTPUT=/project/path/models/cross-device-v1.atml
export ALPAKATUNE_TRAINING_VENV="$ALPAKATUNE_ML_SOURCE/.venv-train"

"$ALPAKATUNE_ML_SOURCE/hpc/rosi/submit-training-array.sh"
```

The helper derives the exact `0-(ensemble_size-1)` array from the training
config and submits `merge-members.sbatch` with an `afterok` dependency. Each
running member requests one exclusive GPU node and no partition. Member jobs
use training and validation labels only for model and epoch selection; test
labels are evaluated only by the merge job. The original single-node
`train.sbatch` remains available for small runs and parity checks.

Before promoting a result, run evaluation and the native alpakaTune latency
benchmark, verify the model card/checksum, and confirm untouched test-device
metrics. The helper prints both submitted job IDs.
