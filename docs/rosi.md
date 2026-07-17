# Rosi workflow

Rosi receives an RSYNC snapshot without Git metadata. Do not run Git on Rosi.
The login node is used only to configure/build the synced sources and create the
two virtual environments. Collection, dataset construction, training, merging,
baseline execution, strategy comparison, plotting, and evaluation all run as
Slurm batch jobs.

Every GPU job requests one node, one GPU, and `--exclusive`. No script selects a
partition and no script uses `srun`.

## 1. Prepare source dependencies locally

Before RSYNC, copy the already checked-out FetchContent source trees into this
layout (only source contents are needed; RSYNC excludes their `.git` metadata):

```text
alpakaTune-ml/dependencies/
  alpaka3/
  yaml_cpp/
  nlohmann_json/
```

The expected local sources are the pinned trees normally found under the
alpakaTune build's `_deps/alpaka3-src`, `_deps/yaml_cpp-src`, and
`_deps/nlohmann_json-src`. Synchronize those three directories with targeted
RSYNC operations. Rosi configuration uses `FETCHCONTENT_FULLY_DISCONNECTED=ON`
and explicit `FETCHCONTENT_SOURCE_DIR_*` values, so CMake cannot clone or update
them remotely.

## 2. Prepare the synced tree on the login node

```bash
RSYNC rosi
ssh rosi

export ALPAKATUNE_ML_SOURCE=/absolute/path/alpakaTune-ml
export ALPAKATUNE_ML_BUILD="$ALPAKATUNE_ML_SOURCE/build"
bash "$ALPAKATUNE_ML_SOURCE/hpc/rosi/setup-login.sh"
```

The script configures `$ALPAKATUNE_ML_SOURCE/alpaka3-tuner` directly into
`$ALPAKATUNE_ML_BUILD/alpaka3-tuner`, with CUDA and OpenMP enabled, CpuSerial
disabled, and runtime tests disabled. It builds from that directory using
`make -j`. It also configures and builds the pre-synced `dependencies/alpaka3`
examples without alpakaTune instrumentation under
`$ALPAKATUNE_ML_BUILD/alpaka-baseline`.

The collection environment is `$ALPAKATUNE_ML_SOURCE/.venv` by default and
includes plotting support. The training environment is
`$ALPAKATUNE_ML_SOURCE/.venv-train` and supplies PyTorch. Override the dependency
source root with `ALPAKATUNE_DEPENDENCY_ROOT`, the environment creation locations
with `ALPAKATUNE_COLLECTION_VENV` and `ALPAKATUNE_TRAINING_VENV`, and the paths
passed to jobs with `ALPAKATUNE_ML_VENV` and `ALPAKATUNE_TRAINING_VENV`.

## 3. Define three CPU/GPU pairs

Before synchronization, copy the two campaign templates, replace their three
revision placeholders with immutable source identifiers, and create the site
manifest:

```bash
cp configs/campaign.example.yaml configs/campaign.rosi-gpu.yaml
cp configs/campaign.cpu.example.yaml configs/campaign.rosi-cpu.yaml
cp configs/campaign-pairs.example.txt configs/campaign-pairs.rosi.txt
```

The GPU template explicitly uses `cuda:nvidiaGpu` plus `gpuCuda`; the CPU
template uses `host:cpu` plus `cpuOmpBlocks`. The pair manifest contains exactly
three non-comment rows:

```text
GPU_CONFIG CPU_CONFIG PAIR_OUTPUT
```

Paths are absolute and contain no whitespace. Manifest row N is array task N.
Each task runs GPU and then CPU sequentially on the same exclusive GPU node and
writes:

```text
PAIR_OUTPUT/
  allocation.json
  gpu/campaign.json
  cpu/campaign.json
```

`allocation.json` records the node, CPU model, visible GPU UUID/name/memory, and
Slurm attempt IDs. Resubmission uses `--resume`; an allocation with a different
node or GPU identity is rejected rather than mixed with partial results.

## 4. Submit collection and dataset construction

```bash
export ALPAKATUNE_ML_SOURCE=/absolute/path/alpakaTune-ml
export PAIR_MANIFEST=/absolute/path/alpakaTune-ml/configs/campaign-pairs.rosi.txt
export DATASET_OUTPUT=/project/path/datasets/first-three-pairs
export ALPAKATUNE_ML_VENV="$ALPAKATUNE_ML_SOURCE/.venv"

"$ALPAKATUNE_ML_SOURCE/hpc/rosi/submit-collection-dataset.sh"
```

The noninteractive helper validates and hashes the manifest, submits
`collect-paired-array.sbatch` as exactly `0-2`, then submits
`prepare-dataset.sbatch` with `afterok` on the complete array. It prints both job
IDs. The legacy `collect-array.sbatch` interface remains available unchanged for
older independent manifests.

The dataset job validates all six full-exhaustive campaigns and requires six
unique device IDs. Repeated hardware models across tasks fail rather than being
presented as cross-device generalization. Split assignment is automatic:

- task 0 CPU + GPU: train;
- task 1 CPU + GPU: validation;
- task 2 CPU + GPU: test.

The generated split YAML defaults to `${DATASET_OUTPUT}.splits.yaml`; override it
with `PAIR_SPLIT_CONFIG`. A completed immutable dataset is checksum-validated and
skipped on resubmission. A partial nonempty dataset remains a hard error.

## 5. Submit three-member training and merge

```bash
export DATASET_ROOT=/project/path/datasets/first-three-pairs
export TRAINING_CONFIG="$ALPAKATUNE_ML_SOURCE/configs/training.rosi.example.yaml"
export MODEL_MEMBER_DIR=/project/path/models/first-three-pairs-members
export MODEL_OUTPUT=/project/path/models/first-three-pairs.atml
export ALPAKATUNE_TRAINING_VENV="$ALPAKATUNE_ML_SOURCE/.venv-train"

"$ALPAKATUNE_ML_SOURCE/hpc/rosi/submit-training-array.sh"
```

With the provided configuration this submits member array `0-2` and an `afterok`
merge. Each member owns an exclusive GPU node. Members select using only train
and validation labels; the merge validates provenance and evaluates test labels.

## 6. Submit same-node strategy comparison

```bash
export COMPARE_OUTPUT=/project/path/evaluations/first-three-pairs
export MODEL_ARTIFACT=/project/path/models/first-three-pairs.atml
export ORACLE_MANIFEST=/project/path/datasets/first-three-pairs/test.manifest.json
export COMPARE_NODE=the-node-recorded-in-pair-2-allocation.json
# Optional when submitting before the model merge has completed:
export MERGE_JOB_ID=123456
export ALPAKATUNE_ML_VENV="$ALPAKATUNE_ML_SOURCE/.venv"

"$ALPAKATUNE_ML_SOURCE/hpc/rosi/submit-comparison.sh"
```

The helper pins the job with `--nodelist=$COMPARE_NODE`; when `MERGE_JOB_ID` is
set it also uses `--dependency=afterok:$MERGE_JOB_ID`. It never selects a
partition. The exclusive GPU-node job records its allocation, runs the instrumentation-free
Alpaka examples ten times, then runs one GPU and one OpenMP benchmark campaign.
Each uses a 40,000-execution cap with exhaustive, random, simulated annealing,
Bayesian optimization, and learned-hybrid in one invocation of the core benchmark
runner, with `--model`, `--tune-until-terminal`, and `--resume`. Learned-hybrid
uses `MODEL_ARTIFACT`; the other strategies remain model-independent.

The job then creates the benchmark plots/dashboard and invokes
`alpakatune-ml evaluate-search` for both backends. `ORACLE_MANIFEST` must contain
the same device model as the comparison allocation for surface IDs to match.
Override the default whitespace-separated example list with `COMPARE_EXAMPLES`.
The baseline raw logs contain every enabled backend; the summary provides
separate CpuOmpBlocks and GpuCuda runtime overlays where the upstream example
reports a comparable kernel or time-step duration.

Campaigns, datasets, models, and comparisons belong in project/scratch storage,
outside the RSYNC source checkout.
