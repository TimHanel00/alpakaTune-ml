# alpakaTune ML

This repository owns research-scale data generation and model development for
[alpakaTune](https://github.com/TimHanel00/alpaka3-tuner). It deliberately does
not own alpakaTune runtime code. Dependency direction is one-way: a pinned
`alpaka3-tuner/` Git submodule supplies the executables used for collection;
alpakaTune never imports this Python package.

The final reviewed `.atml` model and its model card may be promoted into
alpakaTune for native deployment. Raw histories, datasets, checkpoints, and
unapproved artifacts remain outside both Git repositories.

## What is implemented

- Scheduler-neutral full-exhaustive campaign execution, with exact measurement
  policy and a hard failure when any legal candidate is missing.
- Current alpakaTune history schema-v9 import and preferred structured
  schema-v10 import.
- Immutable JSONL datasets with mandatory train, validation, and test manifests.
  Devices, surfaces, and row IDs must be disjoint across all three splits.
- A compact, width-configurable three-member DeepSets ranker trained from random initialization with
  surface-balanced pair sampling, extra weight on the fastest decile, and an
  auxiliary log-runtime loss.
- Versioned `ATMLART1` export plus native-equivalent NumPy evaluation, online
  ridge-residual simulation, model cards, and checksums.
- Complete-candidate plots and an HTML surface switcher. The plots show measured
  candidates directly; they never project only a best-so-far curve.
- Rosi Slurm templates for exclusive-node collection arrays and independent
  ensemble-member training arrays with a validated dependency merge.

## Installation

Clone the pinned runtime dependency with the repository:

```bash
git clone --recurse-submodules https://github.com/TimHanel00/alpakaTune-ml.git
cd alpakaTune-ml
git submodule update --init --recursive
```

The base package is enough to collect and prepare data:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Install training or plotting dependencies only where needed:

```bash
.venv/bin/python -m pip install -e '.[train]'
.venv/bin/python -m pip install -e '.[plot,test]'
```

The rosi scripts use the site's compiler, CMake, CUDA, and Python modules and do
not reinstall them. See [docs/rosi.md](docs/rosi.md).

On Rosi, first run `hpc/rosi/setup-login.sh` interactively: it configures the
synced `alpaka3-tuner` source directly with CUDA and OpenMP enabled, CpuSerial
disabled, builds with `make -j`, and pre-creates the collection and training
venvs. Dependency sources are pre-synced and CMake runs fully disconnected; no
Git metadata or network fetch is needed remotely. All measurements, dataset
construction, training, merging, and comparison run through `sbatch`. The first
experiment is an exact three-task array, with each task running GPU then CPU on
one exclusive GPU node:

```bash
export PAIR_MANIFEST=/absolute/path/campaign-pairs.rosi.txt
export DATASET_OUTPUT=/project/path/datasets/first-three-pairs
export ALPAKATUNE_ML_VENV="$ALPAKATUNE_ML_SOURCE/.venv"
"$ALPAKATUNE_ML_SOURCE/hpc/rosi/submit-collection-dataset.sh"
```

The dependent dataset job maps task 0 to train, task 1 to validation, and task 2
to test, with one CPU and one GPU device in each split.
The CMake, configuration, documentation, and scheduler assets are operational
files from the Git checkout; they are intentionally not bundled into the Python
wheel.

## 1. Generate complete exhaustive surfaces

Copy `configs/campaign.example.yaml` for CUDA or
`configs/campaign.cpu.example.yaml` for host CPU collection, replace every
revision placeholder with a full commit, choose an explicit device ID or
`device_id: auto`, and list executable commands. Then run:

```bash
alpakatune-ml collect configs/campaign.local.yaml \
  --output /bulk/path/campaigns/local-gpu-v1
```

The Rosi manifests use `platform.device_id: auto`; the collector resolves and
pins the ID from the first complete history in each array element. Explicit IDs
remain supported for controlled local campaigns.

Every generated tuner config uses:

```yaml
tuning:
  strategy: exhaustive
  warmup_runs: 1
  runs_per_candidate: 3
  minimum_runs_per_candidate: 3
  mann_whitney_early_stop: false
  max_consecutive_runs: 4
```

Both `maximum_executions` and `maximum_retired_configurations` are removed. A
surface is accepted only when `completion_reason == all_configurations`, every
legal candidate has samples and an estimate, and
`retired_configuration_count + rejected_count == candidate_count`.

This distinction explains the earlier observation that exhaustive often visited
only roughly half of a 4,096-candidate space. The strategy iterates candidates;
the 40,000-launch cap was consumed by warmups and repeated measurements before
all candidates could retire. It was a capped run, not a full exhaustive oracle.
Finite-step examples must also keep enqueueing in their benchmark-only mode until
all tuner contexts finish.

Use only complete exhaustive surfaces as base-model labels. Random, Bayesian
optimization, and simulated annealing histories are evaluation baselines, not
training data.

## 2. Build leakage-safe datasets

Define whole-device membership with all three mandatory splits, following
`configs/splits.example.yaml`, then prepare the dataset:

```bash
alpakatune-ml build-dataset /bulk/path/campaigns \
  --splits configs/splits.local.yaml \
  --output /bulk/path/datasets/cross-device-v1

alpakatune-ml validate-splits \
  --train /bulk/path/datasets/cross-device-v1/train.manifest.json \
  --validation /bulk/path/datasets/cross-device-v1/validation.manifest.json \
  --test /bulk/path/datasets/cross-device-v1/test.manifest.json
```

One `(workload context, device, legal candidate)` robust runtime is one label;
the three raw timings improve that label and are not independent data points.
Split ownership is by entire device, never random rows. The first Rosi experiment
uses three disjoint CPU/GPU pairs: task 0 is training, task 1 is validation, and
task 2 is the untouched test pair.

The six examples listed in `configs/campaign.example.yaml` have nine contexts
and 318,432 candidates per device at the currently pinned alpakaTune revision.
The six-device design therefore contains 1,910,592 labels and 5,731,776
initial timed measurements; revisions that change spaces require recounting.
The earlier single-A30 archive (32,635 labels, roughly 10.3%
coverage) is useful only for pipeline prototyping and cannot support a
cross-device claim.

Schema-v10 histories carry `metadata.model_context` with a strategy/device-
independent workload ID, device class, named numeric context features, and
ordered tuning-dimension descriptors. Schema-v9 imports explicitly derive a
smaller fallback context; they do not invent unavailable device capabilities.
The complete wire contract is documented in
[docs/data-contract.md](docs/data-contract.md).

## 3. Train and export

Copy `configs/training.example.yaml` for local CPU training or
`configs/training.rosi.example.yaml` for CUDA training on Rosi and run:

```bash
alpakatune-ml train \
  --train /bulk/path/datasets/cross-device-v1/train.manifest.json \
  --validation /bulk/path/datasets/cross-device-v1/validation.manifest.json \
  --test /bulk/path/datasets/cross-device-v1/test.manifest.json \
  --config configs/training.local.yaml \
  --output /bulk/path/artifacts/cross-device-v1.atml
```

The model encodes a variable number of tuning dimensions, mean/max pools them,
combines them with automatically persisted tuner/device context, and uses
separate CPU/GPU output adapters. Tunable names use deterministic signed FNV-1a
hash buckets; strategy name and candidate evaluation order are excluded.

The deployment defaults use token widths `[16, 32]` and a 32-value embedding;
the artifact records parameter bytes and per-candidate multiply-add counts.
Training samples surfaces uniformly so a large space such as grayScale cannot
dominate. Three independently seeded members are exported. Validation chooses
each member's best epoch; the test split is touched only after selection.

For multi-node Slurm training, `train-member` writes one valid single-member
ATMLART1 artifact per array element and `merge-members` validates and combines
them without changing the deployment format. On Rosi,
`hpc/rosi/submit-training-array.sh` submits the exact member array followed by
an `afterok` merge job. Only the merge stage evaluates test labels.

See [docs/artifact-format.md](docs/artifact-format.md) for the binary contract.
The compact artifact has no Python, PyTorch, ONNX, or private-header dependency.

## 4. Evaluate and inspect

```bash
alpakatune-ml evaluate \
  --artifact /bulk/path/artifacts/cross-device-v1.atml \
  --split /bulk/path/datasets/cross-device-v1/test.manifest.json \
  --output /bulk/path/evaluations/cross-device-v1.json

alpakatune-ml plot \
  --split /bulk/path/datasets/cross-device-v1/test.manifest.json \
  --artifact /bulk/path/artifacts/cross-device-v1.atml \
  --output /bulk/path/plots/cross-device-v1

alpakatune-ml benchmark-artifact \
  --artifact /bulk/path/artifacts/cross-device-v1.atml \
  --split /bulk/path/datasets/cross-device-v1/test.manifest.json \
  --output /bulk/path/evaluations/reference-latency.json

alpakatune-ml evaluate-search /bulk/path/search-campaigns \
  --oracle /bulk/path/datasets/cross-device-v1/test.manifest.json \
  --output /bulk/path/evaluations/search-baselines.json
```

Evaluation reports top-1/5/10 median regret and log-runtime error. It also
simulates the frozen-ensemble plus ridge-residual adapter at 16 through 1,024
observations per surface. Model promotion into alpakaTune remains a normal
reviewed change: verify checksum, artifact/native compatibility, held-out-device
metrics, licensing/provenance, model card, and the repository's model-size gate.
Native promotion must additionally demonstrate roughly 1–5 µs one-time scoring
latency per candidate and a cached recommendation below 10 µs. If three members
miss the gate, the supported follow-up is teacher-to-single-student distillation
with a softplus uncertainty head, not an unmeasured width reduction.
`benchmark-artifact` is a Python/NumPy smoke benchmark; the native C++ benchmark
in alpakaTune is the promotion gate.
`evaluate-search` maps Bayesian-optimization, simulated-annealing, and random
configurations back to exhaustive oracle labels. Those search histories are
comparison inputs only and never enter the training split.

## Boundaries and known limitations

- Version one targets known kernel families on unseen devices. It makes no
  unseen-kernel transfer claim; that needs instruction/resource or semantic
  kernel features in a later schema.
- Context features are taken from information alpakaTune already owns, so
  application `makeTuner` and `enqueue` calls need no ML-specific arguments.
- The collector has no scheduler logic. Slurm wrappers set up rosi and call the
  same CLI used locally.
- Large data streaming, refinement of the fastest or most uncertain 5%, and
  recurring HPC retraining are follow-up operational work.

## Tests

```bash
PYTHONPATH=src pytest -q
```

Fixtures are intentionally tiny. Tests cover feature hashing, incomplete-space
rejection, whole-device leakage, duplicate surface/row protection, and exact
artifact byte/tensor round trips without committing production data.
