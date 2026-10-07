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
  Whole-device datasets reject device, surface, and row leakage. Explicit
  configuration-holdout datasets reject row leakage and aggregate repeated
  measurements before deterministically assigning configurations.
- A compact, width-configurable three-member DeepSets ranker trained from random initialization with
  surface-balanced pair sampling, extra weight on the fastest decile, and an
  auxiliary log-runtime loss.
- Versioned `ATMLART1` export plus native-equivalent NumPy evaluation, online
  ridge-residual simulation, model cards, and checksums.
- Complete-candidate plots and an HTML surface switcher. The plots show measured
  candidates directly; they never project only a best-so-far curve.

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

Build the pinned runtime examples on the machine that will collect measurements.
The CMake wrapper enables CUDA by default; set
`ALPAKATUNE_ML_ENABLE_CUDA=OFF` for a CPU-only build:

```bash
cmake -S . -B build -DALPAKATUNE_ML_ENABLE_CUDA=ON
cmake --build build --parallel
source build/generated/alpakatune-paths.sh
```

The generated shell file exports `ALPAKATUNE_SOURCE` and `ALPAKATUNE_BUILD`,
which the campaign templates use to find the runtime configuration and examples.
Keep campaign outputs, datasets, checkpoints, and models outside the checkout.
CMake files and example configurations are available in the Git checkout and
are not included in the Python wheel.

## 1. Generate complete exhaustive surfaces

Copy `configs/campaign.example.yaml` for CUDA or
`configs/campaign.cpu.example.yaml` for host CPU collection, replace every
revision placeholder with a full commit, choose an explicit device ID or
`device_id: auto`, and list executable commands. Then run:

```bash
alpakatune-ml collect configs/campaign.local.yaml \
  --output /bulk/path/campaigns/local-gpu-v1
```

With `platform.device_id: auto`, the collector resolves and pins the device ID
from the first complete history. Explicit IDs are also supported.

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

Execution caps include warmups and repeated measurements, so a capped exhaustive
run may stop before covering every candidate. Finite-step examples must keep
enqueueing in their benchmark-only mode until all tuner contexts finish.

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
the raw timings improve that label and are not independent data points.
Cross-architecture claims require whole-device splits with an entire architecture
held out for final evaluation. Repeated measurements on the same device do not
provide cross-device evidence. Configuration-holdout splits test unseen
configurations on known devices and aggregate repeats before assigning rows.

Schema-v10 histories carry `metadata.model_context` with a strategy/device-
independent workload ID, device class, named numeric context features, and
ordered tuning-dimension descriptors. Schema-v9 imports explicitly derive a
smaller fallback context; they do not invent unavailable device capabilities.
The complete wire contract is documented in
[docs/data-contract.md](docs/data-contract.md).

## 3. Train and export

Copy `configs/training.example.yaml` for local CPU training or
`configs/training.cuda.example.yaml` for CUDA training and run:

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
them without changing the deployment format. Schedule one `train-member`
invocation per member, then run `merge-members` after all members finish.
Only the merge stage evaluates test labels.

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
- The collector has no scheduler logic. Use your scheduler to invoke the CLI
  with the appropriate resources and environment.
- Large data streaming, refinement of the fastest or most uncertain 5%, and
  recurring HPC retraining are follow-up operational work.

## Tests

Install the development dependencies and run the source checks:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/pre-commit install
.venv/bin/pre-commit run --all-files
```

Run the Python tests with:

```bash
PYTHONPATH=src pytest -q
```

Fixtures are intentionally tiny. Tests cover feature hashing, incomplete-space
rejection, whole-device leakage, duplicate surface/row protection, and exact
artifact byte/tensor round trips without committing production data.
