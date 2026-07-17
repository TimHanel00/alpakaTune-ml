# History and dataset contract

## Input histories

Schema 9 is supported as a legacy importer. It provides candidate
configurations, raw samples, robust estimates, kernel/device display strings,
launch text, and identity entries. The importer derives only features supported
by those fields and marks no guessed hardware capability.

Schema 10 is the preferred contract. Each context's `metadata.model_context`
contains:

```json
{
  "feature_schema_version": 1,
  "workload_id": "strategy-and-device-independent-id",
  "device_class": "cpu-or-gpu",
  "context_features": {"stable_feature_name": 1.0},
  "dimensions": [
    {
      "name": "blockSize",
      "kind": "launch",
      "cardinality": 3,
      "component_index": 0,
      "vector_arity": 1,
      "concrete_values": [64, 128, 256]
    }
  ]
}
```

The external repository treats this object as data. It does not import private
alpakaTune types. Named context features are reordered according to the trained
artifact. Unknown names are ignored during inference and missing names are raw
zero before normalization.

A history is eligible for oracle/training data only if it is exhaustive, ends
with `all_configurations`, does not hit a budget, has complete candidate arrays,
accounts for all rejected candidates, and has at least three finite positive
timings plus a finite positive estimate for every legal candidate.

## Normalized rows

Prepared JSONL contains one row per legal `(workload, device, candidate)` label:

- Stable `row_id`, `surface_id`, `workload_id`, and `device_id`.
- CPU/GPU class and candidate index/configuration.
- Ordered dimension feature tokens and named context features.
- Robust runtime label and all source timings.
- Source history hash, schema version, and context fingerprint.

Row and surface IDs omit strategy and observation order. The source hash remains
provenance rather than model input.

## Immutable splits

`build-dataset` writes `train.jsonl`, `validation.jsonl`, and `test.jsonl`, each
with a checksum-bearing split manifest, plus `dataset.manifest.json`. All three
splits are mandatory. Every observed device must be assigned exactly once and
every configured device must have data. Validation rejects cross-split device,
surface, and row-ID leakage as well as a modified JSONL checksum.

Dataset identity hashes complete surface summaries and the split assignment.
The wall-clock creation time is descriptive and does not affect identity.

