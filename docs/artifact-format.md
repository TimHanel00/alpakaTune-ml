# ATMLART1 model artifact

The deployment artifact is intentionally independent of Python, PyTorch, and
private alpakaTune C++ layouts. All integers use little-endian byte order and all
tensors are contiguous row-major IEEE-754 `float32` values.

| Offset | Value |
| --- | --- |
| 0 | Eight ASCII bytes `ATMLART1` |
| 8 | Unsigned little-endian 32-bit metadata byte count `N` |
| 12 | `N` bytes of canonical UTF-8 JSON |
| 12 + N | Tensor payloads concatenated in metadata order |

Canonical JSON uses sorted object keys, compact separators, UTF-8, and no NaN
or infinity. Required metadata includes:

- `artifact_version: 1`, `feature_schema_version: 1`, and
  `architecture: deepsets_ensemble_v1`.
- Ensemble/shape fields: one or three members, 18 dimension features, two
  positive `token_hidden_sizes`, a positive `embedding_size`, and CPU/GPU
  adapters. Deployment defaults are `[16, 32]` and 32 respectively.
- Ordered `feature_names.context` and `feature_names.dimension` arrays.
- `hash_buckets.dimension_name: 8`, using FNV-1a 64. The selected bucket is
  `hash % 8`; bit 63 selects the sign.
- Mean and scale arrays for raw context and dimension features. A scale smaller
  than `1e-6` is exported as one. Missing named context features are raw zero
  before normalization; extra runtime features are ignored.
- An ordered `tensors` array of `{name, shape}` objects.

Each member contains `token.0`, `token.2`, `context.0`, `context.2`, and
`adapters.cpu|gpu` weights and biases. Tensor order is enforced by the exporter
and tested by the reader. Linear weights use PyTorch's `[out, in]` shape, so
native inference computes `y = W*x + b`.

`inference_profile` records parameter/payload size, multiply-adds per dimension,
fixed multiply-adds per member, and the deployment latency targets. The runtime
gate is approximately 1–5 µs per candidate during one-time scoring and less than
10 µs for a recommendation after scoring/caching.

If three compact members still miss that gate, a later promotion may distill
them into a single member. A distilled student can add per-device tensors named
`uncertainty.cpu|gpu.weight|bias`, declare
`uncertainty_head: softplus_stddev`, and preserve uncertainty-guided exploration
without executing all three teacher members. Distillation is not silently
enabled by this training pipeline; it requires separate parity and regret tests.

The produced `.sha256` file and model card are promotion inputs. The final
approved artifact may be copied into alpakaTune and installed under
`share/alpakaTune/models/`; datasets and checkpoints must not be copied there.
