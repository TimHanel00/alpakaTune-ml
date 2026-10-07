"""Portable alpakaTune learned-model artifact reader and writer."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from .util import canonical_json_bytes, sha256_file


ARTIFACT_MAGIC = b"ATMLART1"
ARTIFACT_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
ARCHITECTURE = "deepsets_ensemble_v1"


class ArtifactError(ValueError):
    pass


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    expected = {
        "artifact_version": ARTIFACT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ArtifactError(f"{name} must be {value!r}")
    ensemble_size = metadata.get("ensemble_size")
    if ensemble_size not in {1, 3}:
        raise ArtifactError("ensemble_size must be 1 or 3")
    if metadata.get("dimension_feature_count") != 18:
        raise ArtifactError("dimension_feature_count must be 18")
    token_hidden_sizes = metadata.get("token_hidden_sizes")
    if (
        not isinstance(token_hidden_sizes, list)
        or len(token_hidden_sizes) != 2
        or not all(isinstance(item, int) and item > 0 for item in token_hidden_sizes)
    ):
        raise ArtifactError("token_hidden_sizes must contain two positive integers")
    embedding_size = metadata.get("embedding_size")
    if not isinstance(embedding_size, int) or embedding_size <= 0:
        raise ArtifactError("embedding_size must be positive")
    context_count = metadata.get("context_feature_count")
    if not isinstance(context_count, int) or context_count <= 0:
        raise ArtifactError("context_feature_count must be positive")
    feature_names = metadata.get("feature_names")
    if (
        not isinstance(feature_names, Mapping)
        or not isinstance(feature_names.get("context"), list)
        or len(feature_names["context"]) != context_count
        or len(set(feature_names["context"])) != context_count
        or not isinstance(feature_names.get("dimension"), list)
        or len(feature_names["dimension"]) != 18
    ):
        raise ArtifactError("feature_names do not match declared feature counts")
    preprocessing = metadata.get("preprocessing")
    expected_lengths = {
        "context_mean": context_count,
        "context_scale": context_count,
        "dimension_mean": 18,
        "dimension_scale": 18,
    }
    if not isinstance(preprocessing, Mapping) or any(
        not isinstance(preprocessing.get(name), list)
        or len(preprocessing[name]) != length
        for name, length in expected_lengths.items()
    ):
        raise ArtifactError("preprocessing arrays do not match declared feature counts")
    if any(float(value) <= 0.0 for name in ("context_scale", "dimension_scale") for value in preprocessing[name]):
        raise ArtifactError("preprocessing scales must be positive")
    expected = []
    first_hidden, second_hidden = token_hidden_sizes
    shapes = {
        "token.0.weight": [first_hidden, 18],
        "token.0.bias": [first_hidden],
        "token.2.weight": [second_hidden, first_hidden],
        "token.2.bias": [second_hidden],
        "context.0.weight": [embedding_size, 2 * second_hidden + context_count],
        "context.0.bias": [embedding_size],
        "context.2.weight": [embedding_size, embedding_size],
        "context.2.bias": [embedding_size],
        "adapters.cpu.weight": [1, embedding_size],
        "adapters.cpu.bias": [1],
        "adapters.gpu.weight": [1, embedding_size],
        "adapters.gpu.bias": [1],
    }
    if metadata.get("uncertainty_head") is not None:
        if metadata["uncertainty_head"] != "softplus_stddev":
            raise ArtifactError("unsupported uncertainty_head")
        shapes.update(
            {
                "uncertainty.cpu.weight": [1, embedding_size],
                "uncertainty.cpu.bias": [1],
                "uncertainty.gpu.weight": [1, embedding_size],
                "uncertainty.gpu.bias": [1],
            }
        )
    for member in range(ensemble_size):
        expected.extend(
            {"name": f"members.{member}.{name}", "shape": shape}
            for name, shape in shapes.items()
        )
    if metadata.get("tensors") != expected:
        raise ArtifactError("tensor inventory/order/shape does not match DeepSets v1")


def write_artifact(
    path: Path,
    metadata: Mapping[str, Any],
    tensors: Sequence[tuple[str, np.ndarray]],
) -> str:
    document = dict(metadata)
    document["tensors"] = []
    payload = bytearray()
    for name, tensor in tensors:
        array = np.asarray(tensor, dtype="<f4", order="C")
        document["tensors"].append({"name": name, "shape": list(array.shape)})
        payload.extend(array.tobytes(order="C"))
    _validate_metadata(document)
    metadata_bytes = canonical_json_bytes(document)
    if len(metadata_bytes) > 0xFFFFFFFF:
        raise ArtifactError("artifact metadata exceeds uint32 length")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(ARTIFACT_MAGIC)
        stream.write(struct.pack("<I", len(metadata_bytes)))
        stream.write(metadata_bytes)
        stream.write(payload)
    temporary.replace(path)
    return sha256_file(path)


def read_artifact(path: Path) -> tuple[dict[str, Any], list[tuple[str, np.ndarray]]]:
    with path.open("rb") as stream:
        if stream.read(len(ARTIFACT_MAGIC)) != ARTIFACT_MAGIC:
            raise ArtifactError("invalid artifact magic")
        raw_length = stream.read(4)
        if len(raw_length) != 4:
            raise ArtifactError("truncated artifact metadata length")
        length = struct.unpack("<I", raw_length)[0]
        raw_metadata = stream.read(length)
        if len(raw_metadata) != length:
            raise ArtifactError("truncated artifact metadata")
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as exception:
            raise ArtifactError(f"invalid artifact metadata JSON: {exception}") from exception
        _validate_metadata(metadata)
        tensors = []
        for descriptor in metadata.get("tensors", []):
            shape = tuple(int(item) for item in descriptor["shape"])
            count = int(np.prod(shape, dtype=np.int64))
            raw = stream.read(count * 4)
            if len(raw) != count * 4:
                raise ArtifactError(f"truncated tensor {descriptor['name']}")
            tensors.append(
                (descriptor["name"], np.frombuffer(raw, dtype="<f4").reshape(shape).copy())
            )
        if stream.read(1):
            raise ArtifactError("unexpected trailing artifact bytes")
    return metadata, tensors
