from pathlib import Path

import numpy as np
import pytest

from alpakatune_ml.artifact import ArtifactError, read_artifact, write_artifact
from alpakatune_ml.features import DIMENSION_FEATURE_NAMES


def _metadata(context_count=2, token_hidden_sizes=(16, 32), embedding_size=32):
    return {
        "artifact_version": 1,
        "feature_schema_version": 1,
        "architecture": "deepsets_ensemble_v1",
        "ensemble_size": 3,
        "context_feature_count": context_count,
        "dimension_feature_count": 18,
        "token_hidden_sizes": list(token_hidden_sizes),
        "embedding_size": embedding_size,
        "device_class_count": 2,
        "feature_names": {"context": ["a", "b"], "dimension": list(DIMENSION_FEATURE_NAMES)},
        "hash_buckets": {"dimension_name": 8},
        "preprocessing": {
            "context_mean": [0.0] * context_count,
            "context_scale": [1.0] * context_count,
            "dimension_mean": [0.0] * 18,
            "dimension_scale": [1.0] * 18,
        },
    }


def _tensors(context_count=2, token_hidden_sizes=(16, 32), embedding_size=32):
    first_hidden, second_hidden = token_hidden_sizes
    shapes = {
        "token.0.weight": (first_hidden, 18),
        "token.0.bias": (first_hidden,),
        "token.2.weight": (second_hidden, first_hidden),
        "token.2.bias": (second_hidden,),
        "context.0.weight": (embedding_size, 2 * second_hidden + context_count),
        "context.0.bias": (embedding_size,),
        "context.2.weight": (embedding_size, embedding_size),
        "context.2.bias": (embedding_size,),
        "adapters.cpu.weight": (1, embedding_size),
        "adapters.cpu.bias": (1,),
        "adapters.gpu.weight": (1, embedding_size),
        "adapters.gpu.bias": (1,),
    }
    return [
        (f"members.{member}.{name}", np.full(shape, member + 0.25, dtype=np.float32))
        for member in range(3)
        for name, shape in shapes.items()
    ]


def test_artifact_round_trip_is_exact(tmp_path):
    path = tmp_path / "model.atml"
    digest = write_artifact(path, _metadata(), _tensors())
    metadata, tensors = read_artifact(path)
    assert len(digest) == 64
    assert path.read_bytes()[:8] == b"ATMLART1"
    assert metadata["feature_names"]["dimension"] == list(DIMENSION_FEATURE_NAMES)
    assert [name for name, _ in tensors] == [name for name, _ in _tensors()]
    assert np.array_equal(tensors[-1][1], _tensors()[-1][1])


def test_artifact_rejects_wrong_tensor_shape(tmp_path):
    tensors = _tensors()
    tensors[0] = (tensors[0][0], np.zeros((15, 18), dtype=np.float32))
    with pytest.raises(ArtifactError, match="tensor inventory"):
        write_artifact(tmp_path / "bad.atml", _metadata(), tensors)
