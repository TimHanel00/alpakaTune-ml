import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from alpakatune_ml.artifact import read_artifact, write_artifact
from alpakatune_ml.features import DIMENSION_FEATURE_NAMES
from alpakatune_ml import training
from alpakatune_ml.training import TrainingError, merge_member_artifacts
from alpakatune_ml.util import sha256_file


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "training.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "seed": 100,
                "epochs": 1,
                "pairs_per_epoch": 2,
                "batch_size": 2,
                "device": "cpu",
                "token_hidden_sizes": [2, 3],
                "embedding_size": 2,
                "ensemble_size": 3,
            }
        )
    )
    return path


def _manifests(tmp_path: Path):
    result = []
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.manifest.json"
        path.write_text(json.dumps({"split": split, "dataset_id": "dataset-1"}))
        result.append(path)
    return result


def _rows():
    rows = []
    for index, runtime in enumerate((1.0, 2.0)):
        token = [0.0] * 18
        token[0] = float(index)
        rows.append(
            {
                "row_id": f"row-{index}",
                "surface_id": "surface-1",
                "device_class": "gpu",
                "runtime_seconds": runtime,
                "context_features": {"a": 0.0, "b": 0.0},
                "dimension_features": [token],
            }
        )
    return rows


def _member_tensors(bias: float):
    shapes = {
        "token.0.weight": (2, 18),
        "token.0.bias": (2,),
        "token.2.weight": (3, 2),
        "token.2.bias": (3,),
        "context.0.weight": (2, 8),
        "context.0.bias": (2,),
        "context.2.weight": (2, 2),
        "context.2.bias": (2,),
        "adapters.cpu.weight": (1, 2),
        "adapters.cpu.bias": (1,),
        "adapters.gpu.weight": (1, 2),
        "adapters.gpu.bias": (1,),
    }
    result = []
    for name, shape in shapes.items():
        value = np.zeros(shape, dtype=np.float32)
        if name.endswith("adapters.gpu.bias"):
            value.fill(bias)
        result.append((f"members.0.{name}", value))
    return result


def _write_members(tmp_path, config, manifests, *, mismatched_member=None):
    split_checksums = {
        path.stem.split(".")[0]: sha256_file(path) for path in manifests
    }
    result = []
    for member in range(3):
        context_mean = [0.0, 0.0]
        if member == mismatched_member:
            context_mean[0] = 1.0
        metadata = {
            "artifact_version": 1,
            "feature_schema_version": 1,
            "architecture": "deepsets_ensemble_v1",
            "ensemble_size": 1,
            "context_feature_count": 2,
            "dimension_feature_count": 18,
            "token_hidden_sizes": [2, 3],
            "embedding_size": 2,
            "device_class_count": 2,
            "feature_names": {
                "context": ["a", "b"],
                "dimension": list(DIMENSION_FEATURE_NAMES),
            },
            "hash_buckets": {"dimension_name": 8},
            "preprocessing": {
                "context_mean": context_mean,
                "context_scale": [1.0, 1.0],
                "dimension_mean": [0.0] * 18,
                "dimension_scale": [1.0] * 18,
            },
            "training": {
                "mode": "distributed_member",
                "config_sha256": sha256_file(config),
                "dataset_id": "dataset-1",
                "split_manifest_sha256": split_checksums,
                "member_index": member,
                "requested_ensemble_size": 3,
                "seed": 100 + member,
                "validation_metrics": {"top_10_median_regret": 0.0},
            },
        }
        path = tmp_path / f"member-{member}.atml"
        write_artifact(path, metadata, _member_tensors(float(member)))
        result.append(path)
    return result


def test_merge_members_preserves_artifact_contract_without_torch(tmp_path, monkeypatch):
    config = _config(tmp_path)
    manifests = _manifests(tmp_path)
    rows = _rows()
    monkeypatch.setattr(
        training,
        "validate_split_set",
        lambda _paths: {"train": rows, "validation": rows, "test": rows},
    )
    members = _write_members(tmp_path, config, manifests)
    output = tmp_path / "ensemble.atml"

    card = merge_member_artifacts(members, manifests, config, output)
    metadata, tensors = read_artifact(output)

    assert metadata["ensemble_size"] == 3
    assert metadata["training"]["mode"] == "distributed_ensemble_merge"
    assert metadata["training"]["seeds"] == [100, 101, 102]
    assert [name for name, _tensor in tensors if name.endswith("adapters.gpu.bias")] == [
        f"members.{member}.adapters.gpu.bias" for member in range(3)
    ]
    assert card["dataset_id"] == "dataset-1"
    assert output.with_suffix(output.suffix + ".sha256").exists()


def test_merge_rejects_member_preprocessing_mismatch(tmp_path, monkeypatch):
    config = _config(tmp_path)
    manifests = _manifests(tmp_path)
    rows = _rows()
    monkeypatch.setattr(
        training,
        "validate_split_set",
        lambda _paths: {"train": rows, "validation": rows, "test": rows},
    )
    members = _write_members(
        tmp_path, config, manifests, mismatched_member=2
    )

    with pytest.raises(TrainingError, match="contract differs"):
        merge_member_artifacts(
            members, manifests, config, tmp_path / "ensemble.atml"
        )
