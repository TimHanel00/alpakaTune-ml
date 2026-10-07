"""Train the three-member contextual ranker from random initialization."""

from __future__ import annotations

from collections import defaultdict
import datetime as dt
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .artifact import (
    ARCHITECTURE,
    ARTIFACT_VERSION,
    FEATURE_SCHEMA_VERSION,
    read_artifact,
    write_artifact,
)
from .dataset import validate_split_set
from .features import DIMENSION_FEATURE_NAMES, NAME_HASH_BUCKETS
from .model import build_ranker, require_torch
from .util import sha256_file, write_json


TRAINING_SCHEMA_VERSION = 1


class TrainingError(ValueError):
    pass


def load_training_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exception:
        raise TrainingError(f"cannot load training config {path}: {exception}") from exception
    if not isinstance(value, Mapping) or value.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise TrainingError(f"{path}: schema_version must be {TRAINING_SCHEMA_VERSION}")
    result = {
        "seed": int(value.get("seed", 1729)),
        "epochs": int(value.get("epochs", 40)),
        "pairs_per_epoch": int(value.get("pairs_per_epoch", 4096)),
        "batch_size": int(value.get("batch_size", 128)),
        "learning_rate": float(value.get("learning_rate", 1.0e-3)),
        "ranking_weight": float(value.get("ranking_weight", 1.0)),
        "runtime_weight": float(value.get("runtime_weight", 0.1)),
        "fast_decile_weight": float(value.get("fast_decile_weight", 4.0)),
        "device": str(value.get("device", "cpu")),
        "token_hidden_sizes": tuple(value.get("token_hidden_sizes", (16, 32))),
        "embedding_size": int(value.get("embedding_size", 32)),
        "ensemble_size": int(value.get("ensemble_size", 3)),
    }
    if min(result["epochs"], result["pairs_per_epoch"], result["batch_size"]) <= 0:
        raise TrainingError("epochs, pairs_per_epoch, and batch_size must be positive")
    if (
        len(result["token_hidden_sizes"]) != 2
        or not all(isinstance(item, int) and item > 0 for item in result["token_hidden_sizes"])
        or result["embedding_size"] <= 0
        or result["ensemble_size"] not in {1, 3}
    ):
        raise TrainingError(
            "token_hidden_sizes needs two positive integers; embedding_size must be positive; "
            "ensemble_size must be 1 or 3"
        )
    return result


class Preprocessor:
    def __init__(self, context_names: Sequence[str], context_mean, context_scale, dimension_mean, dimension_scale):
        self.context_names = tuple(context_names)
        self.context_mean = np.asarray(context_mean, dtype=np.float32)
        self.context_scale = np.asarray(context_scale, dtype=np.float32)
        self.dimension_mean = np.asarray(dimension_mean, dtype=np.float32)
        self.dimension_scale = np.asarray(dimension_scale, dtype=np.float32)

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, Any]]) -> "Preprocessor":
        context_names = sorted({name for row in rows for name in row["context_features"]})
        if not context_names:
            raise TrainingError("training rows have no context features")
        contexts = np.asarray(
            [[float(row["context_features"].get(name, 0.0)) for name in context_names] for row in rows],
            dtype=np.float64,
        )
        dimensions = np.asarray(
            [token for row in rows for token in row["dimension_features"]], dtype=np.float64
        )
        if dimensions.ndim != 2 or dimensions.shape[1] != len(DIMENSION_FEATURE_NAMES):
            raise TrainingError("dimension feature width does not match schema v1")
        context_scale = contexts.std(axis=0)
        dimension_scale = dimensions.std(axis=0)
        context_scale[context_scale < 1.0e-6] = 1.0
        dimension_scale[dimension_scale < 1.0e-6] = 1.0
        return cls(
            context_names,
            contexts.mean(axis=0),
            context_scale,
            dimensions.mean(axis=0),
            dimension_scale,
        )

    def encode(self, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, int, float]:
        context = np.asarray(
            [float(row["context_features"].get(name, 0.0)) for name in self.context_names],
            dtype=np.float32,
        )
        dimensions = np.asarray(row["dimension_features"], dtype=np.float32)
        context = (context - self.context_mean) / self.context_scale
        dimensions = (dimensions - self.dimension_mean) / self.dimension_scale
        device_class = 1 if row["device_class"] == "gpu" else 0
        return dimensions, context, device_class, math.log(float(row["runtime_seconds"]))

    def metadata(self) -> dict[str, Any]:
        return {
            "context_mean": self.context_mean.tolist(),
            "context_scale": self.context_scale.tolist(),
            "dimension_mean": self.dimension_mean.tolist(),
            "dimension_scale": self.dimension_scale.tolist(),
        }


def _collate(rows: Sequence[Mapping[str, Any]], preprocessor: Preprocessor, device: str):
    torch = require_torch()
    encoded = [preprocessor.encode(row) for row in rows]
    max_dimensions = max(item[0].shape[0] for item in encoded)
    dimension_batch = np.zeros((len(rows), max_dimensions, 18), dtype=np.float32)
    mask = np.zeros((len(rows), max_dimensions), dtype=np.float32)
    contexts = np.zeros((len(rows), len(preprocessor.context_names)), dtype=np.float32)
    classes = np.zeros(len(rows), dtype=np.int64)
    targets = np.zeros(len(rows), dtype=np.float32)
    for index, (dimensions, context, device_class, target) in enumerate(encoded):
        dimension_batch[index, : len(dimensions)] = dimensions
        mask[index, : len(dimensions)] = 1.0
        contexts[index] = context
        classes[index] = device_class
        targets[index] = target
    return (
        torch.from_numpy(dimension_batch).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(contexts).to(device),
        torch.from_numpy(classes).to(device),
        torch.from_numpy(targets).to(device),
    )


def _surface_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["surface_id"])].append(row)
    result = [values for values in groups.values() if len(values) >= 2]
    if not result:
        raise TrainingError("training needs at least one surface with two candidates")
    return result


def _sample_pairs(groups, count: int, rng: random.Random, fast_decile_weight: float):
    faster = []
    slower = []
    weights = []
    thresholds = {
        id(group): float(np.quantile([row["runtime_seconds"] for row in group], 0.1))
        for group in groups
    }
    for _ in range(count):
        group = rng.choice(groups)
        first, second = rng.sample(group, 2)
        if first["runtime_seconds"] <= second["runtime_seconds"]:
            fast, slow = first, second
        else:
            fast, slow = second, first
        faster.append(fast)
        slower.append(slow)
        weights.append(
            1.0 if fast["runtime_seconds"] > thresholds[id(group)] else fast_decile_weight
        )
    return faster, slower, np.asarray(weights, dtype=np.float32)


def _predict(model, rows, preprocessor: Preprocessor, device: str, batch_size: int = 1024):
    torch = require_torch()
    predictions = []
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            dimensions, mask, contexts, classes, _targets = _collate(
                rows[start : start + batch_size], preprocessor, device
            )
            prediction, embedding = model(dimensions, mask, contexts, classes)
            predictions.extend(prediction.cpu().numpy().tolist())
            embeddings.extend(embedding.cpu().numpy().tolist())
    return np.asarray(predictions), np.asarray(embeddings)


def ranking_metrics(rows, predictions) -> dict[str, float]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        grouped[row["surface_id"]].append((float(row["runtime_seconds"]), float(prediction)))
    regrets = {1: [], 5: [], 10: []}
    for values in grouped.values():
        oracle = min(runtime for runtime, _ in values)
        ranked = sorted(values, key=lambda item: item[1])
        for k in regrets:
            selected = min(runtime for runtime, _ in ranked[:k])
            regrets[k].append(selected / oracle - 1.0)
    targets = np.log([float(row["runtime_seconds"]) for row in rows])
    return {
        "log_runtime_mae": float(np.mean(np.abs(targets - predictions))),
        **{f"top_{k}_median_regret": float(np.median(values)) for k, values in regrets.items()},
    }


def _artifact_tensors(models) -> list[tuple[str, np.ndarray]]:
    names = (
        ("token.0.weight", "token.0.weight"),
        ("token.0.bias", "token.0.bias"),
        ("token.2.weight", "token.2.weight"),
        ("token.2.bias", "token.2.bias"),
        ("context.0.weight", "context.0.weight"),
        ("context.0.bias", "context.0.bias"),
        ("context.2.weight", "context.2.weight"),
        ("context.2.bias", "context.2.bias"),
        ("adapters.cpu.weight", "cpu_adapter.weight"),
        ("adapters.cpu.bias", "cpu_adapter.bias"),
        ("adapters.gpu.weight", "gpu_adapter.weight"),
        ("adapters.gpu.bias", "gpu_adapter.bias"),
    )
    tensors = []
    for index, model in enumerate(models):
        state = model.state_dict()
        for artifact_name, state_name in names:
            tensors.append(
                (
                    f"members.{index}.{artifact_name}",
                    state[state_name].detach().cpu().numpy(),
                )
            )
    return tensors


def _train_one_member(
    torch,
    config: Mapping[str, Any],
    member: int,
    splits: Mapping[str, list[Mapping[str, Any]]],
    preprocessor: Preprocessor,
    groups,
):
    seed = int(config["seed"]) + member
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    device = str(config["device"])
    model = build_ranker(
        len(preprocessor.context_names),
        config["token_hidden_sizes"],
        config["embedding_size"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_state = None
    best_regret = math.inf
    for _epoch in range(config["epochs"]):
        fast, slow, weights = _sample_pairs(
            groups, config["pairs_per_epoch"], rng, config["fast_decile_weight"]
        )
        order = list(range(len(fast)))
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), config["batch_size"]):
            indexes = order[start : start + config["batch_size"]]
            fast_rows = [fast[index] for index in indexes]
            slow_rows = [slow[index] for index in indexes]
            fast_batch = _collate(fast_rows, preprocessor, device)
            slow_batch = _collate(slow_rows, preprocessor, device)
            fast_prediction, _ = model(*fast_batch[:4])
            slow_prediction, _ = model(*slow_batch[:4])
            batch_weights = torch.from_numpy(weights[indexes]).to(device)
            ranking = (
                torch.nn.functional.softplus(fast_prediction - slow_prediction)
                * batch_weights
            ).mean()
            runtime = 0.5 * (
                torch.nn.functional.huber_loss(fast_prediction, fast_batch[4])
                + torch.nn.functional.huber_loss(slow_prediction, slow_batch[4])
            )
            loss = config["ranking_weight"] * ranking + config["runtime_weight"] * runtime
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation_predictions, _ = _predict(
            model, splits["validation"], preprocessor, device
        )
        validation = ranking_metrics(splits["validation"], validation_predictions)
        if validation["top_10_median_regret"] < best_regret:
            best_regret = validation["top_10_median_regret"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise TrainingError("member training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    validation_predictions, _ = _predict(
        model, splits["validation"], preprocessor, device
    )
    return model, ranking_metrics(splits["validation"], validation_predictions)


def _split_identity(split_manifests: Sequence[Path]) -> tuple[str, dict[str, str]]:
    dataset_ids = set()
    checksums: dict[str, str] = {}
    for path in split_manifests:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            raise TrainingError(f"cannot inspect split manifest {path}: {exception}") from exception
        split = str(manifest.get("split", ""))
        dataset_id = str(manifest.get("dataset_id", ""))
        if split not in {"train", "validation", "test"} or split in checksums:
            raise TrainingError("distributed training needs one manifest per split")
        if not dataset_id:
            raise TrainingError(f"split manifest has no dataset_id: {path}")
        checksums[split] = sha256_file(path)
        dataset_ids.add(dataset_id)
    if len(dataset_ids) != 1 or set(checksums) != {"train", "validation", "test"}:
        raise TrainingError("distributed training split manifests do not form one dataset")
    return dataset_ids.pop(), checksums


def _inference_profile(
    config: Mapping[str, Any], context_count: int, parameter_count: int
) -> dict[str, Any]:
    first_hidden, second_hidden = config["token_hidden_sizes"]
    embedding_size = config["embedding_size"]
    return {
        "parameter_count": int(parameter_count),
        "tensor_payload_bytes": int(parameter_count * 4),
        "multiply_adds_per_dimension_per_member": 18 * first_hidden
        + first_hidden * second_hidden,
        "fixed_multiply_adds_per_member": (
            (2 * second_hidden + context_count) * embedding_size
            + embedding_size * embedding_size
            + embedding_size
        ),
        "candidate_latency_target_microseconds": [1.0, 5.0],
        "cached_recommend_target_microseconds": 10.0,
    }


def _model_metadata(
    config: Mapping[str, Any],
    preprocessor: Preprocessor,
    ensemble_size: int,
    parameter_count: int,
    training_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "ensemble_size": ensemble_size,
        "context_feature_count": len(preprocessor.context_names),
        "dimension_feature_count": len(DIMENSION_FEATURE_NAMES),
        "token_hidden_sizes": list(config["token_hidden_sizes"]),
        "embedding_size": config["embedding_size"],
        "device_class_count": 2,
        "feature_names": {
            "context": list(preprocessor.context_names),
            "dimension": list(DIMENSION_FEATURE_NAMES),
        },
        "hash_buckets": {"dimension_name": NAME_HASH_BUCKETS},
        "preprocessing": preprocessor.metadata(),
        "inference_profile": _inference_profile(
            config, len(preprocessor.context_names), parameter_count
        ),
        "training": dict(training_metadata),
    }


def _write_artifact_outputs(
    output_artifact: Path,
    metadata: Mapping[str, Any],
    tensors: Sequence[tuple[str, np.ndarray]],
    card: Mapping[str, Any],
) -> dict[str, Any]:
    digest = write_artifact(output_artifact, metadata, tensors)
    output_artifact.with_suffix(output_artifact.suffix + ".sha256").write_text(
        f"{digest}  {output_artifact.name}\n", encoding="utf-8"
    )
    result = {**card, "artifact": output_artifact.name, "sha256": digest}
    write_json(output_artifact.with_suffix(".model-card.json"), result)
    return result


def train_member(
    split_manifests: list[Path],
    config_path: Path,
    member_index: int,
    output_artifact: Path,
) -> dict[str, Any]:
    """Train one deterministic ensemble member without evaluating the test split."""
    torch = require_torch()
    config = load_training_config(config_path)
    if member_index < 0 or member_index >= config["ensemble_size"]:
        raise TrainingError(
            f"member_index must be in [0, {config['ensemble_size'] - 1}]"
        )
    splits = validate_split_set(split_manifests)
    dataset_id, split_checksums = _split_identity(split_manifests)
    preprocessor = Preprocessor.fit(splits["train"])
    groups = _surface_groups(splits["train"])
    device = config["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise TrainingError(f"training device {device!r} requested but CUDA is unavailable")
    # Deliberately withhold the test rows from the member-selection routine.
    selection_splits = {"validation": splits["validation"]}
    model, validation_metrics = _train_one_member(
        torch, config, member_index, selection_splits, preprocessor, groups
    )
    tensors = _artifact_tensors([model.cpu()])
    parameter_count = sum(tensor.size for _name, tensor in tensors)
    seed = config["seed"] + member_index
    metadata = _model_metadata(
        config,
        preprocessor,
        1,
        parameter_count,
        {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": "distributed_member",
            "config_sha256": sha256_file(config_path),
            "dataset_id": dataset_id,
            "split_manifest_sha256": split_checksums,
            "member_index": member_index,
            "requested_ensemble_size": config["ensemble_size"],
            "seed": seed,
            "validation_metrics": validation_metrics,
        },
    )
    return _write_artifact_outputs(
        output_artifact,
        metadata,
        tensors,
        {
            "kind": "distributed_member",
            "dataset_id": dataset_id,
            "member_index": member_index,
            "requested_ensemble_size": config["ensemble_size"],
            "seed": seed,
            "validation_metrics": validation_metrics,
        },
    )


_MERGE_CONTRACT_FIELDS = (
    "artifact_version",
    "feature_schema_version",
    "architecture",
    "context_feature_count",
    "dimension_feature_count",
    "token_hidden_sizes",
    "embedding_size",
    "device_class_count",
    "feature_names",
    "hash_buckets",
    "preprocessing",
)


def merge_member_artifacts(
    member_artifacts: list[Path],
    split_manifests: list[Path],
    config_path: Path,
    output_artifact: Path,
) -> dict[str, Any]:
    """Validate independent members and emit the unchanged ensemble contract."""
    config = load_training_config(config_path)
    splits = validate_split_set(split_manifests)
    dataset_id, split_checksums = _split_identity(split_manifests)
    if len(member_artifacts) != config["ensemble_size"]:
        raise TrainingError(
            f"expected {config['ensemble_size']} member artifacts, got {len(member_artifacts)}"
        )
    expected_config = sha256_file(config_path)
    output_resolved = output_artifact.resolve()
    loaded: dict[int, tuple[Path, dict[str, Any], list[tuple[str, np.ndarray]]]] = {}
    reference = None
    for path in member_artifacts:
        if path.resolve() == output_resolved:
            raise TrainingError("the merged output must not overwrite a member artifact")
        metadata, tensors = read_artifact(path)
        training = metadata.get("training", {})
        if metadata.get("ensemble_size") != 1 or training.get("mode") != "distributed_member":
            raise TrainingError(f"not a distributed single-member artifact: {path}")
        if (
            training.get("config_sha256") != expected_config
            or training.get("dataset_id") != dataset_id
            or training.get("split_manifest_sha256") != split_checksums
            or training.get("requested_ensemble_size") != config["ensemble_size"]
        ):
            raise TrainingError(f"member provenance does not match merge inputs: {path}")
        member_index = int(training.get("member_index", -1))
        if member_index < 0 or member_index >= config["ensemble_size"] or member_index in loaded:
            raise TrainingError(f"duplicate or invalid member_index {member_index}: {path}")
        if training.get("seed") != config["seed"] + member_index:
            raise TrainingError(f"member seed does not match its deterministic index: {path}")
        contract = {name: metadata.get(name) for name in _MERGE_CONTRACT_FIELDS}
        if reference is None:
            reference = contract
        elif contract != reference:
            raise TrainingError(f"member feature/model contract differs: {path}")
        loaded[member_index] = (path, metadata, tensors)
    expected_indexes = set(range(config["ensemble_size"]))
    if set(loaded) != expected_indexes:
        raise TrainingError(
            f"member indexes must be exactly {sorted(expected_indexes)}"
        )
    if (
        reference["token_hidden_sizes"] != list(config["token_hidden_sizes"])
        or reference["embedding_size"] != config["embedding_size"]
    ):
        raise TrainingError("member model dimensions do not match the training config")

    merged_tensors = []
    member_validation_metrics = []
    member_sources = []
    member_predictions = []
    # Import lazily because evaluation imports ranking_metrics from this module.
    from .evaluation import NumpyEnsemble

    for member_index in sorted(loaded):
        path, metadata, tensors = loaded[member_index]
        for name, tensor in tensors:
            prefix = "members.0."
            if not name.startswith(prefix):
                raise TrainingError(f"single-member tensor has an invalid name: {name}")
            merged_tensors.append(
                (f"members.{member_index}.{name[len(prefix):]}", tensor)
            )
        training = metadata["training"]
        member_validation_metrics.append(training["validation_metrics"])
        member_sources.append(
            {
                "member_index": member_index,
                "seed": training["seed"],
                "sha256": sha256_file(path),
                "artifact": path.name,
            }
        )
        member_predictions.append(NumpyEnsemble(path).predict(splits["test"])[0])

    member_test_metrics = [
        ranking_metrics(splits["test"], predictions)
        for predictions in member_predictions
    ]
    ensemble_metrics = ranking_metrics(
        splits["test"], np.mean(member_predictions, axis=0)
    )
    preprocessor = Preprocessor(
        reference["feature_names"]["context"],
        reference["preprocessing"]["context_mean"],
        reference["preprocessing"]["context_scale"],
        reference["preprocessing"]["dimension_mean"],
        reference["preprocessing"]["dimension_scale"],
    )
    parameter_count = sum(tensor.size for _name, tensor in merged_tensors)
    metadata = _model_metadata(
        config,
        preprocessor,
        config["ensemble_size"],
        parameter_count,
        {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": "distributed_ensemble_merge",
            "config_sha256": expected_config,
            "dataset_id": dataset_id,
            "split_manifest_sha256": split_checksums,
            "seeds": [source["seed"] for source in member_sources],
            "member_artifacts": member_sources,
            "member_validation_metrics": member_validation_metrics,
            "member_test_metrics": member_test_metrics,
            "ensemble_test_metrics": ensemble_metrics,
        },
    )
    return _write_artifact_outputs(
        output_artifact,
        metadata,
        merged_tensors,
        {
            "dataset_id": dataset_id,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "architecture": ARCHITECTURE,
            "member_validation_metrics": member_validation_metrics,
            "member_test_metrics": member_test_metrics,
            "ensemble_test_metrics": ensemble_metrics,
            "intended_use": "Rank candidates for known alpakaTune kernel families on unseen devices.",
            "limitations": [
                "No unseen-kernel generalization claim.",
                "Use only on device families represented by the approved training manifest.",
            ],
        },
    )


def train(
    split_manifests: list[Path],
    config_path: Path,
    output_artifact: Path,
) -> dict[str, Any]:
    torch = require_torch()
    config = load_training_config(config_path)
    splits = validate_split_set(split_manifests)
    preprocessor = Preprocessor.fit(splits["train"])
    groups = _surface_groups(splits["train"])
    device = config["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise TrainingError(f"training device {device!r} requested but CUDA is unavailable")
    models = []
    for member in range(config["ensemble_size"]):
        seed = config["seed"] + member
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = random.Random(seed)
        model = build_ranker(
            len(preprocessor.context_names),
            config["token_hidden_sizes"],
            config["embedding_size"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        best_state = None
        best_regret = math.inf
        for _epoch in range(config["epochs"]):
            fast, slow, weights = _sample_pairs(
                groups, config["pairs_per_epoch"], rng, config["fast_decile_weight"]
            )
            order = list(range(len(fast)))
            rng.shuffle(order)
            model.train()
            for start in range(0, len(order), config["batch_size"]):
                indexes = order[start : start + config["batch_size"]]
                fast_rows = [fast[index] for index in indexes]
                slow_rows = [slow[index] for index in indexes]
                fast_batch = _collate(fast_rows, preprocessor, device)
                slow_batch = _collate(slow_rows, preprocessor, device)
                fast_prediction, _ = model(*fast_batch[:4])
                slow_prediction, _ = model(*slow_batch[:4])
                batch_weights = torch.from_numpy(weights[indexes]).to(device)
                ranking = (
                    torch.nn.functional.softplus(fast_prediction - slow_prediction) * batch_weights
                ).mean()
                runtime = 0.5 * (
                    torch.nn.functional.huber_loss(fast_prediction, fast_batch[4])
                    + torch.nn.functional.huber_loss(slow_prediction, slow_batch[4])
                )
                loss = config["ranking_weight"] * ranking + config["runtime_weight"] * runtime
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            validation_predictions, _ = _predict(
                model, splits["validation"], preprocessor, device
            )
            validation = ranking_metrics(splits["validation"], validation_predictions)
            if validation["top_10_median_regret"] < best_regret:
                best_regret = validation["top_10_median_regret"]
                best_state = {
                    name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
                }
        assert best_state is not None
        model.load_state_dict(best_state)
        models.append(model)

    # The untouched test split is evaluated only after every ensemble member
    # has completed validation-based epoch selection.
    member_predictions = [
        _predict(model, splits["test"], preprocessor, device)[0] for model in models
    ]
    member_metrics = [
        ranking_metrics(splits["test"], predictions)
        for predictions in member_predictions
    ]
    ensemble_predictions = np.mean(member_predictions, axis=0)
    ensemble_metrics = ranking_metrics(splits["test"], ensemble_predictions)
    models = [model.cpu() for model in models]

    artifact_tensors = _artifact_tensors(models)
    parameter_count = sum(tensor.size for _name, tensor in artifact_tensors)
    first_hidden, second_hidden = config["token_hidden_sizes"]
    embedding_size = config["embedding_size"]
    per_dimension_madds = 18 * first_hidden + first_hidden * second_hidden
    fixed_madds = (
        (2 * second_hidden + len(preprocessor.context_names)) * embedding_size
        + embedding_size * embedding_size
        + embedding_size
    )
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "ensemble_size": config["ensemble_size"],
        "context_feature_count": len(preprocessor.context_names),
        "dimension_feature_count": len(DIMENSION_FEATURE_NAMES),
        "token_hidden_sizes": list(config["token_hidden_sizes"]),
        "embedding_size": config["embedding_size"],
        "device_class_count": 2,
        "feature_names": {
            "context": list(preprocessor.context_names),
            "dimension": list(DIMENSION_FEATURE_NAMES),
        },
        "hash_buckets": {"dimension_name": NAME_HASH_BUCKETS},
        "preprocessing": preprocessor.metadata(),
        "inference_profile": {
            "parameter_count": int(parameter_count),
            "tensor_payload_bytes": int(parameter_count * 4),
            "multiply_adds_per_dimension_per_member": per_dimension_madds,
            "fixed_multiply_adds_per_member": fixed_madds,
            "candidate_latency_target_microseconds": [1.0, 5.0],
            "cached_recommend_target_microseconds": 10.0,
        },
        "training": {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_sha256": sha256_file(config_path),
            "dataset_id": json.loads(split_manifests[0].read_text(encoding="utf-8"))["dataset_id"],
            "seeds": [config["seed"] + member for member in range(config["ensemble_size"])],
            "member_test_metrics": member_metrics,
            "ensemble_test_metrics": ensemble_metrics,
        },
    }
    digest = write_artifact(output_artifact, metadata, artifact_tensors)
    checksum_path = output_artifact.with_suffix(output_artifact.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {output_artifact.name}\n", encoding="utf-8")
    card = {
        "artifact": output_artifact.name,
        "sha256": digest,
        "dataset_id": metadata["training"]["dataset_id"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "member_test_metrics": member_metrics,
        "ensemble_test_metrics": ensemble_metrics,
        "intended_use": "Rank candidates for known alpakaTune kernel families on unseen devices.",
        "limitations": [
            "No unseen-kernel generalization claim.",
            "Use only on device families represented by the approved training manifest.",
        ],
    }
    write_json(output_artifact.with_suffix(".model-card.json"), card)
    return card
