"""Native-equivalent NumPy evaluation and online residual-adapter simulation."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact import read_artifact
from .dataset import load_split_manifest
from .training import ranking_metrics
from .util import write_json


class NumpyEnsemble:
    def __init__(self, artifact: Path) -> None:
        self.metadata, tensor_list = read_artifact(artifact)
        self.tensors = dict(tensor_list)
        preprocessing = self.metadata["preprocessing"]
        self.context_names = tuple(self.metadata["feature_names"]["context"])
        self.context_mean = np.asarray(preprocessing["context_mean"], dtype=np.float32)
        self.context_scale = np.asarray(preprocessing["context_scale"], dtype=np.float32)
        self.dimension_mean = np.asarray(preprocessing["dimension_mean"], dtype=np.float32)
        self.dimension_scale = np.asarray(preprocessing["dimension_scale"], dtype=np.float32)

    @staticmethod
    def _linear(value, weight, bias):
        return value @ weight.T + bias

    @staticmethod
    def _relu(value):
        return np.maximum(value, 0.0)

    def _member(self, member: int, row: Mapping[str, Any]) -> tuple[float, np.ndarray]:
        prefix = f"members.{member}."
        dimensions = np.asarray(row["dimension_features"], dtype=np.float32)
        dimensions = (dimensions - self.dimension_mean) / self.dimension_scale
        context = np.asarray(
            [float(row["context_features"].get(name, 0.0)) for name in self.context_names],
            dtype=np.float32,
        )
        context = (context - self.context_mean) / self.context_scale
        token = self._relu(
            self._linear(
                dimensions,
                self.tensors[prefix + "token.0.weight"],
                self.tensors[prefix + "token.0.bias"],
            )
        )
        token = self._relu(
            self._linear(
                token,
                self.tensors[prefix + "token.2.weight"],
                self.tensors[prefix + "token.2.bias"],
            )
        )
        pooled = np.concatenate((token.mean(axis=0), token.max(axis=0), context))
        embedding = self._relu(
            self._linear(
                pooled,
                self.tensors[prefix + "context.0.weight"],
                self.tensors[prefix + "context.0.bias"],
            )
        )
        embedding = self._relu(
            self._linear(
                embedding,
                self.tensors[prefix + "context.2.weight"],
                self.tensors[prefix + "context.2.bias"],
            )
        )
        adapter = "gpu" if row["device_class"] == "gpu" else "cpu"
        prediction = self._linear(
            embedding,
            self.tensors[prefix + f"adapters.{adapter}.weight"],
            self.tensors[prefix + f"adapters.{adapter}.bias"],
        )
        return float(np.asarray(prediction).reshape(-1)[0]), embedding

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictions = []
        uncertainties = []
        embeddings = []
        for row in rows:
            member_values = [
                self._member(member, row)
                for member in range(int(self.metadata["ensemble_size"]))
            ]
            values = np.asarray([item[0] for item in member_values])
            predictions.append(values.mean())
            uncertainties.append(values.std())
            embeddings.append(np.mean([item[1] for item in member_values], axis=0))
        return np.asarray(predictions), np.asarray(uncertainties), np.asarray(embeddings)


def _adaptation_metrics(rows, base_predictions, embeddings, budgets: Sequence[int]) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["surface_id"]].append(index)
    results = {}
    for budget in budgets:
        adapted = base_predictions.copy()
        used_surfaces = 0
        for indexes in grouped.values():
            if len(indexes) <= budget:
                continue
            # Row IDs give a deterministic, collection-order-independent observation set.
            observed = sorted(indexes, key=lambda index: rows[index]["row_id"])[:budget]
            matrix = embeddings[observed]
            residual = np.log([rows[index]["runtime_seconds"] for index in observed]) - base_predictions[observed]
            regularization = 1.0e-3
            coefficients = np.linalg.solve(
                matrix.T @ matrix + regularization * np.eye(matrix.shape[1]),
                matrix.T @ residual,
            )
            adapted[indexes] += embeddings[indexes] @ coefficients
            used_surfaces += 1
        if used_surfaces:
            results[str(budget)] = {
                "surface_count": used_surfaces,
                **ranking_metrics(rows, adapted),
            }
    return results


def evaluate(
    artifact: Path,
    split_manifest: Path,
    output: Path,
    adaptation_budgets: Sequence[int] = (16, 32, 64, 128, 256, 512, 1024),
) -> dict[str, Any]:
    manifest, rows = load_split_manifest(split_manifest)
    ensemble = NumpyEnsemble(artifact)
    predictions, uncertainties, embeddings = ensemble.predict(rows)
    result = {
        "schema_version": 1,
        "artifact": str(artifact.resolve()),
        "dataset_id": manifest["dataset_id"],
        "split": manifest["split"],
        "row_count": len(rows),
        "zero_shot": ranking_metrics(rows, predictions),
        "mean_ensemble_uncertainty": float(uncertainties.mean()),
        "online_residual_adapter": _adaptation_metrics(
            rows, predictions, embeddings, adaptation_budgets
        ),
    }
    write_json(output, result)
    return result
