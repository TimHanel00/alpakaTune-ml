"""Versioned history and normalized candidate-label contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

from .features import DIMENSION_FEATURE_NAMES, derive_context_features, dimension_features
from .util import canonical_json_bytes, sha256_file, slug, stable_id


SUPPORTED_HISTORY_SCHEMAS = frozenset({9, 10})
DATASET_SCHEMA_VERSION = 1


class ContractError(ValueError):
    """Raised when collection data cannot satisfy the dataset contract."""


def _unique_values(values):
    result = []
    seen = set()
    for value in values:
        key = canonical_json_bytes(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


@dataclass(frozen=True)
class HistorySurface:
    source: Path
    source_sha256: str
    history_schema_version: int
    fingerprint: str
    cache: Mapping[str, Any]

    @property
    def metadata(self) -> Mapping[str, Any]:
        value = self.cache.get("metadata", {})
        return value if isinstance(value, Mapping) else {}

    @property
    def candidate_count(self) -> int:
        return int(self.cache.get("candidate_count", 0))

    @property
    def device_id(self) -> str:
        structured = self.metadata.get("device_descriptor")
        if isinstance(structured, Mapping) and structured.get("id"):
            return str(structured["id"])
        if self.metadata.get("device_id"):
            return str(self.metadata["device_id"])
        return slug(str(self.metadata.get("device", "unknown-device")))

    @property
    def device_class(self) -> str:
        model_context = self.metadata.get("model_context")
        if isinstance(model_context, Mapping) and model_context.get("device_class") in {"cpu", "gpu"}:
            return str(model_context["device_class"])
        structured = self.metadata.get("device_descriptor")
        if isinstance(structured, Mapping) and structured.get("class") in {"cpu", "gpu"}:
            return str(structured["class"])
        text = " ".join(
            [str(self.metadata.get("device", "")), *map(str, self.metadata.get("identity_entries", ()))]
        ).lower()
        return "gpu" if any(word in text for word in ("gpu", "cuda", "hip", "nvidia", "amd")) else "cpu"

    @property
    def workload_id(self) -> str:
        model_context = self.metadata.get("model_context")
        if isinstance(model_context, Mapping) and model_context.get("workload_id"):
            return str(model_context["workload_id"])
        if self.metadata.get("workload_id"):
            return str(self.metadata["workload_id"])
        identity = {
            "kernel": self.metadata.get("kernel"),
            "identity_entries": self.metadata.get("identity_entries", []),
            "launch_specification": self.metadata.get("launch_specification"),
        }
        return stable_id("workload", identity)

    @property
    def surface_id(self) -> str:
        return stable_id(
            "surface",
            {
                "workload_id": self.workload_id,
                "device_id": self.device_id,
                "context_fingerprint": self.fingerprint,
            },
        )

    def validate_full_exhaustive(self) -> None:
        errors: list[str] = []
        if self.history_schema_version >= 10:
            model_context = self.metadata.get("model_context")
            if not isinstance(model_context, Mapping):
                errors.append("schema-v10 metadata.model_context is missing")
            else:
                if model_context.get("feature_schema_version") != 1:
                    errors.append("model_context.feature_schema_version must be 1")
                if not model_context.get("workload_id"):
                    errors.append("model_context.workload_id is missing")
                if model_context.get("device_class") not in {"cpu", "gpu"}:
                    errors.append("model_context.device_class must be cpu or gpu")
                if not isinstance(model_context.get("context_features"), Mapping):
                    errors.append("model_context.context_features must be an object")
                if not isinstance(model_context.get("dimensions"), list):
                    errors.append("model_context.dimensions must be an array")
        strategy = self.metadata.get("strategy")
        if strategy != "exhaustive":
            errors.append(f"strategy is {strategy!r}, expected 'exhaustive'")
        if self.cache.get("completion_reason") != "all_configurations":
            errors.append(
                f"completion_reason is {self.cache.get('completion_reason')!r}, expected 'all_configurations'"
            )
        if bool(self.cache.get("execution_budget_reached")):
            errors.append("execution_budget_reached is true")
        count = self.candidate_count
        if count <= 0:
            errors.append("candidate_count must be positive")
        arrays = (
            "candidate_configurations",
            "candidate_samples",
            "candidate_estimates",
            "rejected_candidates",
        )
        for name in arrays:
            value = self.cache.get(name)
            if not isinstance(value, list) or len(value) != count:
                errors.append(f"{name} length is not candidate_count ({count})")
        rejected = self.cache.get("rejected_candidates", [])
        samples = self.cache.get("candidate_samples", [])
        estimates = self.cache.get("candidate_estimates", [])
        if isinstance(rejected, list) and isinstance(samples, list) and isinstance(estimates, list):
            measured = 0
            rejected_count = 0
            for index in range(min(count, len(rejected), len(samples), len(estimates))):
                if bool(rejected[index]):
                    rejected_count += 1
                elif (
                    not isinstance(samples[index], list)
                    or len(samples[index]) < 3
                    or estimates[index] is None
                    or not isinstance(estimates[index], (int, float))
                    or not math.isfinite(float(estimates[index]))
                    or float(estimates[index]) <= 0.0
                    or any(
                        not isinstance(sample, (int, float))
                        or not math.isfinite(float(sample))
                        or float(sample) <= 0.0
                        for sample in samples[index]
                    )
                ):
                    errors.append(
                        f"legal candidate {index} lacks three finite positive measurements and estimate"
                    )
                    if len(errors) >= 12:
                        errors.append("additional missing candidates omitted")
                        break
                else:
                    measured += 1
            retired = int(self.cache.get("retired_configuration_count", 0))
            if retired + rejected_count != count:
                errors.append(
                    f"retired ({retired}) + rejected ({rejected_count}) != candidate_count ({count})"
                )
            if measured != count - rejected_count and len(errors) < 12:
                errors.append(f"measured legal candidate count is {measured}, expected {count - rejected_count}")
        if errors:
            raise ContractError(f"{self.source}:{self.fingerprint}: " + "; ".join(errors))

    def rows(self) -> Iterator[dict[str, Any]]:
        configurations = self.cache["candidate_configurations"]
        samples = self.cache["candidate_samples"]
        estimates = self.cache["candidate_estimates"]
        rejected = self.cache["rejected_candidates"]
        model_context = self.metadata.get("model_context", {})
        structured_dimensions = (
            model_context.get("dimensions", []) if isinstance(model_context, Mapping) else []
        )
        if structured_dimensions:
            if not all(isinstance(item, Mapping) and item.get("name") for item in structured_dimensions):
                raise ContractError(f"{self.source}:{self.fingerprint}: invalid model_context.dimensions")
            dimension_names = [str(item["name"]) for item in structured_dimensions]
            descriptors = list(structured_dimensions)
            structured_count = 1
            for descriptor in descriptors:
                cardinality = int(descriptor.get("cardinality", 0))
                if cardinality <= 0:
                    raise ContractError(
                        f"{self.source}:{self.fingerprint}: structured dimension has no candidates"
                    )
                structured_count *= cardinality
            if structured_count != self.candidate_count:
                raise ContractError(
                    f"{self.source}:{self.fingerprint}: structured dimension product "
                    f"{structured_count} != candidate_count {self.candidate_count}"
                )
        else:
            dimension_names = sorted({name for config in configurations for name in config})
            legacy = self.metadata.get("tunable_descriptors", {})
            descriptors = legacy if isinstance(legacy, Mapping) else {}
            domains = {
                name: _unique_values(config.get(name) for config in configurations)
                for name in dimension_names
            }
        context_features = derive_context_features(
            self.metadata, self.candidate_count, len(dimension_names)
        )
        for index, config in enumerate(configurations):
            if rejected[index]:
                continue
            tokens = []
            if structured_dimensions:
                remaining = index
                ordinals = [0] * len(descriptors)
                for position in range(len(descriptors) - 1, -1, -1):
                    cardinality = int(descriptors[position]["cardinality"])
                    ordinals[position] = remaining % cardinality
                    remaining //= cardinality
                dimension_items = []
                for position, descriptor in enumerate(descriptors):
                    cardinality = int(descriptor["cardinality"])
                    concrete_values = descriptor.get("concrete_values")
                    domain = (
                        list(concrete_values)
                        if isinstance(concrete_values, list)
                        and len(concrete_values) == cardinality
                        else list(range(cardinality))
                    )
                    dimension_items.append(
                        (str(descriptor["name"]), descriptor, domain, domain[ordinals[position]])
                    )
            else:
                dimension_items = [
                    (
                        name,
                        descriptors.get(name, {}) if isinstance(descriptors, Mapping) else {},
                        domains[name],
                        config.get(name),
                    )
                    for name in dimension_names
                ]
            for position, (name, descriptor, domain, value) in enumerate(dimension_items):
                tokens.append(
                    dimension_features(
                        name=name,
                        value=value,
                        domain=domain,
                        dimension_position=position,
                        dimension_count=len(dimension_names),
                        kind=str(descriptor.get("kind", "runtime")),
                        component_position=int(
                            descriptor.get("component_index", descriptor.get("component_position", 0))
                        ),
                        arity=int(descriptor.get("vector_arity", descriptor.get("arity", 1))),
                    )
                )
            row_id = stable_id(
                "row",
                {"surface_id": self.surface_id, "candidate_index": index},
            )
            yield {
                "schema_version": DATASET_SCHEMA_VERSION,
                "row_id": row_id,
                "surface_id": self.surface_id,
                "workload_id": self.workload_id,
                "device_id": self.device_id,
                "device_class": self.device_class,
                "candidate_index": index,
                "configuration": config,
                "dimension_feature_names": list(DIMENSION_FEATURE_NAMES),
                "dimension_features": tokens,
                "context_features": context_features,
                "runtime_seconds": float(estimates[index]),
                "samples_seconds": [float(item) for item in samples[index]],
                "source": {
                    "history_sha256": self.source_sha256,
                    "history_schema_version": self.history_schema_version,
                    "context_fingerprint": self.fingerprint,
                },
            }


def load_history(path: Path) -> list[HistorySurface]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise ContractError(f"cannot read history {path}: {exception}") from exception
    schema = document.get("schema_version")
    if schema not in SUPPORTED_HISTORY_SCHEMAS:
        raise ContractError(
            f"{path}: unsupported history schema {schema!r}; supported: {sorted(SUPPORTED_HISTORY_SCHEMAS)}"
        )
    contexts = document.get("contexts")
    if not isinstance(contexts, Mapping) or not contexts:
        raise ContractError(f"{path}: contexts must be a non-empty object")
    digest = sha256_file(path)
    return [
        HistorySurface(path, digest, int(schema), str(fingerprint), cache)
        for fingerprint, cache in contexts.items()
        if isinstance(cache, Mapping)
    ]


def discover_histories(inputs: list[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for value in inputs:
        if value.is_file():
            discovered.add(value.resolve())
        elif value.is_dir():
            discovered.update(path.resolve() for path in value.rglob("history.json"))
        else:
            raise ContractError(f"history input does not exist: {value}")
    if not discovered:
        raise ContractError("no history.json files found")
    return sorted(discovered)
