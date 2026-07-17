"""Feature schema shared by dataset preparation, training, and native inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any


FEATURE_SCHEMA_VERSION = 1
NAME_HASH_BUCKETS = 8

DIMENSION_FEATURE_NAMES = (
    "coordinate",
    "concrete_value",
    "cardinality_log1p",
    "dimension_position",
    "component_position",
    "arity_log1p",
    "kind_runtime",
    "kind_compile_time",
    "kind_launch",
    "kind_categorical",
    *(f"name_hash_{index}" for index in range(NAME_HASH_BUCKETS)),
)


def fnv1a64(value: str) -> int:
    result = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return result


def signed_hash_features(value: str, buckets: int = NAME_HASH_BUCKETS) -> list[float]:
    result = [0.0] * buckets
    hashed = fnv1a64(value)
    result[hashed % buckets] = -1.0 if (hashed >> 63) else 1.0
    return result


def _numeric(value: Any, ordinal: int) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(ordinal)


def dimension_features(
    *,
    name: str,
    value: Any,
    domain: Sequence[Any],
    dimension_position: int,
    dimension_count: int,
    kind: str = "runtime",
    component_position: int = 0,
    arity: int = 1,
) -> list[float]:
    """Encode one tuning dimension using the native schema-v1 order."""
    try:
        ordinal = list(domain).index(value)
    except ValueError:
        ordinal = 0
    coordinate = ordinal / max(1, len(domain) - 1)
    numeric_domain = [_numeric(item, index) for index, item in enumerate(domain)]
    low = min(numeric_domain, default=0.0)
    high = max(numeric_domain, default=0.0)
    concrete = (_numeric(value, ordinal) - low) / (high - low) if high > low else 0.0
    kinds = {
        "runtime": (1.0, 0.0, 0.0, 0.0),
        "compile_time": (0.0, 1.0, 0.0, 0.0),
        "launch": (0.0, 0.0, 1.0, 0.0),
        "categorical": (0.0, 0.0, 0.0, 1.0),
    }
    kind_features = kinds.get(kind, kinds["categorical"])
    return [
        coordinate,
        concrete,
        math.log1p(len(domain)),
        dimension_position / max(1, dimension_count - 1),
        component_position / max(1, arity - 1),
        math.log1p(max(1, arity)),
        *kind_features,
        *signed_hash_features(name),
    ]


def derive_context_features(metadata: Mapping[str, Any], candidate_count: int, dimensions: int) -> dict[str, float]:
    """Prefer persisted structured features, with a conservative schema-v9 fallback."""
    model_context = metadata.get("model_context")
    if isinstance(model_context, Mapping):
        supplied = model_context.get("context_features")
        if isinstance(supplied, Mapping):
            result: dict[str, float] = {}
            for name, value in supplied.items():
                if isinstance(name, str) and isinstance(value, (int, float)) and math.isfinite(float(value)):
                    result[name] = float(value)
            if result:
                return result

    result = {
        "candidate_count_log1p": math.log1p(candidate_count),
        "dimension_count_log1p": math.log1p(dimensions),
    }
    launch = str(metadata.get("launch_specification", ""))
    for index, raw in enumerate(re.findall(r"\d+(?:\.\d+)?", launch)[:6]):
        result[f"launch_value_{index}_log1p"] = math.log1p(float(raw))
    for field, buckets in (("kernel", 8), ("device", 8)):
        values = signed_hash_features(str(metadata.get(field, "")), buckets)
        result.update({f"{field}_hash_{index}": item for index, item in enumerate(values)})
    identities = metadata.get("identity_entries", ())
    identity_text = "|".join(map(str, identities)) if isinstance(identities, Sequence) else str(identities)
    result.update(
        {
            f"identity_hash_{index}": item
            for index, item in enumerate(signed_hash_features(identity_text, 8))
        }
    )
    return result

