import copy
import json
from pathlib import Path

import pytest

from alpakatune_ml.contracts import ContractError, load_history


FIXTURE = Path(__file__).parent / "fixtures/histories/complete-v9.json"


def test_complete_exhaustive_history_produces_candidate_labels():
    surface = load_history(FIXTURE)[0]
    surface.validate_full_exhaustive()
    rows = list(surface.rows())
    assert len(rows) == 2
    assert rows[0]["surface_id"] == rows[1]["surface_id"]
    assert rows[0]["device_id"] == "gpu-train"
    assert len(rows[0]["dimension_features"][0]) == 18


def test_context_fingerprint_separates_surfaces_and_rows(tmp_path):
    document = json.loads(FIXTURE.read_text())
    document["contexts"]["second-context"] = copy.deepcopy(
        document["contexts"]["fixture-context"]
    )
    path = tmp_path / "history.json"
    path.write_text(json.dumps(document))
    first, second = load_history(path)
    assert first.workload_id == second.workload_id
    assert first.device_id == second.device_id
    assert first.surface_id != second.surface_id
    assert {row["row_id"] for row in first.rows()}.isdisjoint(
        row["row_id"] for row in second.rows()
    )


def test_capped_exhaustive_history_is_rejected(tmp_path):
    document = json.loads(FIXTURE.read_text())
    context = document["contexts"]["fixture-context"]
    context["completion_reason"] = "maximum_executions"
    context["execution_budget_reached"] = True
    path = tmp_path / "history.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ContractError, match="maximum_executions"):
        load_history(path)[0].validate_full_exhaustive()


def test_schema_v10_prefers_structured_model_context(tmp_path):
    document = json.loads(FIXTURE.read_text())
    document["schema_version"] = 10
    metadata = document["contexts"]["fixture-context"]["metadata"]
    metadata["model_context"] = {
        "feature_schema_version": 1,
        "workload_id": "structured/vectorAdd",
        "device_class": "gpu",
        "context_features": {"cores_log1p": 7.0},
        "dimensions": [
            {
                "name": "blockSize",
                "kind": "launch",
                "cardinality": 2,
                "component_index": 0,
                "vector_arity": 2,
                "concrete_values": [64, 128],
            },
            {
                "name": "blockSize",
                "kind": "launch",
                "cardinality": 1,
                "component_index": 1,
                "vector_arity": 2,
                "concrete_values": [1],
            },
        ],
    }
    path = tmp_path / "history.json"
    path.write_text(json.dumps(document))
    surface = load_history(path)[0]
    surface.validate_full_exhaustive()
    row = next(surface.rows())
    assert row["workload_id"] == "structured/vectorAdd"
    assert row["context_features"] == {"cores_log1p": 7.0}
    assert row["dimension_features"][0][8] == 1.0
    assert len(row["dimension_features"]) == 2
    assert row["dimension_features"][1][4] == 1.0
