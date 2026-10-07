import json
from pathlib import Path

import pytest
import yaml

from alpakatune_ml.contracts import ContractError
from alpakatune_ml.dataset import build_dataset, load_split_config, validate_split_set
from alpakatune_ml.util import sha256_file


FIXTURE = Path(__file__).parent / "fixtures/histories/complete-v9.json"


def _history(tmp_path, device_id, device_class):
    document = json.loads(FIXTURE.read_text())
    metadata = document["contexts"]["fixture-context"]["metadata"]
    metadata["device"] = device_id
    metadata["device_descriptor"] = {"id": device_id, "class": device_class}
    path = tmp_path / device_id / "history.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document))
    return path


def _split_config(tmp_path):
    path = tmp_path / "splits.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "train": {"devices": ["gpu-train"]},
                "validation": {"devices": ["cpu-validation"]},
                "test": {"devices": ["gpu-test"]},
            }
        )
    )
    return path


def _repeated_configuration_history(tmp_path, repeat, runtime_offset):
    document = json.loads(FIXTURE.read_text())
    context = document["contexts"]["fixture-context"]
    context["metadata"]["device"] = "Repeated A100"
    context["metadata"]["device_descriptor"] = {"id": "nvidia-a100", "class": "gpu"}
    context["candidate_count"] = 10
    context["candidate_configurations"] = [{"blockSize": 32 * (index + 1)} for index in range(10)]
    context["candidate_samples"] = [
        [runtime_offset + index * 1.0e-6 + sample * 1.0e-8 for sample in range(3)]
        for index in range(10)
    ]
    context["candidate_estimates"] = [
        sorted(samples)[1] for samples in context["candidate_samples"]
    ]
    context["rejected_candidates"] = [False] * 10
    context["retired_configuration_count"] = 10
    path = tmp_path / f"repeat-{repeat}" / "history.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document))
    return path


def test_dataset_requires_whole_device_splits(tmp_path):
    histories = [
        _history(tmp_path, "gpu-train", "gpu"),
        _history(tmp_path, "cpu-validation", "cpu"),
        _history(tmp_path, "gpu-test", "gpu"),
    ]
    output = tmp_path / "dataset"
    manifest = build_dataset(histories, _split_config(tmp_path), output)
    assert manifest["row_count"] == 6
    splits = validate_split_set(
        [
            output / "train.manifest.json",
            output / "validation.manifest.json",
            output / "test.manifest.json",
        ]
    )
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }


def test_split_config_rejects_device_leakage(tmp_path):
    path = tmp_path / "splits.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "train": {"devices": ["same-device"]},
                "validation": {"devices": ["validation-device"]},
                "test": {"devices": ["same-device"]},
            }
        )
    )
    with pytest.raises(ContractError, match="leaks"):
        load_split_config(path)


def test_validation_rejects_row_and_surface_leakage(tmp_path):
    histories = [
        _history(tmp_path, "gpu-train", "gpu"),
        _history(tmp_path, "cpu-validation", "cpu"),
        _history(tmp_path, "gpu-test", "gpu"),
    ]
    output = tmp_path / "dataset"
    build_dataset(histories, _split_config(tmp_path), output)
    test_manifest_path = output / "test.manifest.json"
    test_manifest = json.loads(test_manifest_path.read_text())
    train_manifest = json.loads((output / "train.manifest.json").read_text())
    test_manifest["surface_ids"] = train_manifest["surface_ids"]
    test_manifest_path.write_text(json.dumps(test_manifest))
    with pytest.raises(ContractError, match="surface .* leaks"):
        validate_split_set(
            [
                output / "train.manifest.json",
                output / "validation.manifest.json",
                test_manifest_path,
            ]
        )


def test_validation_rejects_row_id_leakage_even_when_surfaces_differ(tmp_path):
    histories = [
        _history(tmp_path, "gpu-train", "gpu"),
        _history(tmp_path, "cpu-validation", "cpu"),
        _history(tmp_path, "gpu-test", "gpu"),
    ]
    output = tmp_path / "dataset"
    build_dataset(histories, _split_config(tmp_path), output)
    train_row_id = json.loads((output / "train.jsonl").read_text().splitlines()[0])["row_id"]
    test_labels = output / "test.jsonl"
    test_rows = [json.loads(line) for line in test_labels.read_text().splitlines()]
    test_rows[0]["row_id"] = train_row_id
    test_labels.write_text("".join(json.dumps(row) + "\n" for row in test_rows))
    test_manifest_path = output / "test.manifest.json"
    test_manifest = json.loads(test_manifest_path.read_text())
    test_manifest["labels_sha256"] = sha256_file(test_labels)
    test_manifest_path.write_text(json.dumps(test_manifest))
    with pytest.raises(ContractError, match="row .* leaks"):
        validate_split_set(
            [
                output / "train.manifest.json",
                output / "validation.manifest.json",
                test_manifest_path,
            ]
        )


def test_configuration_split_aggregates_repeats_and_holds_out_rows(tmp_path):
    histories = [
        _repeated_configuration_history(tmp_path, repeat, 1.0e-5 + repeat * 1.0e-7)
        for repeat in range(3)
    ]
    split_config = tmp_path / "configuration-splits.yaml"
    split_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "policy": "configuration",
                "seed": 1729,
                "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
            }
        )
    )
    output = tmp_path / "configuration-dataset"
    manifest = build_dataset(histories, split_config, output)
    assert manifest["split_policy"] == "configuration"
    assert manifest["row_count"] == 10
    assert manifest["replicate_label_count"] == 30
    splits = validate_split_set(
        [
            output / "train.manifest.json",
            output / "validation.manifest.json",
            output / "test.manifest.json",
        ]
    )
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 8,
        "validation": 1,
        "test": 1,
    }
    assert all(
        row["source"]["replicate_count"] == 3
        for rows in splits.values()
        for row in rows
    )
