from pathlib import Path

import yaml

import pytest

from alpakatune_ml import collect
from alpakatune_ml.collect import exhaustive_config, resolve_device_id
from alpakatune_ml.contracts import ContractError


def test_exhaustive_config_removes_budgets_and_early_stops(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "tuning": {
                    "strategy": "random",
                    "warmup_runs": 4,
                    "runs_per_candidate": 20,
                    "minimum_runs_per_candidate": 1,
                    "mann_whitney_early_stop": True,
                    "max_consecutive_runs": 5,
                    "maximum_executions": 40000,
                    "maximum_retired_configurations": 4000,
                },
            }
        )
    )
    config = exhaustive_config(base, tmp_path / "history.json")
    tuning = config["tuning"]
    assert tuning["strategy"] == "exhaustive"
    assert tuning["runs_per_candidate"] == tuning["minimum_runs_per_candidate"] == 3
    assert tuning["warmup_runs"] == 1
    assert tuning["max_consecutive_runs"] == 4
    assert tuning["mann_whitney_early_stop"] is False
    assert "maximum_executions" not in tuning
    assert "maximum_retired_configurations" not in tuning


def test_auto_device_id_is_resolved_once():
    assert resolve_device_id("auto", None, "nvidia-a100") == "nvidia-a100"
    assert resolve_device_id("auto", "nvidia-a100", "nvidia-a100") == "nvidia-a100"
    with pytest.raises(ContractError, match="moved between devices"):
        resolve_device_id("auto", "nvidia-a100", "nvidia-h100")


def test_explicit_device_id_still_rejects_mismatch():
    with pytest.raises(ContractError, match="does not match campaign platform"):
        resolve_device_id("nvidia-a100", "nvidia-a100", "nvidia-h100")


class _Surface:
    def __init__(self, device_id, device_class="gpu"):
        self.device_id = device_id
        self.device_class = device_class

    def validate_full_exhaustive(self):
        return None


def _campaign(tmp_path, run_names):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"schema_version": 1, "tuning": {}}))
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "resume-auto",
                "base_tuning_config": str(base),
                "revisions": {
                    "alpakatune": "1234567",
                    "alpaka": "2345678",
                    "collector": "3456789",
                },
                "platform": {"device_id": "auto", "device_class": "gpu"},
                "runs": [
                    {
                        "name": name,
                        "workload_id": f"{name}/default",
                        "command": ["unused"],
                    }
                    for name in run_names
                ],
            }
        )
    )
    return campaign


def _completed_run(output, name):
    directory = output / "runs" / name
    directory.mkdir(parents=True)
    (directory / "history.json").write_text("{}")
    (directory / "run.json").write_text('{"name":"%s","status":"completed"}' % name)


def test_resume_auto_resolves_device_from_skipped_history(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, ["first"])
    output = tmp_path / "output"
    _completed_run(output, "first")
    monkeypatch.setattr(collect, "load_history", lambda _path: [_Surface("nvidia-a100")])

    summary = collect.collect_campaign(campaign, output, resume=True)

    assert summary["status"] == "completed"
    assert summary["platform"]["requested_device_id"] == "auto"
    assert summary["platform"]["device_id"] == "nvidia-a100"
    assert summary["runs"][0]["status"] == "skipped"


def test_resume_auto_rejects_mixed_skipped_devices(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, ["first", "second"])
    output = tmp_path / "output"
    _completed_run(output, "first")
    _completed_run(output, "second")

    def histories(path):
        device = "nvidia-a100" if path.parent.name == "first" else "nvidia-h100"
        return [_Surface(device)]

    monkeypatch.setattr(collect, "load_history", histories)
    summary = collect.collect_campaign(campaign, output, resume=True)

    assert summary["status"] == "failed"
    assert summary["failures"] == ["second"]
    assert "moved between devices" in summary["runs"][1]["validation_errors"][0]
    assert summary["platform"]["device_id"] == "nvidia-a100"
