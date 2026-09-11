import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from research.controller import ROOT, Controller, initialize
from research.core import promotion, reservation, validate_result, validate_source


def protocol():
    p = yaml.safe_load((ROOT / "research/protocol.yaml").read_text())
    p.update(max_candidates=1, gpu_hours=16)
    p["training"]["model"].update(
        sequence_dim=32,
        sequence_layers=2,
        heads=4,
        pair_dim=8,
        pair_layers=4,
        triangle_every=2,
        triangle_dim=4,
    )
    return p


def scores(value, stems=("a", "b", "c")):
    return {"per_protein": {r: dict.fromkeys(stems, value) for r in ("all", "long")}}


def test_promotion_requires_both_seeds_and_long_range():
    p = protocol()
    baseline = {s: scores(0.2) for s in p["seeds"]}
    candidate = {s: scores(0.23) for s in p["seeds"]}
    assert promotion(candidate, baseline, p)["accepted"]
    candidate[37] = scores(0.199)
    assert not promotion(candidate, baseline, p)["accepted"]
    candidate[37] = scores(0.23)
    candidate[37]["per_protein"]["long"] = dict.fromkeys(("a", "b", "c"), 0.1)
    assert not promotion(candidate, baseline, p)["accepted"]
    with pytest.raises(ValueError, match="both matched seeds"):
        promotion({17: scores(0.9)}, baseline, p)


def test_source_contract_rejects_io():
    source = (ROOT / "sparsa/model.py").read_text()
    validate_source(source, source)
    with pytest.raises(ValueError, match="Tokenizer"):
        validate_source(source.replace(".upper()", ".lower()"), source)
    for source in (
        "import os",
        "from pathlib import Path",
        "torch.load('labels.pt')",
        "open('labels.json')",
        "eval('1')",
    ):
        with pytest.raises(ValueError):
            validate_source(source)


class FakeBackend:
    """Simulate scheduler lifecycle; scores here are fixtures, not model results."""

    def __init__(self):
        self.jobs = {}
        self.submissions = []
        self.cancelled = []
        self.lose_response = False

    def status(self, job):
        if job not in self.jobs:
            return {"state": "not_found"}
        state = self.jobs[job]
        state["polls"] += 1
        return {
            "state": "running" if state["polls"] == 1 else "succeeded",
            "gpu_hours": 0.05,
            "unknown_attempts": 0,
        }

    def submit(self, job, workspace, gpus, timeout, args):
        assert job not in self.jobs
        assert gpus == 4
        self.submissions.append(job)
        self.jobs[job] = {"workspace": Path(workspace), "polls": 0}
        if self.lose_response:
            self.lose_response = False
            raise subprocess.CalledProcessError(1, "simulated_lost_submit_response")

    def result(self, uri):
        trial_id = uri.split("/")[-2]
        rec = next(v for k, v in self.jobs.items() if k.endswith(trial_id))
        spec = json.loads((rec["workspace"] / "research_trial.json").read_text())
        cfg = yaml.safe_load(
            (rec["workspace"] / "configs/research_trial.yaml").read_text()
        )
        value = 0.2 if spec["candidate_id"] == "c000" else 0.25
        result = scores(value, spec["contract"]["validation_stems"])
        return result | {
            "trial_id": trial_id,
            "candidate_id": spec["candidate_id"],
            "seed": cfg["seed"],
            "step": cfg["steps"],
            "contract": spec["contract"],
            "split": "eval-val",
            "proteins": 97,
            "r_precision": value,
            "long_r_precision": value,
            "parameters": 1000,
            "files": spec["files"],
            "teacher_inventory_sha256": "fixture",
        }

    def cancel(self, job):
        self.cancelled.append(job)

    def diagnostic(self, job):
        return "simulated infrastructure failure"

    def close(self):
        pass


def setup(tmp_path, p=None):
    path = tmp_path / "campaign"
    initialize(
        path,
        p or protocol(),
        {"name": "test", "proposer": "seeded", "output_prefix": "s3://test/campaign"},
    )
    backend = FakeBackend()
    return Controller(path, backend), backend


def finish(controller, limit=40):
    for _ in range(limit):
        controller.tick()
        if controller.state["stage"] in {
            "complete",
            "budget_exhausted",
            "baseline_failed",
            "paused",
        }:
            return
    pytest.fail("Controller did not reach a bounded stopping state")


def test_full_search_promotes_only_after_confirmation(tmp_path):
    c, backend = setup(tmp_path)
    finish(c)
    assert c.state["stage"] == "complete"
    assert len(backend.submissions) == 4
    assert c.state["champion"] == "c001"
    assert len(c.candidate("c001")["results"]) == 2
    assert c.candidate("c001")["decision"]["accepted"]
    assert c.charged() == 4 * reservation(c.protocol)
    assert (c.path / "REPORT.md").exists()


def test_restart_after_uncertain_submission_does_not_duplicate(tmp_path):
    c, backend = setup(tmp_path)
    c.tick()  # persist reservation first
    backend.lose_response = True
    with pytest.raises(subprocess.CalledProcessError):
        c.tick()
    c = Controller(c.path, backend)
    c.tick()
    c.tick()
    assert len(backend.submissions) == 1
    assert c.state["trials"][0]["state"] == "completed"


def test_budget_stops_before_proposing_unfunded_trial(tmp_path):
    p = protocol()
    p["gpu_hours"] = 2 * reservation(p)
    c, backend = setup(tmp_path, p)
    finish(c)
    assert c.state["stage"] == "budget_exhausted"
    assert len(backend.submissions) == 2
    assert len(c.state["candidates"]) == 1


def test_reserved_source_tampering_is_rejected_before_submission(tmp_path):
    c, backend = setup(tmp_path)
    c.tick()
    trial = c.state["trials"][0]
    (Path(trial["workspace"]) / "sparsa/evaluate.py").write_text("# changed scoring")
    with pytest.raises(ValueError, match="source changed"):
        c.tick()
    assert not backend.submissions


def test_stop_does_not_abandon_an_active_trial(tmp_path):
    c, backend = setup(tmp_path)
    c.tick()
    (c.path / "STOP").touch()
    finish(c)
    assert len(backend.submissions) == 1
    assert c.state["trials"][0]["state"] == "completed"
    assert c.state["stage"] == "paused"


def test_result_integrity_rejects_wrong_protocol_or_coverage(tmp_path):
    c, backend = setup(tmp_path)
    c.tick()
    c.tick()
    trial = c.state["trials"][0]
    result = backend.result(trial["out"] + "/result.json")
    validate_result(result, trial, c.protocol, c.contract)
    for bad in (
        result | {"seed": 99},
        result | {"split": "eval-test"},
        result | {"r_precision": 0.99},
        result | {"r_precision": float("nan")},
    ):
        with pytest.raises(ValueError):
            validate_result(bad, trial, c.protocol, c.contract)
    bad = copy.deepcopy(result)
    bad["per_protein"]["all"].pop(next(iter(bad["per_protein"]["all"])))
    with pytest.raises(ValueError, match="validation proteins"):
        validate_result(bad, trial, c.protocol, c.contract)


def test_resource_limit_cancels_only_the_active_trial(tmp_path):
    c, backend = setup(tmp_path)
    c.tick()
    c.tick()
    trial = c.state["trials"][0]
    backend.status = lambda job: {
        "state": "running",
        "gpu_hours": trial["reserved_gpu_hours"] + 0.01,
        "unknown_attempts": 0,
    }
    c.tick()
    assert backend.cancelled == [trial["job"]]
    assert c.charged() > trial["reserved_gpu_hours"]


def test_failed_baseline_does_not_create_an_accuracy_score(tmp_path):
    c, backend = setup(tmp_path)
    c.tick()
    c.tick()
    backend.status = lambda job: {"state": "failed", "gpu_hours": 0.01}
    c.tick()
    c.tick()
    assert c.state["stage"] == "baseline_failed"
    assert not c.candidate("c000")["results"]
    assert len(backend.submissions) == 1
    assert "simulated infrastructure" in c.candidate("c000")["error"]
