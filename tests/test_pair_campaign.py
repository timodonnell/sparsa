import copy

import pytest

from scripts.pair_campaign import Campaign, comparison, shortlist, validate_result
from scripts.pair_trials import training_config
from sparsa.data import benchmark


def score(r):
    return {"validation": {"r_precision": r}}


def test_shortlist_reserves_substantial_design():
    rows = {
        "C0": score(0.2),
        "C1": score(0.25),
        "D1": score(0.24),
        "A1": score(0.18),
        "J1": score(0.19),
    }
    assert shortlist(rows) == ["C1", "J1"]
    rows["A1"] = score(0.26)
    assert shortlist(rows) == ["A1", "C1"]


def test_promotion_requires_two_seeds_and_long_guard():
    def r(v, l):
        return {
            "per_protein": {"all": {"a": v, "b": v + 0.01}, "long": {"a": l, "b": l}}
        }

    base = {s: r(0.2, 0.15) for s in ("17", "37")}
    new = {s: r(0.21, 0.16) for s in ("17", "37")}
    assert comparison(new, base)["passes_target"]
    new["37"] = r(0.195, 0.16)
    assert not comparison(new, base)["passes_target"]
    new = {s: r(0.21, 0.14) for s in ("17", "37")}
    assert not comparison(new, base)["passes_target"]


def test_trial_rejects_exposure_config_and_source_drift():
    cfg = training_config("C0", 17, 25000, {"batch_size": 4})
    stems = {r["stem"] for r in benchmark()}
    row = {
        "step": 25000,
        "validation": {"proteins": 97, "r_precision": 0.2, "long_r_precision": 0.2},
        "provenance": {
            "training_config": cfg,
            "world_size": 8,
            "code_sha256": {"a": "b"},
        },
        "per_protein": {k: {s: 0.2 for s in stems} for k in ("all", "long")},
        "exposure": {"step": 25000, "ranks": [{"digest": "a"}]},
        "source_files_sha256": "shards",
    }
    assert validate_result(row, "C0", 17, 25000, {"a": "b"}, cfg) == row
    reference = copy.deepcopy(row)
    reference["exposure"]["ranks"][0]["digest"] = "different"
    with pytest.raises(ValueError, match="exposure"):
        validate_result(row, "C0", 17, 25000, {"a": "b"}, cfg, reference)
    with pytest.raises(ValueError, match="source"):
        validate_result(row, "C0", 17, 25000, {"a": "changed"}, cfg)
    with pytest.raises(ValueError, match="config"):
        validate_result(row, "C0", 17, 25000, {"a": "b"}, cfg | {"lr": 0.4})


@pytest.mark.parametrize(
    "spent,stage,finalists",
    [(500, "confirmation", ["C2"]), (850, "budget_limited", None)],
)
def test_confirmation_admission_requires_complete_paired_budget(
    spent, stage, finalists
):
    c = Campaign.__new__(Campaign)
    c.state = {
        "stage": "screening",
        "screen_arms": ["C0", "C2", "J1"],
        "results": {
            f"{a.lower()}-s17-25000": score(r)
            for a, r in [("C0", 0.2), ("C2", 0.3), ("J1", 0.25)]
        },
        "profile": {
            "arms": {
                a: {"selected": {"estimated_seconds_per_step": s}}
                for a, s in [("C0", 0.1), ("C2", 0.1), ("J1", 1.0)]
            }
        },
    }
    c.trial = lambda *args: True
    c.spent = lambda: spent
    c.record = lambda **kw: c.state.update(kw)
    c.tick()
    assert c.state["stage"] == stage
    assert c.state.get("finalists") == finalists


def test_confirmation_runs_control_and_candidate_at_same_endpoint_both_seeds():
    c = Campaign.__new__(Campaign)
    c.state = {"stage": "confirmation", "finalists": ["J1"], "results": {}}
    calls = []

    def trial(arm, seed, step):
        calls.append((arm, seed, step))
        value = 0.2 if arm == "C0" else 0.21
        c.state["results"][f"{arm.lower()}-s{seed}-{step}"] = {
            "validation": {"r_precision": value},
            "per_protein": {k: {"a": value, "b": value} for k in ("all", "long")},
        }
        return True

    c.trial = trial
    c.record = lambda **kw: c.state.update(kw)
    c.finish = lambda: c.state.update(published=True)
    c.tick()
    assert calls == [
        ("C0", 17, 100000),
        ("C0", 37, 100000),
        ("J1", 17, 100000),
        ("J1", 37, 100000),
    ]
    assert c.state["stage"] == "complete"
    assert c.state["promotion"] == "J1"


def test_recovery_comparison_allows_roundoff_but_preserves_exact_data_checks():
    import torch

    from scripts.pair_smoke import equal

    a = torch.tensor([1.0, 1e-7])
    b = torch.tensor([1.0, 1e-7 + 1e-14])
    stats = {"elements": 0, "unequal_elements": 0, "max_absolute_difference": 0.0}
    equal(a, b, atol=1e-8, rtol=1e-6, stats=stats)
    assert stats["unequal_elements"] == 1
    assert 0 < stats["max_absolute_difference"] < 1e-13
    with pytest.raises(AssertionError):
        equal(a, b)
    with pytest.raises(AssertionError):
        equal(a, a + 0.001, atol=1e-8, rtol=1e-6)
    with pytest.raises(AssertionError):
        equal({"cursor": b"original"}, {"cursor": b"changed"})
