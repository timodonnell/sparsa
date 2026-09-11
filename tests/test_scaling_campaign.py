import pytest

from scripts.scaling_campaign import choose_extension


def inputs():
    def row(score, params, uri):
        return {
            "validation": {"r_precision": score},
            "parameters": params,
            "checkpoint": uri,
        }

    comparison = {
        "split": "eval-val",
        "test_used": False,
        "models": {
            "release": row(0.2477, 40_000_000, "released"),
            "small-best": row(0.251, 40_000_000, "small"),
            "large-best": row(0.256, 456_000_000, "large"),
            "huge-best": row(0.254, 1_815_000_000, "huge"),
        },
    }
    pilots = [{"name": n, "config": n + ".yaml"} for n in ["small", "large", "huge"]]
    return comparison, pilots


def test_choose_recipe_by_validation_and_extend_to_feasible_capacity():
    comparison, pilots = inputs()
    result = choose_extension(comparison, pilots, {"18b": 6000, "456m": 3000}, 70)
    assert result["source_checkpoint"] == "large"
    assert result["capacity"] == "18b"
    assert result["steps"] == 192000
    assert result["estimated_main_h100_hours"] + 70 + 60 <= 600
    assert not result["test_used"]


def test_capacity_fallback_never_shrinks_a_checkpoint():
    comparison, pilots = inputs()
    comparison["models"]["huge-best"]["validation"]["r_precision"] = 0.26
    result = choose_extension(comparison, pilots, {"18b": 20000, "456m": 3000}, 70)
    assert result["capacity"] == "456m"
    assert result["source_checkpoint"] == "large"
    assert result["steps"] == 200000


def test_reject_heldout_selection_or_exhausted_budget():
    comparison, pilots = inputs()
    comparison["test_used"] = True
    with pytest.raises(ValueError, match="validation"):
        choose_extension(comparison, pilots, {"18b": 6000, "456m": 3000}, 70)
    comparison["test_used"] = False
    with pytest.raises(RuntimeError, match="budget"):
        choose_extension(comparison, pilots, {"18b": 6000, "456m": 3000}, 550)
