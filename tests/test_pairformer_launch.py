import copy

import pandas as pd
import pytest

from scripts.run_pairformer import verify_preflight
from sparsa.data import benchmark
from sparsa.train_diffusion import training_configs_compatible


def test_resume_allows_only_recovery_cadence_changes():
    saved = {
        "checkpoint_every": 1000,
        "lr": 0.0002,
        "steps": 300000,
        "crop": 384,
        "seed": 23,
    }
    assert training_configs_compatible(saved, saved | {"checkpoint_every": 100})
    for key, value in [("lr", 0.001), ("steps", 30000), ("crop", 256), ("seed", 24)]:
        assert not training_configs_compatible(saved, saved | {key: value})


def fixture():
    per = pd.DataFrame(
        [
            {
                "dataset": r["dataset"],
                "stem": r["stem"],
                "eval_set": "eval-val",
                "metric_version": "fixed-r-v2",
                "n_rollouts": 100,
                "oracle_r_precision": 0.25,
                "consensus_r_precision": 0.2,
            }
            for r in benchmark()
        ]
    )
    config = {"rollout_batch": 4, "rollout_seed": 20260925}
    summary = {
        "metric_version": "fixed-r-v2",
        "split": "eval-val",
        "held_out_used": False,
        "proteins": 97,
        "n_rollouts": 100,
        "oracle_r_precision": 0.25,
        "consensus_r_precision": 0.2,
        **config,
    }
    return summary, per, config


def test_preflight_requires_full_fixed_validation_and_matching_metric_means():
    summary, per, config = fixture()
    verify_preflight(summary, per, config)
    for field, value in [
        ("proteins", 96),
        ("n_rollouts", 2),
        ("held_out_used", True),
        ("split", "eval-test"),
        ("rollout_batch", 1),
    ]:
        bad = copy.deepcopy(summary)
        bad[field] = value
        with pytest.raises(ValueError):
            verify_preflight(bad, per, config)
    with pytest.raises(ValueError):
        verify_preflight(summary, per.iloc[:-1], config)
    bad = per.copy()
    bad.loc[0, "stem"] = "wrong-protein"
    with pytest.raises(ValueError):
        verify_preflight(summary, bad, config)
    bad = per.copy()
    bad.loc[0, "oracle_r_precision"] = 1.0
    with pytest.raises(AssertionError):
        verify_preflight(summary, bad, config)
