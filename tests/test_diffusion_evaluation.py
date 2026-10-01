import json

import numpy as np
import pytest
import torch

from sparsa import evaluate_diffusion as evaluation
from sparsa.train_diffusion import validation_prefix
from sparsa.vendor.marinfold_metrics import metric_rows, resolved_pairs, true_matrix


def test_incomplete_ranking_divides_by_ground_truth_count():
    truth = np.zeros((32, 32), dtype=bool)
    truth[0, [6, 12, 24]] = True
    resolved = np.ones(32, dtype=bool)
    assert evaluation._score_ordered([(0, 6)], truth, resolved, 6, None, 3) == 1 / 3
    assert evaluation._score_ordered([], truth, resolved, 6, None, 3) == 0
    assert evaluation._score_ordered([(0, 6)] * 3, truth, resolved, 6, None, 3) == 1 / 3
    assert np.isnan(evaluation._score_ordered([], truth, resolved, 6, None, 0))
    assert (
        evaluation._score_ordered([(0, 6), (0, 24)], truth, resolved, 24, None, 1) == 1
    )


def test_dense_ranking_matches_marinfold_r_precision_for_every_range():
    rng = np.random.default_rng(42)
    length = 45
    pi, pj, sep = resolved_pairs(np.array([i for i in range(length) if i % 7]))
    truth = np.triu(rng.random((length, length)) < 0.2, 6)
    score = rng.random((length, length))
    reference = metric_rows(score, truth, pi, pj, sep, length, with_precision=True)
    for row in reference:
        if row["cut"] != "R":
            continue
        lo, hi = evaluation.RANGES[row["range"]]
        mask = (sep >= lo) & ((sep <= hi) if hi is not None else True)
        hits, top = evaluation._ranked_counts(
            score[pi[mask], pj[mask]], truth[pi[mask], pj[mask]], row["n_true"]
        )
        assert top == row["n_true"]
        assert evaluation._precision(hits, row["n_true"]) == row["precision"]


def test_ties_do_not_use_ground_truth_and_invalid_scores_fail():
    assert evaluation._ranked_counts([0.5, 0.5], [False, True], 1) == (0, 1)
    assert evaluation._ranked_counts([], [], 2) == (0, 0)
    with pytest.raises(ValueError, match="finite"):
        evaluation._ranked_counts([np.nan], [True], 1)


def fixture(monkeypatch, *, zero_truth=False):
    length = 32
    record = {
        "dataset": "test",
        "stem": "protein",
        "eval_set": "eval-val",
        "L": length,
        "sequence": "A" * length,
        "resolved": [i for i in reversed(range(length)) if i != 2],
        "contacts": []
        if zero_truth
        else [(0, 6, 1), (0, 12, 1), (0, 24, 1), (2, 30, 1), (1, 25, 0.0001)],
    }
    probability = torch.zeros(2, length, length)
    probability[:, 0, 6] = 0.9
    probability[:, 0, 12] = 0.8
    probability[:, 0, 24] = 0.7
    probability[:, 1, 7] = torch.tensor([0.99, 0.1])
    probability[:, 2, 30] = 1  # Unresolved; excluded even with maximal score.
    probability[:, 3, 4] = 1  # Too close; excluded from every range.
    probability += probability.transpose(1, 2).clone()
    states = torch.zeros_like(probability, dtype=torch.bool)
    states[0, 0, 6] = True
    states[1, 0, 12] = states[1, 1, 7] = True
    states |= states.transpose(1, 2).clone()
    offset = 0

    def sample(model, schedule, tokens, count, *args):
        nonlocal offset
        ids = torch.arange(offset, offset + count) % 2
        offset += count
        return states[ids], probability[ids]

    monkeypatch.setattr(evaluation, "sample_contact_maps", sample)
    return record, states, probability


@pytest.mark.parametrize("batch", [1, 2])
def test_oracle_probability_and_frequency_are_distinct_fixed_r_readouts(
    monkeypatch, batch
):
    record, states, _ = fixture(monkeypatch)
    proteins, rollouts = evaluation.evaluate_records(
        torch.nn.Identity(),
        None,
        [record],
        torch.device("cpu"),
        n_rollouts=2,
        rollout_batch=batch,
    )
    p = proteins[0]
    assert p["n_true"] == 3
    assert p["oracle_r_precision"] == 1
    assert p["mean_r_precision"] == pytest.approx(5 / 6)
    assert p["sampled_oracle_r_precision"] == 1 / 3
    assert p["sampled_mean_r_precision"] == 1 / 3
    assert p["consensus_r_precision"] == 2 / 3
    assert p["probability_ensemble_r_precision"] == 1
    assert p["oracle_curve"] == {"1": 2 / 3, "2": 1}
    assert p["oracle_rollout"] == 1
    assert all(r["n_top"] == r["n_true"] for r in rollouts)
    for r in rollouts:
        if r["n_true"]:
            assert r["r_precision"] == r["true_positives"] / r["n_true"]
            assert r["sampled_r_precision"] == r["sampled_true_positives"] / r["n_true"]
    # Frequency readout retains the same dense consensus definition as upstream.
    pi, pj, sep = resolved_pairs(np.sort(record["resolved"]))
    reference = metric_rows(
        states.float().mean(0).numpy(),
        true_matrix(32, record["contacts"]),
        pi,
        pj,
        sep,
        32,
        with_precision=True,
    )
    assert (
        next(
            r["precision"] for r in reference if r["range"] == "all" and r["cut"] == "R"
        )
        == p["consensus_r_precision"]
    )
    result = evaluation.summarize(proteins)
    assert result["metric_version"] == evaluation.METRIC_VERSION
    json.dumps(result, allow_nan=False)


def test_partial_batch_and_zero_ground_truth_are_supported(monkeypatch):
    record, _, _ = fixture(monkeypatch, zero_truth=True)
    proteins, rollouts = evaluation.evaluate_records(
        torch.nn.Identity(),
        None,
        [record],
        torch.device("cpu"),
        n_rollouts=3,
        rollout_batch=2,
    )
    assert len(rollouts) == 12
    assert proteins[0]["oracle_rollout"] is None
    result = evaluation.summarize(proteins)
    assert result["oracle_r_precision"] is None
    assert result["consensus_r_precision"] is None
    assert result["scored_proteins"] == 0
    json.dumps(result, allow_nan=False)


def test_legacy_results_cannot_be_mixed_or_skip_corrected_validation():
    with pytest.raises(ValueError, match="legacy"):
        evaluation.summarize([{"metric_version": "legacy"}])
    assert (
        validation_prefix("s3://run", 10000)
        == "s3://run/validation-fixed-r-v2/step-10000"
    )
