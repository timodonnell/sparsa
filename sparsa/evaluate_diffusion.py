"""Fixed-R evaluation of probability-ranked rollouts and contact ensembles.

For each range, R is the ground-truth contact count in the resolved pair
universe. Every precision denominator is R, including sparse sampled maps.
Oracle selection uses each rollout's dense final denoiser probabilities;
frequency consensus and mean-probability ensembles are separate readouts.
"""

from __future__ import annotations

import hashlib
import math
import time
from datetime import timedelta

import numpy as np
import torch

from sparsa.diffusion import sample_contact_maps
from sparsa.model import tokenize
from sparsa.vendor.marinfold_metrics import resolved_pairs, true_matrix

METRIC_VERSION = "fixed-r-v2"
METRIC_POLICY = {
    "metric_version": METRIC_VERSION,
    "oracle_ranking": "final_denoiser_probability_all_eligible_pairs",
    "consensus_ranking": "sampled_contact_frequency_all_eligible_pairs",
    "probability_ensemble_ranking": "mean_final_denoiser_probability",
    "precision_denominator": "ground_truth_contacts_in_resolved_range",
    "tie_break": "ascending_residue_pair",
    "zero_ground_truth": "undefined_excluded_from_macro_average",
}
RANGES = {"all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None)}
CURVE = (1, 2, 4, 8, 16, 32, 64, 100)
# Full 100-rollout validation can leave faster ranks waiting longer than the
# default ten-minute NCCL timeout. This is a communication wait budget, not a
# sampling or training parameter.
COLLECTIVE_TIMEOUT = timedelta(hours=4)
VALIDATION_PARTITION = "greedy_length_cubed_v1"


def validation_shards(records, world):
    """Balance triangle work without changing per-protein seeds or rollouts."""
    if world < 1:
        raise ValueError("Expected a positive world size")
    shards, costs = [[] for _ in range(world)], [0] * world
    ordered = sorted(records, key=lambda r: (-r["L"], r["dataset"], r["stem"]))
    for record in ordered:
        rank = min(range(world), key=lambda i: (costs[i], i))
        shards[rank].append(record)
        costs[rank] += record["L"] ** 3
    return shards


def _seed(base, record):
    key = f"{base}:{record['dataset']}:{record['stem']}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)


def _ranked_counts(scores, labels, n_true):
    """Return hits and selected count; missing predictions never reduce R."""
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    if scores.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Expected equally sized score and label vectors")
    if not np.isfinite(scores).all() or n_true < 0:
        raise ValueError("Expected finite scores and non-negative ground-truth count")
    order = np.argsort(-scores, kind="stable")[:n_true]
    return int(labels[order].sum()), len(order)


def _precision(hits, n_true):
    return hits / n_true if n_true > 0 else float("nan")


def _score_ordered(pairs, truth, resolved, lo, hi, n_true):
    """Score a possibly incomplete unique contact ranking, always dividing by R."""
    if n_true <= 0:
        return float("nan")
    hits, selected, seen = 0, 0, set()
    for i, j in pairs:
        sep = j - i
        if (
            (i, j) not in seen
            and resolved[i]
            and resolved[j]
            and sep >= lo
            and (hi is None or sep <= hi)
        ):
            seen.add((i, j))
            hits += bool(truth[i, j])
            selected += 1
            if selected == n_true:
                break
    return hits / n_true


def _mean_jaccard(sets):
    if len(sets) < 2:
        return 1.0
    values = []
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            union = len(left | right)
            values.append(len(left & right) / union if union else 1.0)
    return float(np.mean(values))


def _reduce(values, operation):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(operation(values)) if len(values) else float("nan")


@torch.inference_mode()
def evaluate_records(
    model,
    schedule,
    records,
    device,
    *,
    n_rollouts=100,
    rollout_batch=4,
    seed=20260925,
    temperature=1.0,
    pos_weight=4.0,
    self_condition_guidance=1.0,
    progress_callback=None,
):
    """Score all resolved pairs; ground truth sets R but never the ranking."""
    if n_rollouts < 1 or rollout_batch < 1:
        raise ValueError("Rollout counts must be positive")
    if pos_weight <= 0:
        raise ValueError("Positive weight must be positive")
    model.eval()
    protein_rows, rollout_rows = [], []
    for rec in records:
        tokens = tokenize(rec["sequence"])[None].to(device)
        generator = torch.Generator(device=device).manual_seed(_seed(seed, rec))
        truth = true_matrix(rec["L"], rec["contacts"])
        # Canonical pair order makes ties independent of labels and input ordering.
        pi, pj, sep = resolved_pairs(np.unique(rec["resolved"]).astype(np.int64))
        eligible = sep >= 6
        pi, pj, sep = pi[eligible], pj[eligible], sep[eligible]
        labels = truth[pi, pj]
        masks, counts = {}, {}
        for name, (lo, hi) in RANGES.items():
            masks[name] = (sep >= lo) & ((sep <= hi) if hi is not None else True)
            counts[name] = int(labels[masks[name]].sum())
        by_range = {name: [] for name in RANGES}
        sampled_by_range = {name: [] for name in RANGES}
        contact_sets = []
        vote = np.zeros(len(pi), dtype=np.int64)
        probability_sum = np.zeros(len(pi), dtype=np.float64)
        start = time.perf_counter()
        sampling_options = {}
        if getattr(getattr(model, "config", None), "pairformer_layers", 0):
            # This cache is per protein, outside both the reverse chain and
            # rollout batching. In noisy mode encode contains only the shallow
            # sequence encoder; every pair block still sees each sampled map.
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                sampling_options["encoded"] = model.encode(tokens)
        for offset in range(0, n_rollouts, rollout_batch):
            count = min(rollout_batch, n_rollouts - offset)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                states, probabilities = sample_contact_maps(
                    model,
                    schedule,
                    tokens,
                    count,
                    generator,
                    temperature,
                    math.log(pos_weight),
                    self_condition_guidance,
                    **sampling_options,
                )
            states = states.cpu().numpy()
            probabilities = probabilities.float().cpu().numpy()
            for index in range(count):
                selected = states[index, pi, pj].astype(bool)
                scores = probabilities[index, pi, pj]
                vote += selected
                probability_sum += scores
                si, sj = np.nonzero(np.triu(states[index], 6))
                contact_sets.append(set(zip(si.tolist(), sj.tolist(), strict=True)))
                for name, mask in masks.items():
                    nt = counts[name]
                    hits, n_top = _ranked_counts(scores[mask], labels[mask], nt)
                    chosen = mask & selected
                    sampled_hits, sampled_top = _ranked_counts(
                        scores[chosen], labels[chosen], nt
                    )
                    score = _precision(hits, nt)
                    sampled_score = _precision(sampled_hits, nt)
                    by_range[name].append(score)
                    sampled_by_range[name].append(sampled_score)
                    rollout_rows.append(
                        {
                            "dataset": rec["dataset"],
                            "stem": rec["stem"],
                            "metric_version": METRIC_VERSION,
                            "range": name,
                            "rollout": offset + index,
                            "r_precision": score,
                            "n_true": nt,
                            "n_candidate": int(mask.sum()),
                            "n_top": n_top,
                            "true_positives": hits,
                            "sampled_r_precision": sampled_score,
                            "sampled_n_top": sampled_top,
                            "sampled_true_positives": sampled_hits,
                            "sampled_contacts_in_range": int(chosen.sum()),
                            "n_contacts": len(contact_sets[-1]),
                        }
                    )
        elapsed = time.perf_counter() - start
        consensus, probability_ensemble = {}, {}
        for name, mask in masks.items():
            nt = counts[name]
            consensus[name] = _precision(
                _ranked_counts(vote[mask], labels[mask], nt)[0], nt
            )
            probability_ensemble[name] = _precision(
                _ranked_counts(probability_sum[mask] / n_rollouts, labels[mask], nt)[0],
                nt,
            )
        all_scores = np.asarray(by_range["all"])
        sampled_scores = np.asarray(sampled_by_range["all"])
        curve = {
            str(n): _reduce(all_scores[:n], np.max) for n in CURVE if n <= n_rollouts
        }
        protein_rows.append(
            {
                "dataset": rec["dataset"],
                "stem": rec["stem"],
                "eval_set": rec["eval_set"],
                "metric_version": METRIC_VERSION,
                "L": rec["L"],
                "n_rollouts": n_rollouts,
                "oracle_r_precision": _reduce(all_scores, np.max),
                "oracle_long_r_precision": _reduce(by_range["long"], np.max),
                "mean_r_precision": _reduce(all_scores, np.mean),
                "mean_long_r_precision": _reduce(by_range["long"], np.mean),
                "sampled_oracle_r_precision": _reduce(sampled_scores, np.max),
                "sampled_oracle_long_r_precision": _reduce(
                    sampled_by_range["long"], np.max
                ),
                "sampled_mean_r_precision": _reduce(sampled_scores, np.mean),
                "sampled_mean_long_r_precision": _reduce(
                    sampled_by_range["long"], np.mean
                ),
                "consensus_r_precision": consensus["all"],
                "consensus_long_r_precision": consensus["long"],
                "probability_ensemble_r_precision": probability_ensemble["all"],
                "probability_ensemble_long_r_precision": probability_ensemble["long"],
                "oracle_rollout": int(np.nanargmax(all_scores))
                if np.isfinite(all_scores).any()
                else None,
                "mean_contacts": float(np.mean([len(x) for x in contact_sets])),
                "unique_maps": len({frozenset(x) for x in contact_sets}),
                "mean_pairwise_jaccard": _mean_jaccard(contact_sets),
                "n_true": counts["all"],
                "n_true_long": counts["long"],
                "oracle_curve": curve,
                "sampled_oracle_curve": {
                    str(n): _reduce(sampled_scores[:n], np.max)
                    for n in CURVE
                    if n <= n_rollouts
                },
                "inference_seconds": elapsed,
            }
        )
        if progress_callback is not None:
            progress_callback(len(protein_rows), len(records), protein_rows[-1])
    return protein_rows, rollout_rows


def summarize(protein_rows):
    if not protein_rows:
        raise ValueError("Cannot summarize an empty evaluation")
    if any(row.get("metric_version") != METRIC_VERSION for row in protein_rows):
        raise ValueError("Cannot combine legacy or mixed scoring versions")

    def mean(key):
        value = _reduce([row[key] for row in protein_rows], np.mean)
        return value if np.isfinite(value) else None

    result = dict(METRIC_POLICY)
    result["proteins"] = len(protein_rows)
    for key in (
        "oracle_r_precision",
        "oracle_long_r_precision",
        "mean_r_precision",
        "mean_long_r_precision",
        "sampled_oracle_r_precision",
        "sampled_oracle_long_r_precision",
        "sampled_mean_r_precision",
        "sampled_mean_long_r_precision",
        "consensus_r_precision",
        "consensus_long_r_precision",
        "probability_ensemble_r_precision",
        "probability_ensemble_long_r_precision",
        "mean_contacts",
        "mean_pairwise_jaccard",
    ):
        result[key] = mean(key)
    result["mean_unique_maps"] = mean("unique_maps")
    result["scored_proteins"] = sum(row["n_true"] > 0 for row in protein_rows)
    result["scored_long_proteins"] = sum(row["n_true_long"] > 0 for row in protein_rows)
    result["inference_seconds"] = sum(row["inference_seconds"] for row in protein_rows)
    for name in ("oracle_curve", "sampled_oracle_curve"):
        keys = set(protein_rows[0][name])
        if any(set(row[name]) != keys for row in protein_rows):
            raise ValueError("Cannot combine evaluations with different rollout curves")
        result[name] = {}
        for n in sorted(keys, key=int):
            value = _reduce([row[name][n] for row in protein_rows], np.mean)
            result[name][n] = value if np.isfinite(value) else None
    return result
