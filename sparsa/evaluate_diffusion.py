"""Oracle-best rollout evaluation for discrete contact diffusion.

The per-rollout score intentionally matches MarinFold's diagnostic: retain the
rollout's ordered selected contacts, take its first R resolved contacts in each
range, and then take the maximum precision across N rollouts.  Consensus and
diversity statistics are diagnostics; oracle best-of-N is the selection metric.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from sparsa.diffusion import sample_contact_maps
from sparsa.model import tokenize
from sparsa.vendor.marinfold_metrics import metric_rows, resolved_pairs, true_matrix

RANGES = {"all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None)}
CURVE = (1, 2, 4, 8, 16, 32, 64, 100)


def _seed(base, record):
    key = f"{base}:{record['dataset']}:{record['stem']}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)


def _score_ordered(pairs, truth, resolved, lo, hi, n_true):
    if n_true <= 0:
        return float("nan")
    hits = []
    for i, j in pairs:
        sep = j - i
        if resolved[i] and resolved[j] and sep >= lo and (hi is None or sep <= hi):
            hits.append(bool(truth[i, j]))
            if len(hits) == n_true:
                break
    return float(np.mean(hits)) if hits else 0.0


def _mean_jaccard(sets):
    if len(sets) < 2:
        return 1.0
    values = []
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            union = len(left | right)
            values.append(len(left & right) / union if union else 1.0)
    return float(np.mean(values))


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
):
    """Evaluate an arbitrary record shard and return auditable per-protein rows."""
    if n_rollouts < 1 or rollout_batch < 1:
        raise ValueError("Rollout counts must be positive")
    model.eval()
    protein_rows, rollout_rows = [], []
    for rec in records:
        tokens = tokenize(rec["sequence"])[None].to(device)
        generator = torch.Generator(device=device).manual_seed(_seed(seed, rec))
        ordered, contact_sets = [], []
        vote = np.zeros((rec["L"], rec["L"]), dtype=np.float32)
        start = time.perf_counter()
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
                )
            states = states.cpu().numpy()
            probabilities = probabilities.float().cpu().numpy()
            for index in range(count):
                i, j = np.nonzero(np.triu(states[index], 6))
                if len(i):
                    order = np.argsort(-probabilities[index, i, j], kind="mergesort")
                    pairs = list(zip(i[order].tolist(), j[order].tolist(), strict=True))
                    vote[i, j] += 1
                    vote[j, i] += 1
                else:
                    pairs = []
                ordered.append(pairs)
                contact_sets.append(set(pairs))
        elapsed = time.perf_counter() - start

        truth = true_matrix(rec["L"], rec["contacts"])
        resolved = np.zeros(rec["L"], dtype=bool)
        resolved[np.asarray(rec["resolved"], dtype=np.int64)] = True
        pi, pj, sep = resolved_pairs(np.asarray(rec["resolved"], dtype=np.int64))
        counts, by_range = {}, {}
        for name, (lo, hi) in RANGES.items():
            in_range = sep >= lo
            if hi is not None:
                in_range &= sep <= hi
            n_true = int(truth[pi[in_range], pj[in_range]].sum())
            counts[name] = n_true
            scores = [
                _score_ordered(pairs, truth, resolved, lo, hi, n_true)
                for pairs in ordered
            ]
            by_range[name] = scores
            for rollout, score in enumerate(scores):
                rollout_rows.append(
                    {
                        "dataset": rec["dataset"],
                        "stem": rec["stem"],
                        "range": name,
                        "rollout": rollout,
                        "r_precision": score,
                        "n_true": n_true,
                        "n_contacts": len(ordered[rollout]),
                    }
                )

        consensus_rows = metric_rows(
            vote / n_rollouts,
            truth,
            pi,
            pj,
            sep,
            rec["L"],
            with_precision=True,
        )
        consensus = {
            row["range"]: float(row["precision"])
            for row in consensus_rows
            if row["cut"] == "R"
        }
        all_scores = np.asarray(by_range["all"], dtype=float)
        long_scores = np.asarray(by_range["long"], dtype=float)
        curve = {
            str(n): float(np.nanmax(all_scores[: min(n, n_rollouts)]))
            for n in CURVE
            if n <= n_rollouts
        }
        protein_rows.append(
            {
                "dataset": rec["dataset"],
                "stem": rec["stem"],
                "eval_set": rec["eval_set"],
                "L": rec["L"],
                "n_rollouts": n_rollouts,
                "oracle_r_precision": float(np.nanmax(all_scores)),
                "oracle_long_r_precision": float(np.nanmax(long_scores)),
                "mean_r_precision": float(np.nanmean(all_scores)),
                "mean_long_r_precision": float(np.nanmean(long_scores)),
                "consensus_r_precision": consensus["all"],
                "consensus_long_r_precision": consensus["long"],
                "oracle_rollout": int(np.nanargmax(all_scores)),
                "mean_contacts": float(np.mean([len(x) for x in ordered])),
                "unique_maps": len({frozenset(x) for x in contact_sets}),
                "mean_pairwise_jaccard": _mean_jaccard(contact_sets),
                "n_true": counts["all"],
                "oracle_curve": curve,
                "inference_seconds": elapsed,
            }
        )
    return protein_rows, rollout_rows


def summarize(protein_rows):
    if not protein_rows:
        raise ValueError("Cannot summarize an empty evaluation")

    def mean(key):
        values = np.asarray([row[key] for row in protein_rows], dtype=float)
        return float(np.nanmean(values))

    curve_keys = sorted(
        {int(n) for row in protein_rows for n in row["oracle_curve"]},
    )
    return {
        "proteins": len(protein_rows),
        "oracle_r_precision": mean("oracle_r_precision"),
        "oracle_long_r_precision": mean("oracle_long_r_precision"),
        "mean_r_precision": mean("mean_r_precision"),
        "mean_long_r_precision": mean("mean_long_r_precision"),
        "consensus_r_precision": mean("consensus_r_precision"),
        "consensus_long_r_precision": mean("consensus_long_r_precision"),
        "mean_contacts": mean("mean_contacts"),
        "mean_unique_maps": mean("unique_maps"),
        "mean_pairwise_jaccard": mean("mean_pairwise_jaccard"),
        "inference_seconds": float(
            sum(row["inference_seconds"] for row in protein_rows)
        ),
        "oracle_curve": {
            str(n): float(
                np.nanmean([row["oracle_curve"][str(n)] for row in protein_rows])
            )
            for n in curve_keys
        },
    }
