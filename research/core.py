"""Pure protocol, provenance, and promotion logic; no cluster side effects."""

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identity(source, model_config):
    return digest(
        source.encode() + b"\0" + json.dumps(model_config, sort_keys=True).encode()
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_protocol(p):
    if p["gpus"] not in (1, 2, 4, 8):
        raise ValueError("Expected 1, 2, 4, or 8 H100s")
    if len(p["seeds"]) != 2 or len(set(p["seeds"])) != 2:
        raise ValueError("Two distinct matched seeds are required")
    if any(not isinstance(s, int) or not 0 <= s < 2**31 for s in p["seeds"]):
        raise ValueError("Seeds must be nonnegative integers below 2**31")
    for key in (
        "gpu_hours",
        "trial_timeout_seconds",
        "max_candidates",
        "max_parameters",
        "proposer_timeout_seconds",
        "poll_seconds",
        "bootstrap_samples",
    ):
        if not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 5 <= p["poll_seconds"] <= 60:
        raise ValueError("Poll interval must be 5–60 seconds")
    if any(
        not math.isfinite(p[k]) or p[k] < 0
        for k in ("minimum_gain", "maximum_long_regression")
    ):
        raise ValueError("Invalid promotion thresholds")
    cfg = p["training"]
    for key in (
        "steps",
        "batch_size",
        "accumulation",
        "crop",
        "eval_every",
        "checkpoint_every",
    ):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"Invalid training {key}")
    if any(
        k in cfg
        for k in (
            "init_from",
            "resume",
            "initialize_max_distance",
            "limit_shards",
            "data_root",
        )
    ):
        raise ValueError("Search uses fresh weights and the full frozen teacher corpus")
    if p["gpu_hours"] < 2 * reservation(p):
        raise ValueError("Budget must reserve both baseline seeds")


def reservation(protocol):
    """Conservative per-trial allocation; failed trials also consume a slot."""
    return protocol["gpus"] * protocol["trial_timeout_seconds"] / 3600


def validate_source(source, reference=None):
    """Reject accidental I/O/contract changes, not a security sandbox for hostile code."""
    if len(source.encode()) > 150_000:
        raise ValueError("Model source exceeds 150 KB")
    tree = ast.parse(source)
    if reference is not None:

        def tokenizer_contract(module):
            nodes = []
            for node in module.body:
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "tokenize"
                    or (
                        isinstance(node, ast.Assign)
                        and any(
                            isinstance(t, ast.Name)
                            and t.id in {"ALPHABET", "TOKEN_IDS"}
                            for t in node.targets
                        )
                    )
                ):
                    nodes.append(ast.dump(node, include_attributes=False))
            return nodes

        if tokenizer_contract(tree) != tokenizer_contract(ast.parse(reference)):
            raise ValueError("Tokenizer and amino-acid vocabulary are frozen")
    allowed = {"torch", "math", "dataclasses", "typing", "functools"}
    forbidden = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "breakpoint",
        "input",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            n.name.split(".")[0] not in allowed for n in node.names
        ):
            raise ValueError("Model imports must be tensor/math utilities only")
        if isinstance(node, ast.ImportFrom) and (
            node.level
            or (
                (node.module or "").split(".")[0] not in allowed
                and not (
                    node.module == "sparsa.pair_trunk"
                    and all(
                        n.name in {"ResidueToPair", "ReasoningBlock"}
                        for n in node.names
                    )
                )
            )
        ):
            raise ValueError("Model imports must be tensor/math utilities only")
        if isinstance(node, ast.Name) and node.id in forbidden:
            raise ValueError(f"Forbidden model operation: {node.id}")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__")
            and node.attr != "__init__"
            or node.attr
            in {
                "load",
                "save",
                "hub",
                "distributed",
                "from_file",
                "frombuffer",
                "set_default_device",
                "set_default_dtype",
                "manual_seed",
            }
        ):
            raise ValueError(f"Forbidden model operation: {node.attr}")
    return tree


def validate_result(result, trial, protocol, contract):
    for key, value in {
        "trial_id": trial["id"],
        "candidate_id": trial["candidate"],
        "seed": trial["seed"],
        "step": protocol["training"]["steps"],
        "contract": contract,
        "split": "eval-val",
        "proteins": 97,
    }.items():
        if result.get(key) != value:
            raise ValueError(f"Result mismatch: {key}")
    for name in ("all", "long"):
        scores = result["per_protein"][name]
        if set(scores) != set(contract["validation_stems"]):
            raise ValueError("Incomplete or unexpected validation proteins")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores.values()):
            raise ValueError("Invalid R-precision")
        metric = "r_precision" if name == "all" else "long_r_precision"
        if (
            not math.isfinite(result[metric])
            or abs(np.mean(list(scores.values())) - result[metric]) > 1e-12
        ):
            raise ValueError("Aggregate does not match per-protein scores")
    if not 0 < result["parameters"] <= protocol["max_parameters"]:
        raise ValueError("Parameter budget exceeded")


def promotion(candidate, parent, protocol):
    """Pair proteins and seeds; bootstrap proteins after averaging matched seeds."""
    if set(candidate) != set(protocol["seeds"]) or set(parent) != set(candidate):
        raise ValueError("Promotion requires both matched seeds")
    diffs, long_diffs, seed_gains = [], [], []
    stems = sorted(candidate[protocol["seeds"][0]]["per_protein"]["all"])
    for seed in protocol["seeds"]:
        for result in (candidate[seed], parent[seed]):
            if any(
                set(result["per_protein"][r]) != set(stems) for r in ("all", "long")
            ):
                raise ValueError("Paired protein coverage mismatch")
        delta = np.array(
            [
                candidate[seed]["per_protein"]["all"][s]
                - parent[seed]["per_protein"]["all"][s]
                for s in stems
            ]
        )
        diffs.append(delta)
        seed_gains.append(float(delta.mean()))
        long_diffs.append(
            [
                candidate[seed]["per_protein"]["long"][s]
                - parent[seed]["per_protein"]["long"][s]
                for s in stems
            ]
        )
    paired = np.mean(diffs, axis=0)
    rng = np.random.default_rng(20260911)
    draws = paired[
        rng.integers(0, len(stems), size=(protocol["bootstrap_samples"], len(stems)))
    ].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    gain, long_gain = float(paired.mean()), float(np.mean(long_diffs))
    accepted = (
        all(d > 0 for d in seed_gains)
        and gain >= protocol["minimum_gain"]
        and low > 0
        and long_gain >= -protocol["maximum_long_regression"]
    )
    return {
        "accepted": bool(accepted),
        "mean_gain": gain,
        "long_gain": long_gain,
        "seed_gains": seed_gains,
        "paired_ci95": [float(low), float(high)],
        "interpretation": "Validation search evidence; adaptive selection makes this interval descriptive, not confirmatory.",
    }
