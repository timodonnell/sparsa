"""Compare completed scaling pilots on validation only, with paired intervals."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sparsa.data import benchmark
from sparsa.evaluate import evaluate
from sparsa.model import ContactModel, ModelConfig
from sparsa.train import load_checkpoint, storage, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="name=run URI")
    parser.add_argument("--reference", required=True, help="Released checkpoint URI")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    records = benchmark()
    choices = [("release", args.reference)]
    elapsed = {}
    for value in args.run:
        name, uri = value.split("=", 1)
        fs, prefix = storage(uri)
        if not fs.exists(prefix + "/complete.json"):
            raise ValueError(f"Pilot {name} has not completed")
        complete = json.loads(fs.cat_file(prefix + "/complete.json"))
        if complete["step"] != 6000:
            raise ValueError(f"Pilot {name} did not finish the matched 6,000 steps")
        elapsed[name] = complete["elapsed_seconds"]
        for pointer in ("latest", "best"):
            state = json.loads(fs.cat_file(prefix + f"/{pointer}.json"))
            choices.append((name + "-" + pointer, state["checkpoint"]))
    cache, tables, results = {}, {}, {}
    fs, prefix = storage(args.out)
    for name, uri in choices:
        if uri in cache:
            source = cache[uri]
            tables[name] = tables[source]
            results[name] = {**results[source], "same_weights_as": source}
            continue
        state = load_checkpoint(uri)
        model = ContactModel(ModelConfig(**state["model_config"])).cuda().eval()
        model.load_state_dict(state["ema"])
        model.relative_max_distance = state.get("inference_config", {}).get(
            "relative_max_distance"
        )
        out = Path("/tmp/scaling-comparison") / name
        metrics = evaluate(
            model,
            records,
            torch.device("cuda"),
            out=out,
            label=name,
            pos_weight=state["training_config"]["pos_weight"],
        )
        tables[name] = pd.read_csv(out / "per_protein.csv")
        results[name] = {
            "checkpoint": uri,
            "step": state["step"],
            "validation": metrics,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        cache[uri] = name
        fs.pipe_file(prefix + f"/{name}.csv", (out / "per_protein.csv").read_bytes())
        print("PILOT_VALIDATION " + json.dumps({name: results[name]}), flush=True)
        del state, model
        gc.collect()
        torch.cuda.empty_cache()
    paired = []
    names = list(tables)
    rng = np.random.default_rng(17)
    for i, name in enumerate(names):
        for ref in names[:i]:
            for metric_range in ("all", "long"):
                a, b = tables[name], tables[ref]
                a = a[(a["range"] == metric_range) & (a.cut == "R")]
                b = b[(b["range"] == metric_range) & (b.cut == "R")]
                pair = a.merge(
                    b,
                    on=["dataset", "stem"],
                    suffixes=("_a", "_b"),
                    validate="one_to_one",
                )
                assert len(pair) == len(a) == len(b) == 97
                for column in ("n_candidate", "n_true", "n_top"):
                    assert np.array_equal(pair[column + "_a"], pair[column + "_b"])
                delta = (pair.precision_a - pair.precision_b).dropna().to_numpy()
                boot = delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(1)
                paired.append(
                    {
                        "model": name,
                        "reference": ref,
                        "range": metric_range,
                        "n": len(delta),
                        "delta": float(delta.mean()),
                        "ci_low": float(np.quantile(boot, 0.025)),
                        "ci_high": float(np.quantile(boot, 0.975)),
                    }
                )
    report = {
        "split": "eval-val",
        "test_used": False,
        "models": results,
        "paired": paired,
        "pilot_elapsed_seconds": elapsed,
        "scope": "Final-EMA comparisons match pilot exposure; best checkpoints are validation-selected. Intervals are descriptive under adaptive validation reuse.",
    }
    write_json(report, args.out + "/comparison.json")
    # Keep individual log records below container-runtime line-size limits.
    for row in paired:
        print("PILOT_PAIRED " + json.dumps(row), flush=True)
    print(
        "PILOT_COMPARISON "
        + json.dumps({k: v for k, v in report.items() if k != "paired"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
