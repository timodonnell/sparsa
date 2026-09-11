"""Select checkpoint and distance readout using only experimental validation.

The training run already evaluated the original readout at each saved checkpoint.
Recheck its best checkpoint, and evaluate distance capping at every saved step.
This script never loads or scores the test/de novo splits.
"""

import argparse
import json
import time

import torch

from sparsa.cli import load_model
from sparsa.data import benchmark
from sparsa.evaluate import evaluate
from sparsa.train import storage, write_json


def select_readout(run, device):
    fs, root = storage(run)
    original = json.loads(fs.cat_file(root + "/best.json"))
    history = sorted(
        (json.loads(fs.cat_file(p)) for p in fs.glob(root + "/validation/step-*.json")),
        key=lambda r: r["step"],
    )
    # During a validation-only preview, the newest checkpoint may still be uploading.
    saved = [
        r for r in history if fs.exists(root + f"/checkpoints/step-{r['step']}.pt")
    ]
    if not saved:
        raise RuntimeError("No durable validation checkpoints")
    best_raw = max(saved, key=lambda r: r["r_precision"])
    if best_raw["r_precision"] > original["validation"]["r_precision"] + 1e-12:
        raise RuntimeError(
            "Best-checkpoint pointer is stale; retry after checkpoint upload"
        )
    validation = benchmark(splits=("eval-val",))
    started = time.perf_counter()
    candidates = []
    model, state = load_model(original["checkpoint"], device)
    metrics = evaluate(
        model,
        validation,
        device,
        pos_weight=state["training_config"].get("pos_weight", 1.0),
    )
    candidates.append(
        dict(
            checkpoint=original["checkpoint"],
            step=state["step"],
            relative_max_distance=None,
            **metrics,
        )
    )
    for rec in saved:
        uri = run.rstrip("/") + f"/checkpoints/step-{rec['step']}.pt"
        model, state = load_model(uri, device)
        cap = state["training_config"]["crop"] - 1
        model.relative_max_distance = cap
        metrics = evaluate(
            model,
            validation,
            device,
            pos_weight=state["training_config"].get("pos_weight", 1.0),
        )
        row = dict(
            checkpoint=uri, step=state["step"], relative_max_distance=cap, **metrics
        )
        candidates.append(row)
        print("READOUT_VALIDATION " + json.dumps(row), flush=True)
    chosen = max(candidates, key=lambda r: r["r_precision"])
    return {
        "split": "eval-val",
        "method": "best original readout plus capped readout at every durable checkpoint",
        "training_best_checkpoint": original["checkpoint"],
        "validation_history": history,
        "candidates": candidates,
        "selected": {
            "checkpoint": chosen["checkpoint"],
            "step": chosen["step"],
            "validation": {
                k: chosen[k] for k in ("r_precision", "long_r_precision", "proteins")
            },
        },
        "inference_config": {"relative_max_distance": chosen["relative_max_distance"]},
        "selection_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = select_readout(args.run, torch.device("cuda"))
    write_json(result, args.out)
    print("READOUT_SELECTION " + json.dumps(result["selected"]), flush=True)


if __name__ == "__main__":
    main()
