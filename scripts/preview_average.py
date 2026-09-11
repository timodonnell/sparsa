"""Validation-only checkpoint averaging experiment; never loads held-out splits."""

import argparse
import json
import time

import torch

from scripts.average_checkpoints import average_ema
from sparsa.cli import load_model
from sparsa.data import benchmark
from sparsa.evaluate import evaluate
from sparsa.model import ContactModel, ModelConfig
from sparsa.train import save_checkpoint, storage, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--max-step", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    fs, root = storage(args.run)
    history = [
        json.loads(fs.cat_file(p)) for p in fs.glob(root + "/validation/step-*.json")
    ]
    saved = sorted(
        [
            r
            for r in history
            if r["step"] <= args.max_step
            and fs.exists(root + f"/checkpoints/step-{r['step']}.pt")
        ],
        key=lambda r: (-r["r_precision"], r["step"]),
    )
    if len(saved) < 5:
        raise ValueError("This experiment requires five saved validation checkpoints")
    validation = benchmark(splits=("eval-val",))
    started = time.perf_counter()
    candidates = []

    def score(model, state, uri, cap, kind):
        model.relative_max_distance = cap
        metrics = evaluate(
            model,
            validation,
            device,
            pos_weight=state["training_config"].get("pos_weight", 1.0),
        )
        row = dict(
            kind=kind,
            checkpoint=uri,
            step=state["step"],
            relative_max_distance=cap,
            **metrics,
        )
        if "averaged_checkpoints" in state:
            row["averaged_checkpoints"] = state["averaged_checkpoints"]
        candidates.append(row)
        print("AVERAGE_VALIDATION " + json.dumps(row), flush=True)

    for index, rec in enumerate(saved):
        uri = args.run.rstrip("/") + f"/checkpoints/step-{rec['step']}.pt"
        model, state = load_model(uri, device)
        if index == 0:
            score(model, state, uri, None, "single")
        score(model, state, uri, state["training_config"]["crop"] - 1, "single")
        del model, state
    for count in (2, 3, 5):
        sources = [
            args.run.rstrip("/") + f"/checkpoints/step-{r['step']}.pt"
            for r in saved[:count]
        ]
        state = average_ema(sources)
        uri = args.out.rstrip("/") + f"/checkpoints/top-{count}.pt"
        save_checkpoint(uri, state)
        model = ContactModel(ModelConfig(**state["model_config"])).to(device).eval()
        model.load_state_dict(state["ema"])
        for cap in (None, state["training_config"]["crop"] - 1):
            score(model, state, uri, cap, "average")
        del model, state
    best_single = max(
        (r for r in candidates if r["kind"] == "single"), key=lambda r: r["r_precision"]
    )
    best_average = max(
        (r for r in candidates if r["kind"] == "average"),
        key=lambda r: r["r_precision"],
    )
    result = {
        "run": args.run,
        "max_step": args.max_step,
        "split": "eval-val",
        "method": "Uniform EMA weight averages of the top 2, 3, and 5 checkpoints by original validation R-precision; original and capped readouts. Controls: best original readout and every capped checkpoint through max_step.",
        "candidates": candidates,
        "best_single": best_single,
        "best_average": best_average,
        "average_improves": best_average["r_precision"] > best_single["r_precision"],
        "seconds": time.perf_counter() - started,
    }
    write_json(result, args.out.rstrip("/") + "/result.json")
    print(
        "AVERAGE_RESULT "
        + json.dumps(
            {
                k: result[k]
                for k in ("best_single", "best_average", "average_improves", "seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
