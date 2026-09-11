"""Evaluate the validation-selected checkpoint once on the frozen experimental sets.

Run as a batch-priority Iris GPU task after training completes. It writes full
per-protein results, paired baseline comparisons, and a lean inference checkpoint
back to the same object store. Experimental labels never choose a contact budget.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from scripts.compare import compare
from scripts.select_readout import select_readout
from sparsa.cli import export_contacts, load_model
from sparsa.data import benchmark
from sparsa.evaluate import evaluate
from sparsa.model import tokenize
from sparsa.train import storage, write_json


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def export_training_metadata(run, local):
    fs, root = storage(run)
    local.mkdir(parents=True, exist_ok=True)
    for source, target in {
        "provenance.json": "training_provenance.json",
        "training_log.json": "training_log.json",
        "complete.json": "training_complete.json",
    }.items():
        (local / target).write_bytes(fs.cat_file(root + "/" + source))
    resumes = [
        json.loads(fs.cat_file(p))
        for p in fs.glob(root + "/resume-step-*-provenance.json")
    ]
    (local / "training_resumes.json").write_text(
        json.dumps(sorted(resumes, key=lambda r: r["started_utc"]), indent=2)
    )
    history = [
        json.loads(fs.cat_file(p)) for p in fs.glob(root + "/validation/step-*.json")
    ]
    (local / "training_validation.json").write_text(
        json.dumps(sorted(history, key=lambda r: r["step"]), indent=2)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidate-run", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = [args.run, *args.candidate_run]
    for run in runs:
        fs, root = storage(run)
        if not fs.exists(root + "/complete.json"):
            raise RuntimeError(
                "Every candidate run must complete before held-out evaluation"
            )
        completion = json.loads(fs.cat_file(root + "/complete.json"))
        selection = json.loads(fs.cat_file(root + "/best.json"))
        if (
            abs(
                selection["validation"]["r_precision"]
                - completion["best_validation_r_precision"]
            )
            > 1e-12
        ):
            raise RuntimeError(
                "Best-checkpoint pointer is stale; inspect checkpoint recovery"
            )
    torch.set_num_threads(4)
    device = torch.device("cuda")
    run_readouts = [dict(run=run, **select_readout(run, device)) for run in runs]
    readout = max(
        run_readouts, key=lambda r: r["selected"]["validation"]["r_precision"]
    )
    selection = readout["selected"]
    inference_config = readout["inference_config"]
    start = time.perf_counter()
    model, state = load_model(selection["checkpoint"], device)
    model.relative_max_distance = inference_config["relative_max_distance"]
    load_seconds = time.perf_counter() - start
    records = benchmark(splits=("eval-val", "eval-test", "eval-denovo"))
    local = Path("outputs/final_evaluation")
    local.mkdir(parents=True, exist_ok=True)
    (local / "readout_selection.json").write_text(json.dumps(readout, indent=2))
    (local / "candidate_run_selection.json").write_text(
        json.dumps(run_readouts, indent=2)
    )
    result = evaluate(
        model,
        records,
        device,
        local,
        label=f"sparsa-{readout['run'].rsplit('/', 1)[-1]}-step-{state['step']}",
        pos_weight=state["training_config"].get("pos_weight", 1.0),
    )
    paired = compare(
        local / "per_protein.csv",
        "data/benchmark/marinfold_baselines.csv",
        "data/benchmark/eval_sets.csv",
    )
    paired.to_csv(local / "paired_comparisons.csv", index=False)
    compare(
        local / "per_protein.csv",
        "data/benchmark/marinfold_step363000.csv",
        "data/benchmark/eval_sets.csv",
        splits=("eval-val", "eval-denovo"),
    ).to_csv(local / "paired_comparisons_step363000.csv", index=False)
    lean = {
        k: state[k]
        for k in [
            "format_version",
            "alphabet",
            "model_config",
            "training_config",
            "step",
            "ema",
        ]
    }
    lean["source_checkpoint"] = selection["checkpoint"]
    lean["validation_selection"] = selection
    lean["inference_config"] = inference_config
    torch.save(lean, local / "sparsa.pt")
    (local / "model_config.json").write_text(
        json.dumps(state["model_config"], indent=2)
    )
    (local / "vocabulary.json").write_text(
        json.dumps(
            {
                "padding": 0,
                "amino_acids": {aa: i + 1 for i, aa in enumerate(state["alphabet"])},
            },
            indent=2,
        )
    )
    # A Helico-ready example taken from validation, with a fixed top-L budget.
    rec = next(r for r in records if r["eval_set"] == "eval-val")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        scores = (
            model(tokenize(rec["sequence"])[None].to(device))[0]
            .float()
            .sigmoid()
            .cpu()
            .numpy()
        )
    export_contacts(scores, local / "example_helico_contacts.txt", rec["L"])
    (local / "example.fasta").write_text(f">{rec['stem']}\n{rec['sequence']}\n")
    result.update(
        selected_checkpoint=selection,
        load_seconds=load_seconds,
        proteins=len(records),
        test_used_for_selection=False,
        gpu=torch.cuda.get_device_name(),
        model_parameters=sum(p.numel() for p in model.parameters()),
        inference_config=inference_config,
        selection_seconds=sum(r["selection_seconds"] for r in run_readouts),
        selected_training_run=readout["run"],
        training_runs={run: f"training_runs/run-{i}" for i, run in enumerate(runs)},
        evaluation_source_sha256={
            str(path): sha256(path)
            for path in [
                *sorted(Path("sparsa").rglob("*.py")),
                Path(__file__),
                Path("scripts/compare.py"),
                Path("scripts/select_readout.py"),
            ]
        },
        benchmark_sha256={
            path.name: sha256(path)
            for path in sorted(Path("data/benchmark").glob("*"))
            if path.is_file()
        },
    )
    (local / "evaluation_manifest.json").write_text(json.dumps(result, indent=2))
    export_training_metadata(readout["run"], local)
    for i, run in enumerate(runs):
        export_training_metadata(run, local / "training_runs" / f"run-{i}")
    hashes = {
        str(path.relative_to(local)): sha256(path)
        for path in sorted(local.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    (local / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2))
    destination_fs, destination = storage(args.out)
    for path in sorted(local.rglob("*")):
        if path.is_file():
            destination_fs.put_file(
                str(path), destination + "/" + str(path.relative_to(local))
            )
    write_json(
        {
            "complete": True,
            "selected_checkpoint": selection["checkpoint"],
            "proteins": len(records),
        },
        args.out + "/complete.json",
    )
    print("FINAL_EVALUATION " + json.dumps(result), flush=True)
    print(
        paired[(paired["range"] == "all") & (paired.cut == "R")].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
