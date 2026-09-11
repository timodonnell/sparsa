"""One immutable Iris trial: fresh/resumed training then fixed-endpoint validation."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from research.core import digest
from sparsa.cli import load_model
from sparsa.data import benchmark
from sparsa.evaluate import evaluate
from sparsa.model import ContactModel, ModelConfig
from sparsa.train import storage, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="research_trial.json")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    out = spec["out"]
    fs, root = storage(out)
    if fs.exists(root + "/result.json"):
        print(
            "RESEARCH_RESULT " + fs.cat_file(root + "/result.json").decode(), flush=True
        )
        return
    for name, sha in spec["files"].items():
        if digest(Path(name).read_bytes()) != sha:
            raise ValueError(f"Trial source mismatch: {name}")
    # A persistent deadline includes queue gaps after preemption. It bounds the
    # resumed worker's lifespan conservatively instead of resetting its budget.
    if not fs.exists(root + "/deadline.json"):
        write_json(
            {"deadline": time.time() + spec["timeout_seconds"]}, out + "/deadline.json"
        )
    deadline = json.loads(fs.cat_file(root + "/deadline.json"))["deadline"]
    if time.time() >= deadline:
        raise TimeoutError("Trial lifetime exhausted across batch preemptions")
    cfg = yaml.safe_load(Path("configs/research_trial.yaml").read_text())
    torch.set_num_threads(4)
    # Fail an oversized architecture before spending its training allocation.
    model = ContactModel(ModelConfig(**cfg["model"])).cuda().eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        value = model(torch.ones(1, 920, dtype=torch.long, device="cuda"))
        if value.shape != (1, 920, 920) or not torch.isfinite(value).all():
            raise ValueError("Full-length GPU contract failed")
    del model, value
    torch.cuda.empty_cache()
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={spec['gpus']}",
        "-m",
        "sparsa.train",
        "--config",
        "configs/research_trial.yaml",
        "--out",
        out,
        "--auto-resume",
    ]
    process = subprocess.Popen(argv, start_new_session=True)
    try:
        process.wait(timeout=max(1, deadline - time.time() - 120))
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"Training exited {process.returncode}")
    completion = json.loads(fs.cat_file(root + "/complete.json"))
    provenance = json.loads(fs.cat_file(root + "/provenance.json"))
    selected = json.loads(fs.cat_file(root + "/latest.json"))
    if completion["step"] != cfg["steps"] or selected["step"] != cfg["steps"]:
        raise ValueError("Only the completed fixed endpoint is comparable")
    model, state = load_model(selected["checkpoint"], torch.device("cuda"))
    local = Path("outputs/research_evaluation")
    metrics = evaluate(
        model,
        benchmark(splits=("eval-val",)),
        torch.device("cuda"),
        local,
        label=spec["trial_id"],
        pos_weight=cfg["pos_weight"],
    )
    frame = pd.read_csv(local / "per_protein.csv")
    per_protein = {}
    for name in ("all", "long"):
        rows = frame[(frame["range"] == name) & (frame["cut"] == "R")]
        if len(rows) != 97 or rows.stem.nunique() != 97:
            raise ValueError("Unexpected validation coverage")
        per_protein[name] = dict(zip(rows.stem, rows.precision, strict=True))
    curve = sorted(
        [json.loads(fs.cat_file(p)) for p in fs.glob(root + "/validation/step-*.json")],
        key=lambda row: row["step"],
    )
    training_log = json.loads(fs.cat_file(root + "/training_log.json"))
    result = {
        **metrics,
        "trial_id": spec["trial_id"],
        "candidate_id": spec["candidate_id"],
        "seed": cfg["seed"],
        "step": state["step"],
        "split": "eval-val",
        "parameters": sum(p.numel() for p in model.parameters()),
        "contract": spec["contract"],
        "per_protein": per_protein,
        "validation_curve": curve,
        "recent_training": training_log[-5:],
        "checkpoint": selected["checkpoint"],
        "files": spec["files"],
        "teacher_inventory_sha256": digest(
            json.dumps(provenance["source_files"], sort_keys=True).encode()
        ),
        "selection": "fixed final EMA checkpoint; original readout; no test/de novo evaluation",
    }
    for name in ("per_protein.csv", "summary.csv", "timings.csv"):
        fs.put_file(str(local / name), root + "/evaluation/" + name)
    fs.put_file("sparsa/model.py", root + "/model.py")
    write_json(spec, out + "/trial.json")
    write_json(result, out + "/result.json")
    print("RESEARCH_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
