"""Recover only versioned fixed-validation metrics via a live project Iris pod."""

import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sparsa.data import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument(
        "--kubeconfig", default=str(Path.home() / ".kube/coreweave-iris-rno2a")
    )
    args = parser.parse_args()
    if not args.pod.startswith("iris-bizon-sparsa-"):
        raise ValueError("Expected our own project task")
    report = Path("reports/diffusion_fixed_r")
    runs = json.loads((report / "restarts-20261001.json").read_text())["runs"]
    configs = [
        {"model": r["model"], "root": r["output"], "automatic": True} for r in runs
    ]
    configs.append(
        {
            "model": "g1-long",
            "root": "s3://marin-us-east-02a/marin/protein-structure/sparsa/fixed-r-v2/g1-long-step300000",
            "automatic": False,
        }
    )
    remote = """import json,sys,fsspec
fs=fsspec.filesystem("s3")
result=[]
for item in json.loads(sys.argv[1]):
    root=item["root"].removeprefix("s3://")
    summaries=fs.glob(root+"/validation-fixed-r-v2/step-*.json") if item["automatic"] else [root+"/summary.json"]
    for path in summaries:
        if not fs.exists(path): continue
        summary=json.loads(fs.cat_file(path))
        per_path=path.removesuffix(".json")+"-per_protein.csv" if item["automatic"] else root+"/per_protein.csv"
        result.append({"model":item["model"],"summary":summary,"per_protein":fs.cat_file(per_path).decode()})
print(json.dumps(result))
"""
    command = [
        "kubectl",
        "--kubeconfig",
        args.kubeconfig,
        "--context",
        "marin-rn02a_RNO2A",
        "-n",
        "iris",
        "exec",
        args.pod,
        "-c",
        "task",
        "--",
        "/app/.venv/bin/python",
        "-c",
        remote,
        json.dumps(configs),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=60
    )
    records = json.loads(result.stdout)
    expected = {(r["dataset"], r["stem"]) for r in benchmark()}
    out = report / "milestones"
    out.mkdir(exist_ok=True)
    for record in records:
        summary = record["summary"]
        if (
            summary.get("metric_version") != "fixed-r-v2"
            or summary["split"] != "eval-val"
            or summary["held_out_used"]
            or summary["proteins"] != 97
            or summary["n_rollouts"] != 100
        ):
            raise ValueError("Not a complete fixed-R validation")
        per = pd.read_csv(io.StringIO(record["per_protein"]))
        if (
            len(per) != 97
            or set(zip(per.dataset, per.stem, strict=True)) != expected
            or not per.metric_version.eq("fixed-r-v2").all()
            or not per.eval_set.eq("eval-val").all()
            or not per.n_rollouts.eq(100).all()
        ):
            raise ValueError("Mismatched per-protein records")
        for metric in [
            "oracle_r_precision",
            "consensus_r_precision",
            "sampled_oracle_r_precision",
            "probability_ensemble_r_precision",
        ]:
            np.testing.assert_allclose(per[metric].mean(), summary[metric])
        prefix = (
            f"{record['model']}-step{summary['step']}-seed{summary['rollout_seed']}"
        )
        (out / f"{prefix}-per-protein.csv").write_text(record["per_protein"])
        (out / f"{prefix}-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(
            record["model"],
            summary["step"],
            {
                m: summary[m]
                for m in [
                    "oracle_r_precision",
                    "consensus_r_precision",
                    "sampled_oracle_r_precision",
                    "probability_ensemble_r_precision",
                ]
            },
        )
    print(f"Recovered {len(records)} complete evaluations.")


if __name__ == "__main__":
    main()
