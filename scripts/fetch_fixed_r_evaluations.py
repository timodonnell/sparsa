"""Recover versioned fixed-validation metrics from Iris or a dedicated node."""

import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sparsa.data import benchmark


def fetch_iris(args, report):
    if not args.pod.startswith("iris-bizon-sparsa-"):
        raise ValueError("Expected our own project task")
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
    return json.loads(result.stdout)


def fetch_dedicated(args):
    if not args.run_dir or not args.model:
        raise ValueError("Dedicated recovery requires --run-dir and --model")
    # Only the source text travels over stdin; paths are quoted Python literals,
    # never interpolated into the remote shell command. Connection details are
    # supplied by the caller and are not written to evaluation artifacts.
    remote = f"""import json
from pathlib import Path
root=Path({args.run_dir!r})
assert root.is_absolute() and (root/'provenance.json').is_file()
result=[]
for path in sorted((root/'validation-fixed-r-v2').glob('step-*.json')):
    per=path.with_name(path.stem+'-per_protein.csv')
    result.append({{'model':{args.model!r},'summary':json.loads(path.read_text()),'per_protein':per.read_text()}})
print(json.dumps(result))
"""
    result = subprocess.run(
        ["ssh", "--", args.host, "python3 -"],
        input=remote,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pod")
    source.add_argument("--host", help="Dedicated node SSH destination")
    parser.add_argument("--run-dir", help="Absolute output directory on that node")
    parser.add_argument("--model", choices=("g4-u4", "g4-l4"))
    parser.add_argument(
        "--kubeconfig", default=str(Path.home() / ".kube/coreweave-iris-rno2a")
    )
    args = parser.parse_args()
    report = Path("reports/diffusion_fixed_r")
    records = fetch_dedicated(args) if args.host else fetch_iris(args, report)
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
