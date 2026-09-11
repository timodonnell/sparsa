"""Read only this campaign's pod logs and preserve validation/checkpoint evidence."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", default="outputs/scaling-v2/budget.json")
    parser.add_argument("--out", default="reports/scaling_v2/progress.json")
    args = parser.parse_args()
    jobs = json.loads(Path(args.budget).read_text())["jobs"]
    if not jobs or any(not j.startswith("/bizon/sparsa-") for j in jobs):
        raise ValueError("Expected an explicit Sparsa job list")
    kube = [
        "kubectl",
        "--kubeconfig",
        "/home/bizon/.kube/coreweave-iris",
        "--context",
        "marin-rn02a_RNO2A",
        "-n",
        "iris",
    ]

    def run(cmd):
        return subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=60
        ).stdout

    labels = [j.strip("/").replace("/", ".") for j in jobs]
    pods = json.loads(
        run(
            kube
            + [
                "get",
                "pods",
                "-l",
                "iris.job_id in (" + ",".join(labels) + ")",
                "-o",
                "json",
            ]
        )
    )["items"]
    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {"jobs": {}}
    present_jobs = {
        "/" + p["metadata"]["labels"]["iris.job_id"].replace(".", "/", 1) for p in pods
    }
    for job, state in report["jobs"].items():
        if job not in present_jobs:
            state["pod_phase"] = "Absent"
    for pod in sorted(pods, key=lambda p: p["metadata"]["creationTimestamp"]):
        label = pod["metadata"]["labels"]["iris.job_id"]
        job = jobs[labels.index(label)]
        name = pod["metadata"]["name"]
        try:
            logs = run(kube + ["logs", name, "-c", "task"])
        except subprocess.CalledProcessError:
            continue
        state = report["jobs"].get(job, {"validation": [], "checkpoints": []})
        validated = {r["step"]: r for r in state["validation"]}
        checkpoints = {r["step"]: r for r in state["checkpoints"]}
        for line in logs.splitlines():
            if line.startswith("VALIDATION "):
                row = json.loads(line.removeprefix("VALIDATION "))
                validated[row["step"]] = row
            elif line.startswith("CHECKPOINT "):
                row = json.loads(line.removeprefix("CHECKPOINT "))
                checkpoints[row["step"]] = row
            elif line.startswith('{"step":'):
                state["latest_training"] = json.loads(line)
            elif line.startswith('{"model_config":'):
                row = json.loads(line)
                if "training_config" in row:
                    state["provenance"] = row
        if "provenance" not in state:
            continue
        state.update(
            pod=name,
            pod_phase=pod["status"]["phase"],
            validation=sorted(validated.values(), key=lambda r: r["step"]),
            checkpoints=sorted(checkpoints.values(), key=lambda r: r["step"]),
        )
        report["jobs"][job] = state
    report["observed_utc"] = datetime.now(UTC).isoformat()
    report["scope"] = (
        "Training logs and successful checkpoint-write messages; selection uses durable S3 pointers after completed pilots. Validation only."
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(out)
    print(
        json.dumps(
            {
                name: {
                    "step": r.get("latest_training", {}).get("step"),
                    "validation": r["validation"][-1:],
                }
                for name, r in report["jobs"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
