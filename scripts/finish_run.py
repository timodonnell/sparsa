"""Resume the local training -> batch evaluation -> CPU recovery handoff.

This coordinator submits no training jobs and never evaluates an unfinished run.
Its journal and recovered weights belong under the ignored outputs directory.
Run with the same --name to resume an interrupted coordinator without submitting
duplicate evaluation jobs. A failed training/evaluation job requires inspection.
"""

import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--profile-job")
    parser.add_argument("--iris", default=".tools/iris/bin/iris")
    parser.add_argument(
        "--cluster-config",
        default="/home/bizon/git/marin-freshiris/lib/iris/config/cw-rno2a.yaml",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{1,35}", args.name):
        raise ValueError("Expected a short Sparsa run name")
    if args.profile_job and not args.profile_job.startswith("/bizon/sparsa-"):
        raise ValueError("Expected a Sparsa profiling job")
    root = Path(__file__).resolve().parents[1]
    local = root / "outputs" / args.name
    local.mkdir(parents=True, exist_ok=True)
    lock = (local / "handoff.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    journal = local / "handoff.json"
    state = json.loads(journal.read_text()) if journal.exists() else {"name": args.name}
    iris = [str(root / args.iris), "--config", args.cluster_config]
    kube = [
        "kubectl",
        "--kubeconfig",
        str(Path.home() / ".kube/coreweave-iris"),
        "--context",
        "marin-rn02a_RNO2A",
        "-n",
        "iris",
    ]
    base = "s3://marin-us-east-02a/marin/protein-structure/sparsa"
    evaluation = f"{base}/evaluations/{args.name}"

    def record(stage, **extra):
        state.update(stage=stage, updated_utc=datetime.now(UTC).isoformat(), **extra)
        temporary = journal.with_suffix(".partial")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(journal)
        print(json.dumps(state), flush=True)

    def command(argv, **kwargs):
        return subprocess.run(
            argv, cwd=root, check=True, text=True, capture_output=True, **kwargs
        ).stdout

    def wait_job(job):
        # Iris follows task retries/preemptions. Reconnect if only the client
        # tunnel failed; do not relaunch a terminally failed remote job.
        for attempt in range(6):
            try:
                command(iris + ["job", "wait", job])
                return
            except subprocess.CalledProcessError:
                description = command(iris + ["job", "describe", job])
                if "State: succeeded" in description:
                    return
                if re.search(r"State: (failed|cancelled|canceled|killed)", description):
                    raise RuntimeError(description)
                if attempt == 5:
                    raise
                time.sleep(30)

    def submit(label, gpu, script_args):
        job = f"/bizon/sparsa-{label}-{args.name}"
        try:
            command(iris + ["job", "describe", job])
            return job
        except subprocess.CalledProcessError as error:
            detail = (error.stdout + error.stderr).lower()
            if "not found" not in detail and "not_found" not in detail:
                raise
        argv = iris + [
            "job",
            "run",
            "--priority",
            "batch",
            "--enable-extra-resources",
            "--cpu",
            str(max(8, gpu * 6)) if gpu else "2",
            "--memory",
            f"{gpu * 32}GB" if gpu else "8GB",
            "--disk",
            "40GB" if gpu > 1 else "20GB",
            "--timeout",
            "14400" if gpu > 1 else "3600" if gpu else "1800",
            "--max-retries",
            "3",
            "--job-name",
            job.split("/")[-1],
            "--no-wait",
        ]
        if gpu:
            argv += ["--gpu", f"H100x{gpu}"]
        argv += ["--", "python", *script_args]
        output = command(argv)
        if job not in output:
            raise RuntimeError(f"Unexpected submission response: {output}")
        (root / "reports" / "jobs" / f"{label}-{args.name}.json").write_text(
            json.dumps(
                {
                    "job": job,
                    "command": argv,
                    "priority": "batch",
                    "submitted_utc": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
        )
        return job

    try:
        if state.get("stage") == "recovered":
            print("Artifacts already recovered:", state["artifacts"], flush=True)
            return
        record("waiting_for_training")
        wait_job(f"/bizon/sparsa-{args.name}")
        if args.profile_job:
            record("waiting_for_long_profile", profile_job=args.profile_job)
            wait_job(args.profile_job)
        long_run = f"{base}/runs/{args.name}-long"
        job = submit(
            "long",
            8,
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                "scripts/train_from_best.py",
                "--source-run",
                f"{base}/runs/{args.name}",
                "--config",
                "configs/long_finetune.yaml",
                "--out",
                long_run,
                "--auto-resume",
            ],
        )
        record("waiting_for_long_finetune", finetune_job=job, finetune_uri=long_run)
        wait_job(job)
        job = submit(
            "final",
            1,
            [
                "scripts/final_evaluate.py",
                "--run",
                f"{base}/runs/{args.name}",
                "--candidate-run",
                long_run,
                "--out",
                evaluation,
            ],
        )
        record("waiting_for_evaluation", evaluation_job=job, evaluation_uri=evaluation)
        wait_job(job)
        job = submit(
            "recover",
            0,
            [
                "scripts/recover_results.py",
                "--source",
                evaluation,
            ],
        )
        record("recovering", recovery_job=job)
        pod = None
        for _ in range(120):
            response = json.loads(
                command(
                    kube
                    + [
                        "get",
                        "pods",
                        "-l",
                        "iris.job_id=" + job.strip("/").replace("/", "."),
                        "-o",
                        "json",
                    ]
                )
            )
            running = [
                p for p in response["items"] if p["status"]["phase"] == "Running"
            ]
            if running:
                pod = max(running, key=lambda p: p["metadata"]["creationTimestamp"])[
                    "metadata"
                ]["name"]
                try:
                    logs = command(kube + ["logs", pod, "-c", "task", "--tail=5"])
                    if "READY /app/outputs/recovered" in logs:
                        break
                except subprocess.CalledProcessError:
                    pass  # Container is still starting.
            time.sleep(10)
        else:
            raise TimeoutError(
                "Recovery task did not stage its artifacts in 20 minutes"
            )
        command(
            [
                sys.executable,
                "scripts/fetch_results.py",
                "--pod",
                pod,
                "--out",
                str(local),
            ]
        )
        artifacts = local / "sparsa-results"
        hashes = json.loads((artifacts / "SHA256SUMS.json").read_text())
        for name, expected in hashes.items():
            path = artifacts / name
            with path.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError(f"Recovered artifact checksum mismatch: {name}")
        command(iris + ["job", "cancel", job])
        record("recovered", artifacts=str(artifacts), verified_files=len(hashes))
    except Exception as error:
        record("needs_inspection", error=str(error))
        raise


if __name__ == "__main__":
    main()
