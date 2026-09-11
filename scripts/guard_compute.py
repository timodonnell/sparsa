"""Watch an explicit campaign job list and stop jobs at its running GPU budget.

Run locally with a process supervisor. The config can add newly submitted jobs
atomically. No jobs are submitted by this guard. Unknown stopped-attempt timing
uses the job's whole elapsed lifetime as a conservative bound for enforcement;
the underlying accounting report retains the measured/estimated distinction.
"""

import argparse
import fcntl
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

TERMINAL = {"succeeded", "failed", "killed", "cancelled", "canceled"}


def conservative_hours(report):
    total = 0.0
    for job in report["jobs"]:
        attempts = [a for t in job["tasks"] for a in t["attempts"]]
        if any(a["running_gpu_hours"] is None for a in attempts):
            total += (
                job["wall_seconds"] / 3600 * sum(t["gpu_count"] for t in job["tasks"])
            )
        else:
            total += sum(a["running_gpu_hours"] for a in attempts)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "guard.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        config = json.loads(Path(args.config).read_text())
        jobs = config["jobs"]
        if (
            not jobs
            or len(set(jobs)) != len(jobs)
            or any(not j.startswith("/bizon/sparsa-") for j in jobs)
        ):
            raise ValueError("Expected distinct, explicitly listed Sparsa jobs")
        limit = float(config["max_h100_hours"])
        if not 0 < limit <= 600:
            raise ValueError("This campaign has an initial maximum of 600 H100-hours")
        iris = str(root / ".tools/iris/bin/iris")
        cluster = "/home/bizon/git/marin-freshiris/lib/iris/config/cw-rno2a.yaml"
        command = [
            str(root / ".tools/iris/bin/python"),
            "scripts/job_accounting.py",
            "--out",
            str(out / "accounting.json"),
        ]
        for job in jobs:
            command.extend(["--job", job])
        try:
            subprocess.run(
                command,
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            report = json.loads((out / "accounting.json").read_text())
            charged = conservative_hours(report)
            active = [j["job"] for j in report["jobs"] if j["state"] not in TERMINAL]
            status = {
                "observed_utc": datetime.now(UTC).isoformat(),
                "running_h100_hours": report["running_gpu_hours"],
                "enforcement_h100_hours": charged,
                "max_h100_hours": limit,
                "active_jobs": active,
                "state": "monitoring" if active else "all_terminal",
            }
            if charged >= limit and active:
                active_gpu = [
                    j["job"]
                    for j in report["jobs"]
                    if j["job"] in active and any(t["gpu_count"] for t in j["tasks"])
                ]
                for job in active_gpu:
                    subprocess.run(
                        [iris, "--config", cluster, "job", "cancel", job],
                        cwd=root,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                status["state"] = (
                    "budget_stop_requested" if active_gpu else "cpu_recovery_only"
                )
            (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            print(json.dumps(status), flush=True)
            if not active and not config.get("keep_watching", False):
                return
        except (subprocess.SubprocessError, OSError) as error:
            # Retry connection failures without revealing commands or credentials.
            print(
                json.dumps(
                    {
                        "observed_utc": datetime.now(UTC).isoformat(),
                        "state": "accounting_retry",
                        "error_type": type(error).__name__,
                    }
                ),
                flush=True,
            )
        time.sleep(60)


if __name__ == "__main__":
    main()
