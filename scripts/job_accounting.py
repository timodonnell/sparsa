"""Read Iris attempt timing without counting queue waits as GPU computation.

Run with the isolated Iris environment, e.g. .tools/iris/bin/python. Only this
project's explicitly named jobs are inspected. Durations follow Iris's running
timestamps and are not a billing statement or a measure of GPU utilization.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.cluster.types import JobName


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--cluster-config",
        default="/home/bizon/git/marin-freshiris/lib/iris/config/cw-rno2a.yaml",
    )
    args = parser.parse_args()
    if any(not job.startswith("/bizon/sparsa-") for job in args.job):
        raise ValueError("Only explicitly named Sparsa jobs may be inspected")
    now = datetime.now(UTC)
    now_ms = int(now.timestamp() * 1000)
    jobs = []
    with open_iris_client(
        config_file=Path(args.cluster_config), workspace=None
    ) as client:
        for name in args.job:
            job = client.job_status(JobName.from_wire(name))
            tasks = []
            for task in client.list_tasks(JobName.from_wire(name)):
                description = client.describe_task(task.task_id)
                device = description.resources.device
                gpu_count = device.count if device and device.kind.value == "gpu" else 0
                attempts = []
                for attempt in description.status.attempts:
                    start = (
                        attempt.started_at.epoch_ms() if attempt.started_at else None
                    )
                    finish = (
                        attempt.finished_at.epoch_ms() if attempt.finished_at else None
                    )
                    seconds = (
                        max(0, (finish or now_ms) - start) / 1000
                        if start is not None
                        else 0
                    )
                    attempts.append(
                        {
                            "number": attempt.attempt_number,
                            "uid": attempt.attempt_uid,
                            "state": attempt.state.value,
                            "start_ms": start,
                            "finish_ms": finish,
                            "running_seconds": seconds,
                            "running_gpu_hours": seconds * gpu_count / 3600,
                            "node": attempt.node_name,
                            "terminal_reason": attempt.terminal_reason,
                        }
                    )
                tasks.append(
                    {
                        "task": task.task_id.to_wire(),
                        "state": description.status.state.value,
                        "gpu_count": gpu_count,
                        "gpu_variant": device.variant if gpu_count else None,
                        "attempts": attempts,
                    }
                )
            submitted = job.submitted_at.epoch_ms() if job.submitted_at else None
            finished = job.finished_at.epoch_ms() if job.finished_at else None
            jobs.append(
                {
                    "job": name,
                    "state": job.state.value,
                    "submitted_ms": submitted,
                    "finished_ms": finished,
                    "wall_seconds": ((finished or now_ms) - submitted) / 1000
                    if submitted
                    else None,
                    "preemptions": job.preemption_count,
                    "failures": job.failure_count,
                    "tasks": tasks,
                }
            )
    total = sum(
        a["running_gpu_hours"] for j in jobs for t in j["tasks"] for a in t["attempts"]
    )
    report = {
        "observed_utc": now.isoformat(),
        "running_gpu_hours": total,
        "jobs": jobs,
        "timing_definition": "Iris attempt started_at through finished_at (or observation time for a running attempt)",
        "limitations": "Excludes queue waits and pre-start build/allocation time; attempts without a start timestamp contribute zero. This is running resource time, not GPU utilization or billing.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"jobs": len(jobs), "running_gpu_hours": total, "out": str(out)}))


if __name__ == "__main__":
    main()
