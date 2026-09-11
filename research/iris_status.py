"""Structured read-only Iris status, run with the isolated Iris Python."""

import argparse
import json
import time
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.cluster.types import JobName

from scripts.job_accounting import attempt_timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    if not args.job.startswith("/bizon/sparsa-ar-"):
        raise ValueError("Only this search system's jobs may be inspected")
    with open_iris_client(config_file=Path(args.config), workspace=None) as client:
        try:
            name = JobName.from_wire(args.job)
            job = client.job_status(name)
        except Exception as error:
            if "NOT_FOUND" in str(error) or "not found" in str(error).lower():
                print(json.dumps({"state": "not_found"}))
                return
            raise
        attempts, hours, unknown = [], 0.0, 0
        for task in client.list_tasks(name):
            description = client.describe_task(task.task_id)
            device = description.resources.device
            gpus = device.count if device and device.kind.value == "gpu" else 0
            last = max(
                (a.attempt_number for a in description.status.attempts), default=None
            )
            for a in description.status.attempts:
                timing = attempt_timing(
                    start_ms=a.started_at.epoch_ms() if a.started_at else None,
                    finish_ms=a.finished_at.epoch_ms() if a.finished_at else None,
                    state=a.state.value,
                    observed_ms=int(time.time() * 1000),
                    job_finish_ms=job.finished_at.epoch_ms()
                    if job.finished_at and a.attempt_number == last
                    else None,
                )
                seconds = timing["running_seconds"]
                unknown += seconds is None
                hours += (seconds or 0) * gpus / 3600
                attempts.append(
                    {
                        "attempt": a.attempt_number,
                        "state": a.state.value,
                        "gpus": gpus,
                        "node": a.node_name,
                        "terminal_reason": a.terminal_reason,
                        **timing,
                    }
                )
        print(
            json.dumps(
                {
                    "state": job.state.value,
                    "gpu_hours": hours,
                    "unknown_attempts": unknown,
                    "attempts": attempts,
                    "preemptions": job.preemption_count,
                    "failures": job.failure_count,
                }
            )
        )


if __name__ == "__main__":
    main()
