"""One-shot supervised handoff to a prepared, immutable source snapshot.

The private plan supplies connection details and commands; none are logged.
Run one instance per training run. State is durable across supervisor retries.
Existing validation must finish and a checkpoint must be fresh before stopping
the old job. The new process starts only after the old process has stopped.
"""

import argparse
import fcntl
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


def checkpoint_ready(observed, minimum_step):
    complete = observed.get("complete") or {}
    if not (
        complete.get("full_validation_proteins") == 97
        and complete.get("rollouts") == 100
        and complete.get("held_out_used") is False
    ):
        return False
    latest = observed.get("latest") or {}
    step = latest.get("step", 0)
    if step < minimum_step or not observed.get("checkpoint_exists"):
        return False
    # At the end of startup validation, step 4 is still the current state.
    if step == 4 and observed.get("complete_age", float("inf")) < 30:
        return True
    # latest.json is published after the checkpoint, before the training log.
    # Require both publications and a fresh marker to avoid replaying old work.
    return (
        observed.get("saved_log_step") == step
        and observed.get("latest_age", float("inf")) < 30
    )


OBSERVE = """
import json,time,fsspec
from datetime import datetime,timezone
fs,root=fsspec.core.url_to_fs(OUTPUT)
def read(name):
 p=root+'/'+name
 return json.loads(fs.cat_file(p)) if fs.exists(p) else None
def age(name):
 p=root+'/'+name
 if not fs.exists(p):return 1e30
 info=fs.info(p)
 t=info.get('LastModified',info.get('mtime'))
 if isinstance(t,datetime):t=t.timestamp()
 return max(0,time.time()-float(t)) if t is not None else 1e30
latest=read('latest.json')
history=read('training_log.json') or []
print(json.dumps(dict(latest=latest,complete=read('preflight/complete.json'),
 runtime=read('runtime.json'),latest_age=age('latest.json'),
 complete_age=age('preflight/complete.json'),
 saved_log_step=history[-1]['step'] if history else None,
 checkpoint_exists=bool(latest and fs.exists(fsspec.core.url_to_fs(latest['checkpoint'])[1])))))
"""


class Restarter:
    def __init__(self, plan_path):
        self.plan = json.loads(plan_path.read_text())
        self.state_path = plan_path.with_name("status.json")
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {"model": self.plan["model"], "phase": "waiting", "created_utc": now()}
        )

    def record(self, **fields):
        self.state.update(fields, observed_utc=now())
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.state_path)

    def command(self, command, *, check=True, input=None, cwd=None):
        result = subprocess.run(
            command,
            input=input,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
            cwd=cwd,
        )
        if check and result.returncode:
            # Commands can contain private connection details. Keep them out
            # of status files and supervisor logs.
            raise RuntimeError("Operational command failed")
        return result

    def pods(self, job):
        result = self.command(
            self.plan["kubectl"]
            + [
                "get",
                "pods",
                "-l",
                "iris.job_id=" + job.strip("/").replace("/", "."),
                "-o",
                "json",
            ]
        )
        return json.loads(result.stdout)["items"]

    def observe(self, new=False):
        code = OBSERVE.replace("OUTPUT", repr(self.plan["output"]))
        if self.plan["backend"] == "dedicated":
            result = self.command(
                self.plan["ssh"] + [self.plan["python"] + " -"], input=code
            )
        else:
            job = self.plan["new_job" if new else "old_job"]
            running = [p for p in self.pods(job) if p["status"]["phase"] == "Running"]
            if not running:
                raise RuntimeError("No running training pod")
            pod = max(running, key=lambda p: p["metadata"]["creationTimestamp"])
            result = self.command(
                self.plan["kubectl"]
                + [
                    "exec",
                    pod["metadata"]["name"],
                    "-c",
                    "task",
                    "--",
                    "/app/.venv/bin/python",
                    "-c",
                    code,
                ]
            )
        return json.loads(result.stdout)

    def stop_old(self):
        if self.plan["backend"] == "dedicated":
            self.command(
                self.plan["ssh"] + ["systemctl --user stop " + self.plan["old_service"]]
            )
        else:
            self.command(self.plan["iris"] + ["job", "cancel", self.plan["old_job"]])
            for _ in range(60):
                active = [
                    p
                    for p in self.pods(self.plan["old_job"])
                    if p["status"]["phase"] not in {"Failed", "Succeeded"}
                ]
                if not active:
                    return
                time.sleep(2)
            raise RuntimeError("Old training pod has not stopped")

    def start_new(self):
        if self.plan["backend"] == "dedicated":
            # Check existence before launch, so a retry after an interrupted
            # SSH response cannot start two training services.
            code = """
import json,subprocess
from pathlib import Path
p=subprocess.run(['systemctl','--user','show',UNIT,'-p','LoadState','--value'],capture_output=True,text=True,check=True)
if p.stdout.strip()=='not-found':
 subprocess.run(COMMAND,check=True,capture_output=True,text=True)
else:
 subprocess.run(['systemctl','--user','start',UNIT],check=True,capture_output=True,text=True)
print(json.dumps({'service':UNIT}))
""".replace("UNIT", repr(self.plan["new_service"])).replace(
                "COMMAND", repr(self.plan["launch_command"])
            )
            self.command(self.plan["ssh"] + [self.plan["python"] + " -"], input=code)
        else:
            existing = self.command(
                self.plan["iris"] + ["job", "describe", self.plan["new_job"]],
                check=False,
            )
            if existing.returncode:
                # Submission uses one fixed job name. The controller rejects
                # duplicate names if a previous submission was acknowledged late.
                self.command(self.plan["launch_command"], cwd=self.plan["snapshot"])

    def publish(self):
        # The two mirrored manifests are shared by independent run watchers.
        with open(self.plan["manifest_lock"], "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for name in self.plan["manifests"]:
                path = Path(name)
                manifest = json.loads(path.read_text())
                run = next(
                    r for r in manifest["runs"] if r["model"] == self.plan["model"]
                )
                key = "job" if self.plan["backend"] == "iris" else "service"
                if run[key] not in {self.plan["old_" + key], self.plan["new_" + key]}:
                    raise RuntimeError("Active run changed outside this handoff")
                run.update(
                    {
                        key: self.plan["new_" + key],
                        "previous_" + key: self.plan["old_" + key],
                        "source_commit": self.plan["source_commit"],
                        "restarted_from_step": self.state["checkpoint"]["step"],
                        "restart_verified_utc": self.state["verified_utc"],
                        "relaunch_reason": "Numerically verified attention/chunk checkpoint optimization.",
                    }
                )
                manifest["updated_utc"] = now()
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(manifest, indent=2) + "\n")
                temporary.replace(path)

    def run(self):
        deadline = datetime.fromisoformat(self.plan["deadline_utc"])
        while datetime.now(UTC) < deadline:
            try:
                phase = self.state["phase"]
                if phase == "waiting":
                    observed = self.observe()
                    self.record(last_observation=observed, error=None)
                    if checkpoint_ready(observed, self.plan["minimum_step"]):
                        self.record(
                            phase="stopping",
                            checkpoint=observed["latest"],
                            boundary_utc=now(),
                        )
                elif phase == "stopping":
                    self.stop_old()
                    self.record(phase="starting", old_stopped_utc=now())
                elif phase == "starting":
                    self.start_new()
                    self.record(phase="verifying", submitted_utc=now())
                elif phase == "verifying":
                    observed = self.observe(new=True)
                    runtime = observed.get("runtime") or {}
                    hashes = runtime.get("code_sha256", {})
                    if (
                        runtime.get("resumed_from")
                        == self.state["checkpoint"]["checkpoint"]
                        and runtime.get("resumed_step")
                        == self.state["checkpoint"]["step"]
                        and runtime.get("world_size") == 8
                        and all(
                            hashes.get(k) == v
                            for k, v in self.plan["code_sha256"].items()
                        )
                    ):
                        self.record(
                            phase="resumed",
                            resumed_runtime=runtime,
                            verified_utc=now(),
                            error=None,
                        )
                elif phase == "resumed":
                    if not self.state.get("manifests_updated"):
                        self.publish()
                        self.record(manifests_updated=True)
                    observed = self.observe(new=True)
                    self.record(last_observation=observed, error=None)
                    if (observed.get("latest") or {}).get("step", 0) > self.state[
                        "checkpoint"
                    ]["step"]:
                        self.record(phase="complete", completed_utc=now())
                        return
                elif phase == "complete":
                    return
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
            ) as error:
                self.record(error=type(error).__name__, error_utc=now())
            time.sleep(5)
        self.record(error="DeadlineExceeded", error_utc=now())
        raise RuntimeError("Restart handoff did not complete before its deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    Restarter(args.plan).run()


if __name__ == "__main__":
    main()
