"""Batch-only Iris operations and a CPU-only artifact reader."""

import json
import subprocess
from pathlib import Path

TERMINAL = {"succeeded", "failed", "killed", "cancelled", "canceled"}


class IrisBackend:
    def __init__(self, root, campaign):
        self.root = Path(root)
        self.campaign = Path(campaign)
        self.settings = json.loads((self.campaign / "settings.json").read_text())
        self.iris = [self.settings["iris"], "--config", self.settings["cluster_config"]]
        self.kube = [
            "kubectl",
            "--kubeconfig",
            self.settings["kubeconfig"],
            "--context",
            self.settings["kube_context"],
            "-n",
            "iris",
        ]
        self.prefix = (
            f"/bizon/sparsa-ar-{self.settings.get('run_name', self.settings['name'])}-"
        )

    def command(self, argv, cwd=None, timeout=60):
        return subprocess.run(
            argv,
            cwd=cwd or self.root,
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        ).stdout

    def status(self, job):
        if not job.startswith(self.prefix):
            raise ValueError("Job is outside this campaign")
        output = self.command(
            [
                self.settings["iris_python"],
                "-m",
                "research.iris_status",
                "--config",
                self.settings["cluster_config"],
                "--job",
                job,
            ]
        )
        return json.loads(output)

    def submit(self, job, workspace, gpus, timeout, args):
        if self.status(job)["state"] != "not_found":
            return  # Same deterministic handle after a lost submission response.
        argv = self.iris + [
            "job",
            "run",
            "--priority",
            "batch",
            "--enable-extra-resources",
            "--cpu",
            str(gpus * 6 if gpus else 2),
            "--memory",
            f"{gpus * 32 if gpus else 4}GB",
            "--disk",
            "40GB",
            "--timeout",
            str(timeout),
            "--max-retries",
            "0",
            "--job-name",
            job.rsplit("/", 1)[1],
            "--no-wait",
        ]
        if gpus:
            argv += ["--gpu", f"H100x{gpus}"]
        argv += ["--", "python", *args]
        from research.core import write_json

        write_json(
            self.campaign / "submissions" / (job.rsplit("/", 1)[1] + ".json"),
            {
                "job": job,
                "priority": "batch",
                "command": argv,
                "workspace": str(workspace),
            },
        )
        # The journal is written before submission by the caller. A timeout here
        # is an uncertain response, never permission to use a second job name.
        response = self.command(argv, cwd=workspace, timeout=300)
        if job not in response:
            raise RuntimeError(
                "Unexpected Iris submission response; inspect the recorded job"
            )

    def cancel(self, job):
        if not job.startswith(self.prefix):
            raise ValueError("Job is outside this campaign")
        self.command(self.iris + ["job", "cancel", job])

    def diagnostic(self, job):
        if not job.startswith(self.prefix):
            raise ValueError("Job is outside this campaign")
        pods = json.loads(
            self.command(
                self.kube
                + [
                    "get",
                    "pods",
                    "-l",
                    "iris.job_id=" + job.strip("/").replace("/", "."),
                    "-o",
                    "json",
                ]
            )
        )["items"]
        if not pods:
            return "Pod logs unavailable; see recorded Iris attempt diagnostics."
        pod = max(pods, key=lambda p: p["metadata"]["creationTimestamp"])["metadata"][
            "name"
        ]
        return self.command(self.kube + ["logs", pod, "-c", "task", "--tail=80"])[
            -16000:
        ]

    def bridge(self):
        state_path = self.campaign / "bridge.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"generation": 0}
        )
        job = self.prefix + f"bridge-{state['generation']}"
        status = self.status(job)["state"]
        if status in TERMINAL:
            state["generation"] += 1
            job = self.prefix + f"bridge-{state['generation']}"
        from research.core import write_json

        write_json(state_path, state | {"job": job})
        self.submit(job, self.campaign / "base", 0, 14400, ["-m", "research.bridge"])
        pods = json.loads(
            self.command(
                self.kube
                + [
                    "get",
                    "pods",
                    "-l",
                    "iris.job_id=" + job.strip("/").replace("/", "."),
                    "-o",
                    "json",
                ]
            )
        )["items"]
        running = [p for p in pods if p["status"]["phase"] == "Running"]
        if not running:
            return None
        return max(running, key=lambda p: p["metadata"]["creationTimestamp"])[
            "metadata"
        ]["name"]

    def result(self, uri):
        expected = self.settings["output_prefix"] + "/"
        if not uri.startswith(expected):
            raise ValueError("Artifact is outside this campaign")
        pod = self.bridge()
        if pod is None:
            return None
        code = (
            "import fsspec,sys; fs,p=fsspec.core.url_to_fs(sys.argv[1]); "
            "print(fs.cat_file(p).decode() if fs.exists(p) else 'null')"
        )
        response = self.command(
            self.kube
            + [
                "exec",
                pod,
                "-c",
                "task",
                "--",
                "/app/.venv/bin/python",
                "-c",
                code,
                uri,
            ],
            timeout=90,
        )
        result = json.loads(response)
        if result is None:
            raise ValueError(
                "Succeeded trial is missing its result.json; inspect the remote run"
            )
        return result

    def close(self):
        path = self.campaign / "bridge.json"
        if path.exists():
            job = json.loads(path.read_text())["job"]
            if self.status(job)["state"] not in TERMINAL | {"not_found"}:
                self.cancel(job)
