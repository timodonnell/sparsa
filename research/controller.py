"""Persistent serial architecture search. No mutation of the working project's model."""

import argparse
import copy
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

from research.backend import TERMINAL, IrisBackend
from research.core import (
    digest,
    identity,
    promotion,
    reservation,
    validate_protocol,
    validate_result,
    validate_source,
    write_json,
)
from research.propose import ProposerUnavailable, codex_proposal, seeded_proposal

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(UTC).isoformat()


def snapshot_files(root):
    names = [
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".gitignore",
        "scripts/job_accounting.py",
        "research/program.md",
        "research/protocol.yaml",
    ]
    for directory, pattern in (
        ("sparsa", "*.py"),
        ("research", "*.py"),
        ("data/benchmark", "*"),
    ):
        names += [
            str(p.relative_to(root))
            for p in (root / directory).rglob(pattern)
            if p.is_file() and "__pycache__" not in p.parts
        ]
    return sorted(set(names))


def initialize(path, protocol, settings, source_root=ROOT):
    validate_protocol(protocol)
    if (path / "state.json").exists():
        raise ValueError("Campaign already exists; use run to resume")
    path.mkdir(parents=True, exist_ok=True)
    base = path / "base"
    if base.exists():
        raise ValueError(
            "Incomplete initialization exists; choose a new campaign directory"
        )
    base.mkdir()
    files = snapshot_files(source_root)
    for name in files:
        src, dest = source_root / name, base / name
        if src.is_symlink():
            raise ValueError(f"Snapshot file must be regular: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    # A separate repository prevents Iris from accidentally bundling the parent
    # workspace when the snapshot lives under the ignored outputs directory.
    subprocess.run(["git", "init", "-q", str(base)], check=True)
    from sparsa.data import benchmark

    frozen = {
        name: digest((base / name).read_bytes())
        for name in files
        if name != "sparsa/model.py"
    }
    recipe = {k: v for k, v in protocol["training"].items() if k != "model"}
    contract = {
        "frozen_files": frozen,
        "training": recipe,
        "gpus": protocol["gpus"],
        "validation_stems": sorted(r["stem"] for r in benchmark()),
        "readout": "final-step EMA, original distance readout",
    }
    source = (base / "sparsa/model.py").read_text()
    candidate = {
        "id": "c000",
        "parent": None,
        "hypothesis": "Matched-budget original Sparsa baseline",
        "model_config": protocol["training"]["model"],
        "source_hash": identity(source, protocol["training"]["model"]),
        "status": "baseline",
        "results": {},
    }
    dest = path / "candidates/c000"
    dest.mkdir(parents=True)
    (dest / "model.py").write_text(source)
    write_json(dest / "candidate.json", candidate)
    write_json(path / "protocol.json", protocol)
    write_json(path / "settings.json", settings)
    write_json(path / "contract.json", contract)
    (path / "IDEAS.md").write_text(
        "# Research steering\n\nAdd architectural hypotheses or priorities here. "
        "The next code proposal receives this text; the training/scoring protocol stays fixed.\n"
    )
    write_json(
        path / "state.json",
        {
            "version": 1,
            "created_utc": now(),
            "stage": "ready",
            "champion": "c000",
            "budget_gpu_hours": protocol["gpu_hours"],
            "candidates": [candidate],
            "trials": [],
        },
    )
    return path


class Controller:
    def __init__(self, path, backend=None):
        self.path = Path(path).resolve()
        self.protocol = json.loads((self.path / "protocol.json").read_text())
        self.settings = json.loads((self.path / "settings.json").read_text())
        self.contract = json.loads((self.path / "contract.json").read_text())
        self.state = json.loads((self.path / "state.json").read_text())
        validate_protocol(self.protocol)
        self.backend = backend or IrisBackend(ROOT, self.path)

    def save(self, event, **details):
        self.state["updated_utc"] = now()
        write_json(self.path / "state.json", self.state)
        record = {"utc": now(), "event": event, **details}
        with (self.path / "events.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
        self.report()

    def report(self):
        lines = [
            "# Sparsa architecture search",
            "",
            f"Stage: {self.state['stage']}; champion: {self.state['champion']}",
            f"Reserved budget: {self.charged():.2f} / {self.state['budget_gpu_hours']:.2f} H100-hours.",
            "",
            "Scores below use the fixed short training budget, not the production run.",
            "",
            "| Candidate | Status | Seed R-precisions | Hypothesis |",
            "|---|---|---|---|",
        ]
        for c in self.state["candidates"]:
            metrics = ", ".join(
                f"{s}: {r['r_precision']:.6f}" for s, r in c["results"].items()
            )
            lines.append(
                f"| {c['id']} | {c['status']} | {metrics} | {c['hypothesis'].replace('|', '/').replace(chr(10), ' ')} |"
            )
        (self.path / "REPORT.md").write_text("\n".join(lines) + "\n")

    def charged(self):
        return sum(
            max(t["reserved_gpu_hours"], t.get("observation", {}).get("gpu_hours", 0))
            for t in self.state["trials"]
        )

    def candidate(self, cid):
        return next(c for c in self.state["candidates"] if c["id"] == cid)

    def workspace(self, candidate, seed):
        name = f"{candidate['id']}-s{seed}"
        path = self.path / "work" / name
        if not path.exists():
            shutil.copytree(
                self.path / "base",
                path,
                ignore=shutil.ignore_patterns(".git", "__pycache__"),
            )
            subprocess.run(["git", "init", "-q", str(path)], check=True)
        source = (self.path / "candidates" / candidate["id"] / "model.py").read_text()
        if identity(source, candidate["model_config"]) != candidate["source_hash"]:
            raise ValueError("Candidate changed after it was recorded")
        validate_source(source, (self.path / "base/sparsa/model.py").read_text())
        (path / "sparsa/model.py").write_text(source)
        cfg = copy.deepcopy(self.protocol["training"])
        cfg.update(seed=seed, model=candidate["model_config"])
        (path / "configs").mkdir(exist_ok=True)
        (path / "configs/research_trial.yaml").write_text(yaml.safe_dump(cfg))
        for file, expected in self.contract["frozen_files"].items():
            if digest((path / file).read_bytes()) != expected:
                raise ValueError(f"Frozen file changed: {file}")
        return path

    def schedule(self, candidate, seed):
        cost = reservation(self.protocol)
        if self.charged() + cost > self.state["budget_gpu_hours"] + 1e-9:
            self.state["stage"] = "budget_exhausted"
            self.save("budget_exhausted")
            return
        workspace = self.workspace(candidate, seed)
        try:
            checked = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research.check_model",
                    "--config",
                    "configs/research_trial.yaml",
                    "--max-parameters",
                    str(self.protocol["max_parameters"]),
                ],
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired:
            candidate.update(
                status="invalid", error="CPU model preflight exceeded 180 seconds"
            )
            self.save("preflight_failed", candidate=candidate["id"])
            return
        (workspace / "preflight.log").write_text(checked.stdout + checked.stderr)
        if checked.returncode:
            candidate.update(status="invalid", error=checked.stderr[-4000:])
            self.save("preflight_failed", candidate=candidate["id"])
            return
        trial_id = f"{candidate['id']}-s{seed}"
        run_name = self.settings.get("run_name", self.settings["name"])
        job = f"/bizon/sparsa-ar-{run_name}-{trial_id}"
        out = self.settings["output_prefix"] + "/" + trial_id
        files = self.contract["frozen_files"] | {
            "sparsa/model.py": digest((workspace / "sparsa/model.py").read_bytes()),
            "configs/research_trial.yaml": digest(
                (workspace / "configs/research_trial.yaml").read_bytes()
            ),
        }
        spec = {
            "trial_id": trial_id,
            "candidate_id": candidate["id"],
            "out": out,
            "gpus": self.protocol["gpus"],
            "timeout_seconds": self.protocol["trial_timeout_seconds"],
            "contract": self.contract,
            "files": files,
        }
        write_json(workspace / "research_trial.json", spec)
        self.state["trials"].append(
            {
                "id": trial_id,
                "candidate": candidate["id"],
                "seed": seed,
                "job": job,
                "out": out,
                "workspace": str(workspace),
                "state": "reserved",
                "reserved_gpu_hours": cost,
                "spec": spec,
            }
        )
        self.state["stage"] = "running"
        self.save("trial_reserved", trial=trial_id, job=job)

    def observe(self, trial):
        observation = self.backend.status(trial["job"])
        trial["observation"] = observation
        status = observation["state"]
        if status == "not_found":
            workspace = Path(trial["workspace"])
            if (
                json.loads((workspace / "research_trial.json").read_text())
                != trial["spec"]
            ):
                raise ValueError(
                    "Reserved trial specification changed before submission"
                )
            for name, expected in trial["spec"]["files"].items():
                if digest((workspace / name).read_bytes()) != expected:
                    raise ValueError(f"Reserved trial source changed: {name}")
            self.backend.submit(
                trial["job"],
                trial["workspace"],
                self.protocol["gpus"],
                self.protocol["trial_timeout_seconds"],
                ["-m", "research.trial"],
            )
            trial["state"] = "submitted"
            self.save("trial_submitted", trial=trial["id"])
            return
        if status not in TERMINAL:
            if observation.get("gpu_hours", 0) >= trial[
                "reserved_gpu_hours"
            ] or observation.get("unknown_attempts"):
                self.backend.cancel(trial["job"])
                trial["cancel_reason"] = "trial_resource_limit_or_incomplete_accounting"
                self.save("trial_cancelling", trial=trial["id"])
            elif trial["state"] != status:
                trial["state"] = status
                self.save("trial_state", trial=trial["id"], state=status)
            else:
                write_json(self.path / "state.json", self.state)
            return
        if status != "succeeded":
            try:
                diagnostic = self.backend.diagnostic(trial["job"])
            except (subprocess.SubprocessError, OSError) as error:
                diagnostic = f"Could not recover pod logs: {error}"
            path = self.path / "failures" / f"{trial['id']}.log"
            path.parent.mkdir(exist_ok=True)
            path.write_text(diagnostic)
            trial["state"] = "failed"
            self.candidate(trial["candidate"]).update(
                status="failed",
                error=trial.get("cancel_reason", status) + "\n" + diagnostic[-4000:],
            )
            self.save("trial_failed", trial=trial["id"], observation=observation)
            return
        result = self.backend.result(trial["out"] + "/result.json")
        if result is None:
            trial["state"] = "awaiting_artifact"
            self.save("awaiting_artifact", trial=trial["id"])
            return
        validate_result(result, trial, self.protocol, self.contract)
        if result.get("files") != trial["spec"]["files"]:
            raise ValueError("Result source hashes do not match the submitted snapshot")
        expected_inventory = self.state.get("teacher_inventory_sha256")
        if (
            expected_inventory
            and result["teacher_inventory_sha256"] != expected_inventory
        ):
            raise ValueError("Teacher shard inventory changed during search")
        self.state["teacher_inventory_sha256"] = result["teacher_inventory_sha256"]
        trial["state"] = "completed"
        self.candidate(trial["candidate"])["results"][str(trial["seed"])] = result
        write_json(self.path / "results" / f"{trial['id']}.json", result)
        self.save(
            "trial_completed", trial=trial["id"], r_precision=result["r_precision"]
        )

    def propose(self):
        parent = self.candidate(self.state["champion"])
        index = len(self.state["candidates"])
        cid = f"c{index:03d}"
        path = self.path / "candidates" / cid
        path.mkdir(parents=True, exist_ok=True)
        candidate = {
            "id": cid,
            "parent": parent["id"],
            "status": "proposing",
            "results": {},
            "hypothesis": "Pending proposal",
        }
        self.state["candidates"].append(candidate)
        self.save("proposing", candidate=cid, parent=parent["id"])
        source = (self.path / "candidates" / parent["id"] / "model.py").read_text()
        try:
            mode = self.settings["proposer"]
            if mode == "seeded" or mode == "hybrid" and index <= 3:
                # Initial ablations stay anchored to the original baseline.
                parent = self.candidate("c000")
                candidate["parent"] = parent["id"]
                source = (self.path / "candidates/c000/model.py").read_text()
                proposal = seeded_proposal(index, source, parent["model_config"])
            else:
                history = [
                    {
                        k: c.get(k)
                        for k in (
                            "id",
                            "parent",
                            "hypothesis",
                            "status",
                            "decision",
                            "error",
                            "model_config",
                        )
                    }
                    | {
                        "scores": {
                            s: {
                                k: r[k]
                                for k in (
                                    "r_precision",
                                    "long_r_precision",
                                    "parameters",
                                )
                            }
                            | {
                                k: r.get(k)
                                for k in ("validation_curve", "recent_training")
                            }
                            for s, r in c["results"].items()
                        }
                    }
                    for c in self.state["candidates"][:-1]
                ]
                proposal = codex_proposal(
                    path / "proposal",
                    source,
                    parent["model_config"],
                    history,
                    self.protocol,
                    self.settings,
                    guidance=(self.path / "IDEAS.md").read_text(),
                )
            self.accept_proposal(candidate, proposal)
        except (ProposerUnavailable, FileNotFoundError) as error:
            candidate.update(status="invalid", error=str(error))
            self.state["stage"] = "proposer_failed"
            self.save("proposer_failed", candidate=cid, error=str(error))
        except (
            ValueError,
            SyntaxError,
            TypeError,
            KeyError,
            RuntimeError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            candidate.update(status="invalid", error=str(error))
            self.save("proposal_failed", candidate=cid, error=str(error))

    def accept_proposal(self, candidate, proposal):
        validate_source(
            proposal["model_source"], (self.path / "base/sparsa/model.py").read_text()
        )
        cfg = json.loads(proposal["model_config_json"])
        if not isinstance(cfg, dict) or not set(
            self.protocol["training"]["model"]
        ) <= set(cfg):
            raise ValueError(
                "Candidate must preserve existing model configuration fields"
            )
        signature = identity(proposal["model_source"], cfg)
        if any(c.get("source_hash") == signature for c in self.state["candidates"]):
            raise ValueError("Duplicate architecture/configuration")
        candidate.update(
            status="screening",
            hypothesis=proposal["hypothesis"],
            model_config=cfg,
            source_hash=signature,
        )
        directory = self.path / "candidates" / candidate["id"]
        (directory / "model.py").write_text(proposal["model_source"])
        write_json(directory / "candidate.json", candidate)
        self.save(
            "proposal_ready",
            candidate=candidate["id"],
            hypothesis=candidate["hypothesis"],
        )

    def tick(self):
        active = [
            t for t in self.state["trials"] if t["state"] not in {"completed", "failed"}
        ]
        if len(active) > 1:
            raise ValueError("Only one GPU trial may be active")
        if active:
            self.observe(active[0])
            return
        if (self.path / "STOP").exists():
            self.state["stage"] = "paused"
            self.save("paused")
            return
        baseline = self.candidate("c000")
        if baseline["status"] in {"failed", "invalid"}:
            self.state["stage"] = "baseline_failed"
            self.save("baseline_failed")
            return
        current = self.state["candidates"][-1]
        if current["status"] == "proposing":
            # A controller interruption may have left a complete proposal on disk.
            path = self.path / "candidates" / current["id"] / "proposal/proposal.json"
            if path.exists():
                self.accept_proposal(current, json.loads(path.read_text()))
            else:
                current.update(
                    status="invalid", error="Interrupted proposal; no complete output"
                )
                self.save("proposal_interrupted", candidate=current["id"])
            return
        if current["status"] in {"baseline", "screening", "confirming"}:
            seeds = self.protocol["seeds"]
            if str(seeds[0]) not in current["results"]:
                self.schedule(current, seeds[0])
                return
            parent = self.candidate(current["parent"]) if current["parent"] else None
            if parent and str(seeds[1]) not in current["results"]:
                gain = (
                    current["results"][str(seeds[0])]["r_precision"]
                    - parent["results"][str(seeds[0])]["r_precision"]
                )
                if gain < self.protocol["minimum_gain"]:
                    current.update(
                        status="rejected",
                        decision={
                            "reason": "screening_gain_below_threshold",
                            "gain": gain,
                        },
                    )
                    self.save("candidate_rejected", candidate=current["id"], gain=gain)
                    return
                current["status"] = "confirming"
            if str(seeds[1]) not in current["results"]:
                self.schedule(current, seeds[1])
                return
            if parent:
                decision = promotion(
                    {int(s): r for s, r in current["results"].items()},
                    {int(s): r for s, r in parent["results"].items()},
                    self.protocol,
                )
                current["decision"] = decision
                current["status"] = "accepted" if decision["accepted"] else "rejected"
                # Seeded ablations compare with the original baseline. A later
                # accepted ablation must also beat the current champion to replace it.
                champion = self.candidate(self.state["champion"])
                if decision["accepted"] and champion["id"] != parent["id"]:
                    decision = promotion(
                        {int(s): r for s, r in current["results"].items()},
                        {int(s): r for s, r in champion["results"].items()},
                        self.protocol,
                    )
                    current["champion_comparison"] = decision
                if decision["accepted"]:
                    self.state["champion"] = current["id"]
            else:
                current["status"] = "accepted"
            self.save(
                "candidate_decided",
                candidate=current["id"],
                status=current["status"],
                champion=self.state["champion"],
            )
            return
        if (
            len(self.state["candidates"]) - 1 >= self.protocol["max_candidates"]
            or self.settings["proposer"] == "seeded"
            and len(self.state["candidates"]) > 3
        ):
            self.state["stage"] = "complete"
            self.save("search_complete")
        elif (
            self.charged() + reservation(self.protocol)
            > self.state["budget_gpu_hours"] + 1e-9
        ):
            self.state["stage"] = "budget_exhausted"
            self.save("budget_exhausted")
        else:
            self.propose()


def main():
    parser = argparse.ArgumentParser(prog="python -m research.controller")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--name", required=True)
    init.add_argument("--protocol", default=str(ROOT / "research/protocol.yaml"))
    init.add_argument("--gpu-hours", type=float)
    init.add_argument(
        "--proposer", choices=("seeded", "codex", "hybrid"), default="codex"
    )
    init.add_argument("--codex", default="codex")
    init.add_argument("--codex-model")
    init.add_argument("--iris", default=str(ROOT / ".tools/iris/bin/iris"))
    init.add_argument("--iris-python", default=str(ROOT / ".tools/iris/bin/python"))
    init.add_argument(
        "--cluster-config",
        default="/home/bizon/git/marin-freshiris/lib/iris/config/cw-rno2a.yaml",
    )
    init.add_argument("--kubeconfig", default=str(Path.home() / ".kube/coreweave-iris"))
    init.add_argument("--kube-context", default="marin-rn02a_RNO2A")
    run = sub.add_parser("run")
    run.add_argument("--watch", action="store_true")
    sub.add_parser("status")
    sub.add_parser("stop")
    resume = sub.add_parser("resume")
    resume.add_argument("--gpu-hours", type=float)
    export = sub.add_parser("export")
    export.add_argument("--out", required=True)
    for p in (
        init,
        run,
        *[sub.choices[k] for k in ("status", "stop", "resume", "export")],
    ):
        p.add_argument("--campaign", required=True)
    args = parser.parse_args()
    path = Path(args.campaign).resolve()
    if args.command == "run" and ROOT.resolve() != (path / "base").resolve():
        # Pin the controller itself, not just the GPU workspace. Later edits to
        # this repository must not silently change an in-flight campaign.
        argv = [
            sys.executable,
            "-m",
            "research.controller",
            "run",
            "--campaign",
            str(path),
        ]
        if args.watch:
            argv.append("--watch")
        os.chdir(path / "base")
        os.execv(sys.executable, argv)
    if args.command == "init":
        if not re.fullmatch(r"[a-z0-9-]{1,20}", args.name):
            raise ValueError(
                "Campaign name must be 1–20 lowercase letters, digits, or hyphens"
            )
        p = yaml.safe_load(Path(args.protocol).read_text())
        if args.gpu_hours is not None:
            p["gpu_hours"] = args.gpu_hours
        settings = {
            key: getattr(args, key)
            for key in (
                "name",
                "proposer",
                "codex",
                "codex_model",
                "iris",
                "iris_python",
                "cluster_config",
                "kubeconfig",
                "kube_context",
            )
        }
        settings["run_name"] = f"{args.name}-{uuid.uuid4().hex[:6]}"
        settings["output_prefix"] = (
            "s3://marin-us-east-02a/marin/protein-structure/sparsa/autoresearch/"
            + settings["run_name"]
        )
        initialize(path, p, settings)
        Controller(path).report()
        print(f"Initialized {path}; no jobs submitted")
        return
    if args.command == "stop":
        (path / "STOP").touch()
        print(
            "Stop requested; the active trial will finish and no new trial will start"
        )
        return
    if args.command == "status":
        print((path / "REPORT.md").read_text())
        return
    with (path / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = Controller(path)
        if args.command == "resume":
            if args.gpu_hours is not None:
                if (
                    not math.isfinite(args.gpu_hours)
                    or args.gpu_hours < controller.charged()
                ):
                    raise ValueError("New budget is below already reserved resources")
                controller.state["budget_gpu_hours"] = args.gpu_hours
            (path / "STOP").unlink(missing_ok=True)
            controller.state["stage"] = "ready"
            controller.save("resumed")
            return
        if args.command == "export":
            champion = controller.candidate(controller.state["champion"])
            if len(champion["results"]) != 2:
                raise ValueError("No confirmed champion yet")
            source = controller.workspace(champion, controller.protocol["seeds"][0])
            shutil.copytree(
                source,
                args.out,
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "outputs", "*.log"
                ),
            )
            write_json(Path(args.out) / "champion.json", champion)
            write_json(Path(args.out) / "search_protocol.json", controller.protocol)
            print(f"Exported runnable source and checkpoint references to {args.out}")
            return
        stopped = {
            "complete",
            "budget_exhausted",
            "paused",
            "baseline_failed",
            "proposer_failed",
            "needs_inspection",
        }
        while controller.state["stage"] not in stopped:
            try:
                controller.tick()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                # A failed status tunnel is not a failed GPU experiment.
                controller.save("transport_error", error=str(error))
            except Exception as error:
                controller.state["stage"] = "needs_inspection"
                controller.save("needs_inspection", error=str(error))
                raise
            if not args.watch:
                break
            if controller.state["stage"] not in stopped:
                time.sleep(controller.protocol["poll_seconds"])
        if controller.state["stage"] in stopped:
            controller.backend.close()


if __name__ == "__main__":
    main()
