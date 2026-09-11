"""Complete fixed scaling pilots, choose on validation, and launch longer training.

Run locally under a process supervisor. The manifest fixes pilot identities and
configs; the journal freezes the decision before submission. Iris job names make
restarts idempotent. All GPUs use batch priority and join the campaign guard.
"""

import argparse
import fcntl
import json
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

BASE = "s3://marin-us-east-02a/marin/protein-structure/sparsa"


def choose_extension(comparison, pilots, elapsed, spent, limit=600):
    """Use pilot validation for recipe and measured time for a bounded run length."""
    if comparison["split"] != "eval-val" or comparison["test_used"]:
        raise ValueError("Only validation may choose the training recipe")
    if any(v <= 0 for v in elapsed.values()) or not 0 <= spent < limit <= 600:
        raise ValueError("Invalid measured time or campaign budget")
    candidates = [
        (name, row)
        for name, row in comparison["models"].items()
        if name.endswith("-best") and name != "release"
    ]
    name, source = max(
        candidates, key=lambda item: item[1]["validation"]["r_precision"]
    )
    pilot = next(p for p in pilots if p["name"] == name.removesuffix("-best"))
    # Reserve 60 GPU-hours for long-crop finetuning, selection, and contingencies.
    available = limit - spent - 60
    if available <= 0:
        raise RuntimeError("Insufficient remaining campaign budget")
    capacity = "18b"
    seconds = elapsed[capacity] / 6000 * 1.10
    steps = min(200000, math.floor(available * 3600 / (8 * seconds) / 1000) * 1000)
    if steps < 100000:
        capacity = "456m"
        seconds = elapsed[capacity] / 6000 * 1.10
        steps = min(200000, math.floor(available * 3600 / (8 * seconds) / 1000) * 1000)
        # Widening is supported; shrinking a selected 1.8B source is not.
        candidates = [(n, r) for n, r in candidates if r["parameters"] < 500000000]
        name, source = max(
            candidates, key=lambda item: item[1]["validation"]["r_precision"]
        )
        pilot = next(p for p in pilots if p["name"] == name.removesuffix("-best"))
    if steps < 100000:
        raise RuntimeError("Budget cannot support the planned extended training")
    if (
        comparison["models"]["release"]["validation"]["r_precision"]
        > source["validation"]["r_precision"]
    ):
        name, source = "release", comparison["models"]["release"]
    return {
        "source_name": name,
        "source_checkpoint": source["checkpoint"],
        "source_validation": source["validation"],
        "source_config": pilot["config"],
        "capacity": capacity,
        "steps": steps,
        "estimated_seconds_per_step": seconds,
        "estimated_main_h100_hours": steps * seconds * 8 / 3600,
        "spent_h100_hours": spent,
        "reserved_h100_hours": 60,
        "decision_split": "eval-val",
        "test_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(Path(args.manifest).read_text())
    name = manifest["name"]
    local = root / "outputs" / name
    local.mkdir(parents=True, exist_ok=True)
    lock = (local / "campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    journal = local / "campaign.json"
    state = json.loads(journal.read_text()) if journal.exists() else {}
    iris = [
        str(root / ".tools/iris/bin/iris"),
        "--config",
        "/home/bizon/git/marin-freshiris/lib/iris/config/cw-rno2a.yaml",
    ]
    kube = [
        "kubectl",
        "--kubeconfig",
        "/home/bizon/.kube/coreweave-iris",
        "--context",
        "marin-rn02a_RNO2A",
        "-n",
        "iris",
    ]

    def run(cmd, **kw):
        return subprocess.run(
            cmd, cwd=root, check=True, text=True, capture_output=True, **kw
        ).stdout

    def record(stage, **kw):
        state.update(stage=stage, updated_utc=datetime.now(UTC).isoformat(), **kw)
        tmp = journal.with_suffix(".partial")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(journal)
        print(json.dumps(state), flush=True)

    def register(job):
        path = Path(manifest["budget"])
        budget = json.loads(path.read_text())
        if job not in budget["jobs"]:
            budget["jobs"].append(job)
            tmp = path.with_suffix(".partial")
            tmp.write_text(json.dumps(budget, indent=2) + "\n")
            tmp.replace(path)

    def exists(job):
        try:
            run(iris + ["job", "describe", job])
            return True
        except subprocess.CalledProcessError as error:
            if not any(
                s in (error.stdout + error.stderr).lower()
                for s in ("not found", "not_found")
            ):
                raise
            return False

    def wait(job):
        for attempt in range(6):
            try:
                run(iris + ["job", "wait", job])
                return
            except subprocess.CalledProcessError:
                description = run(iris + ["job", "describe", job])
                if "State: succeeded" in description:
                    return
                if any(
                    "State: " + s in description
                    for s in ["failed", "killed", "cancelled", "canceled"]
                ):
                    raise RuntimeError("Inspect terminal job: " + job)
                if attempt == 5:
                    raise
                time.sleep(30)

    def marker(job, prefix):
        pods = json.loads(
            run(
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
        )["items"]
        for pod in sorted(
            pods, key=lambda p: p["metadata"]["creationTimestamp"], reverse=True
        ):
            try:
                lines = run(
                    kube + ["logs", pod["metadata"]["name"], "-c", "task"]
                ).splitlines()
            except subprocess.CalledProcessError:
                continue
            rows = [
                json.loads(line.removeprefix(prefix))
                for line in lines
                if line.startswith(prefix)
            ]
            if rows:
                return rows[-1]
        raise RuntimeError("Missing durable job log marker: " + job)

    try:
        if state.get("stage") == "complete":
            return
        for pilot in manifest["pilots"]:
            record("waiting_for_pilot", pilot=pilot["name"])
            wait(pilot["job"])
        compare_job = "/bizon/sparsa-compare-" + name
        if not exists(compare_job):
            cmd = iris + [
                "job",
                "run",
                "--priority",
                "batch",
                "--enable-extra-resources",
                "--gpu",
                "H100x1",
                "--cpu",
                "12",
                "--memory",
                "128GB",
                "--disk",
                "30GB",
                "--timeout",
                "14400",
                "--max-retries",
                "2",
                "--job-name",
                compare_job.split("/")[-1],
                "--no-wait",
                "--",
                "python",
                "-m",
                "scripts.compare_scaling",
                "--reference",
                manifest["reference"],
                "--out",
                BASE + "/scaling-v2/comparison",
            ]
            for p in manifest["pilots"]:
                cmd += ["--run", p["name"] + "=" + p["run"]]
            output = run(cmd)
            if compare_job not in output:
                raise RuntimeError("Unexpected comparison submission")
            (root / "reports/jobs" / f"compare-{name}.json").write_text(
                json.dumps(
                    {"job": compare_job, "command": cmd, "priority": "batch"}, indent=2
                )
                + "\n"
            )
        register(compare_job)
        record("waiting_for_comparison", comparison_job=compare_job)
        wait(compare_job)
        comparison = marker(compare_job, "PILOT_COMPARISON ")
        report = root / "reports/scaling_v2/comparison.json"
        report.write_text(json.dumps(comparison, indent=2) + "\n")
        # Completion metadata includes upload time; use a full fresh compiled pilot
        # at each size, not a resumed partial segment or a synthetic benchmark.
        elapsed = comparison["pilot_elapsed_seconds"]
        guard = json.loads(Path(manifest["guard_status"]).read_text())
        age = (
            datetime.now(UTC) - datetime.fromisoformat(guard["observed_utc"])
        ).total_seconds()
        if age > 180 or guard["state"] not in ("monitoring", "all_terminal"):
            raise RuntimeError("Compute guard is not healthy and current")
        if "decision" not in state:
            decision = choose_extension(
                comparison,
                manifest["pilots"],
                {"18b": elapsed["grown18b50"], "456m": elapsed["grown456m50-lr4"]},
                guard["enforcement_h100_hours"],
                guard["max_h100_hours"],
            )
            record("decision_frozen", decision=decision)
        decision = state["decision"]
        (root / "reports/scaling_v2/selection.json").write_text(
            json.dumps(decision, indent=2) + "\n"
        )
        cfg = yaml.safe_load((root / decision["source_config"]).read_text())
        big = decision["capacity"] == "18b"
        cfg["model"].update(
            sequence_dim=2048 if big else 1024,
            sequence_layers=36,
            heads=32 if big else 16,
            gradient_checkpointing=False,
        )
        cfg.update(
            seed=417,
            steps=decision["steps"],
            warmup=2000,
            batch_size=2 if big else 4,
            accumulation=4 if big else 2,
            workers=2,
            crop=512,
            eval_every=5000,
            checkpoint_every=2000,
            log_every=200,
            grow_from_ema=True,
            compile_pair_blocks=True,
            compile_cache_limit=128,
            keep_recovery_checkpoints=2,
        )
        main_config = "configs/scaling_main.yaml"
        (root / main_config).write_text(yaml.safe_dump(cfg, sort_keys=False))
        long_cfg = json.loads(json.dumps(cfg))
        long_cfg["model"]["gradient_checkpointing"] = True
        long_cfg.update(
            seed=419,
            steps=5000,
            warmup=100,
            crop=1024,
            batch_size=1,
            accumulation=4,
            lr=min(cfg["lr"] / 4, 0.00005),
            eval_every=500,
            checkpoint_every=500,
            log_every=100,
        )
        (root / "configs/scaling_long.yaml").write_text(
            yaml.safe_dump(long_cfg, sort_keys=False)
        )
        job = "/bizon/sparsa-" + name
        if not exists(job):
            run(
                [
                    sys.executable,
                    "scripts/submit.py",
                    "--name",
                    name,
                    "--config",
                    main_config,
                    "--gpus",
                    "8",
                    "--memory-gb",
                    "640" if big else "256",
                    "--timeout",
                    "259200",
                    "--init-from",
                    decision["source_checkpoint"],
                ]
            )
        register(job)
        record("main_training", training_job=job, training_run=BASE + "/runs/" + name)
        # The completion helper separately journals finetuning, validation selection,
        # one held-out evaluation, checksummed recovery, and actual CLI/Helico checks.
        subprocess.run(
            [
                sys.executable,
                "-u",
                "scripts/finish_run.py",
                "--name",
                name,
                "--training-job",
                job,
                "--long-config",
                "configs/scaling_long.yaml",
                "--training-memory-gb",
                "640" if big else "256",
                "--evaluation-memory-gb",
                "192",
                "--finetune-timeout",
                "28800",
                "--evaluation-timeout",
                "14400",
                "--extra-candidate-run",
                manifest["baseline_run"],
                "--budget-config",
                manifest["budget"],
                "--verify-release",
            ],
            cwd=root,
            check=True,
        )
        artifacts = local / "sparsa-results"
        budget_path = Path(manifest["budget"])
        budget = json.loads(budget_path.read_text())
        accounting = root / "reports/scaling_v2/job_accounting.json"
        command = [
            str(root / ".tools/iris/bin/python"),
            "scripts/job_accounting.py",
            "--out",
            str(accounting),
        ]
        for handle in budget["jobs"]:
            command += ["--job", handle]
        run(command)
        readme_clean = (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", "README.md"],
                cwd=root,
                check=False,
            ).returncode
            == 0
        )
        run(
            [
                sys.executable,
                "scripts/report_scaling.py",
                "--artifacts",
                str(artifacts),
                "--out",
                "reports/scaling_v2/final",
                "--evaluation-uri",
                BASE + "/evaluations/" + name,
                "--accounting",
                str(accounting),
                *(["--update-readme"] if readme_clean else []),
            ]
        )
        budget["keep_watching"] = False
        temporary = budget_path.with_suffix(".partial")
        temporary.write_text(json.dumps(budget, indent=2) + "\n")
        temporary.replace(budget_path)
        record(
            "complete",
            artifacts=str(artifacts),
            report="reports/scaling_v2/FINAL.md",
            published=False,
        )
        # Publish only campaign-owned outputs, preserving any pre-existing index.
        clean_index = (
            subprocess.run(
                ["git", "diff", "--cached", "--quiet"], cwd=root, check=False
            ).returncode
            == 0
        )
        if clean_index and run(["git", "branch", "--show-current"]).strip() == "main":
            paths = [
                "configs/scaling_main.yaml",
                "configs/scaling_long.yaml",
                "reports/scaling_v2/comparison.json",
                "reports/scaling_v2/selection.json",
                "reports/scaling_v2/job_accounting.json",
                "reports/scaling_v2/FINAL.md",
                "reports/scaling_v2/final",
                *(["README.md"] if readme_clean else []),
            ]
            paths += [
                str(p.relative_to(root))
                for p in (root / "reports/jobs").glob(f"*{name}.json")
            ]
            try:
                run(["git", "add", "--", *paths])
                if subprocess.run(
                    ["git", "diff", "--cached", "--quiet"], cwd=root, check=False
                ).returncode:
                    run(
                        [
                            "git",
                            "commit",
                            "-m",
                            "Record completed scaling campaign and verified contact-model release",
                        ]
                    )
                run(["git", "push", "origin", "main"])
                record(
                    "complete",
                    published=True,
                    commit=run(["git", "rev-parse", "HEAD"]).strip(),
                )
            except subprocess.CalledProcessError as error:
                record("complete", publication_error=str(error))
    except Exception as error:
        record("needs_inspection", error=str(error))
        raise


if __name__ == "__main__":
    main()
