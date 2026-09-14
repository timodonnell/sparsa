"""Persistent, budgeted pair-trunk screening and two-seed confirmation.

GPU workers use immutable source copies. Selection never reads held-out results.
Journal deterministic job IDs before submission; reuse them after uncertain RPCs.
"""

import argparse
import fcntl
import hashlib
import json
import math
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from research.backend import TERMINAL, IrisBackend
from scripts.pair_trials import ARMS, training_config

ROOT = Path(__file__).resolve().parents[1]
NAME = "pair-trunk-v1-20260914"
RUN_NAME = "pair-v1-20260914"
PREFIX = "s3://marin-us-east-02a/marin/protein-structure/sparsa/" + NAME


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Backend(IrisBackend):
    def command(self, argv, cwd=None, timeout=60):
        argv = list(argv)
        if "run" in argv and "--max-retries" in argv:
            argv[argv.index("--max-retries") + 1] = "2"
            if "--gpu" in argv and argv[argv.index("--gpu") + 1] == "H100x8":
                argv[argv.index("--memory") + 1] = "512GB"
            elif "--verify-only" in argv:
                argv[argv.index("--memory") + 1] = "64GB"
                argv[argv.index("--cpu") + 1] = "8"
            journal = (
                self.campaign
                / "submissions"
                / (argv[argv.index("--job-name") + 1] + ".json")
            )
            if journal.exists():
                record = json.loads(journal.read_text())
                write(journal, record | {"command": argv})
        return super().command(argv, cwd, timeout)


def cost(profile, steps):
    # Conservative single-GPU profile extrapolation; actual attempt accounting
    # governs every subsequent admission and the independent guard.
    return profile["estimated_seconds_per_step"] * steps * 8 / 3600 * 1.25 + 4


def shortlist(results):
    scored = {
        k: v["validation"]["r_precision"] for k, v in results.items() if k != "C0"
    }
    if not scored:
        raise ValueError("No viable architectural candidate")
    chosen = [max(scored, key=scored.get)]
    ambitious = [k for k in ("A1", "J1") if k in scored and k not in chosen]
    rest = [k for k in scored if k not in chosen]
    if ambitious and chosen[0] not in ("A1", "J1"):
        chosen.append(max(ambitious, key=scored.get))
    elif rest:
        chosen.append(max(rest, key=scored.get))
    return chosen


def validate_result(
    result, arm, seed, step, expected_source, expected_config, reference=None
):
    if result["step"] != step or result["validation"]["proteins"] != 97:
        raise ValueError("Wrong trial endpoint/coverage")
    cfg = result["provenance"]["training_config"]
    if cfg != expected_config or cfg["seed"] != seed:
        raise ValueError("Training config mismatch")
    if result["provenance"]["world_size"] != 8:
        raise ValueError("World size mismatch")
    if result["provenance"]["code_sha256"] != expected_source:
        raise ValueError("Trial source mismatch")
    rows = result["per_protein"]
    from sparsa.data import benchmark

    stems = {r["stem"] for r in benchmark()}
    for region in ("all", "long"):
        if set(rows[region]) != stems:
            raise ValueError("Validation identity mismatch")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in rows[region].values()):
            raise ValueError("Invalid score")
        metric = "r_precision" if region == "all" else "long_r_precision"
        if not math.isclose(
            float(np.mean(list(rows[region].values()))),
            result["validation"][metric],
            abs_tol=1e-8,
        ):
            raise ValueError("Summary disagrees with per-protein scores")
    if reference:
        if result["source_files_sha256"] != reference["source_files_sha256"]:
            raise ValueError("Teacher inventory changed")
        if result["exposure"] != reference["exposure"]:
            raise ValueError("Logical training exposure differs")
    return result


def comparison(candidate, baseline):
    all_deltas = []
    long_deltas = []
    seed_gains = []
    for seed in ("17", "37"):
        c, b = candidate[seed]["per_protein"], baseline[seed]["per_protein"]
        stems = sorted(b["all"])
        delta = np.array([c["all"][s] - b["all"][s] for s in stems])
        all_deltas.append(delta)
        seed_gains.append(float(delta.mean()))
        long_deltas.append([c["long"][s] - b["long"][s] for s in stems])
    means = np.mean(all_deltas, axis=0)
    rng = np.random.default_rng(20260914)
    boot = means[rng.integers(0, len(means), (5000, len(means)))].mean(1)
    gain = float(means.mean())
    long_gain = float(np.mean(long_deltas))
    return {
        "seed_gains": seed_gains,
        "mean_gain": gain,
        "long_gain": long_gain,
        "descriptive_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "passes_target": min(seed_gains) > 0 and gain >= 0.005 and long_gain >= -0.002,
        "interval_scope": "Descriptive under adaptive validation reuse",
    }


def initialize(path):
    if (path / "state.json").exists():
        return
    path.mkdir(parents=True, exist_ok=True)
    base = path / "base"
    base.mkdir()
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for name in files:
        src = ROOT / name
        if src.is_file() and not name.startswith("reports/"):
            dst = base / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    settings = {
        "name": RUN_NAME,
        "run_name": RUN_NAME,
        "output_prefix": PREFIX,
        "iris": str(ROOT / ".tools/iris/bin/iris"),
        "iris_python": str(ROOT / ".tools/iris/bin/python"),
        "cluster_config": "/home/bizon/git/marin-freshiris/lib/iris/config/cw-rno2a.yaml",
        "kubeconfig": "/home/bizon/.kube/coreweave-iris",
        "kube_context": "marin-rn02a_RNO2A",
    }
    write(path / "settings.json", settings)
    hashes = {
        str(p.relative_to(base)): digest(p)
        for group in ("sparsa", "scripts", "research", "data/benchmark")
        for p in (base / group).rglob("*")
        if p.is_file() and "__pycache__" not in str(p)
    }
    write(path / "contract.json", {"files": hashes})
    write(
        path / "state.json",
        {
            "stage": "profiling",
            "jobs": {},
            "results": {},
            "failures": {},
            "profile_job": "/bizon/sparsa-ar-" + RUN_NAME + "-profile-v4",
            "prepare_job": "/bizon/sparsa-ar-" + RUN_NAME + "-prepare",
            "kernel_profile_job": "/bizon/sparsa-ar-" + RUN_NAME + "-profile-v3",
        },
    )


class Campaign:
    def __init__(self, path):
        self.path = path
        self.state = json.loads((path / "state.json").read_text())
        self.backend = Backend(ROOT, path)
        self.prefix = self.backend.prefix
        self.contract = json.loads((path / "contract.json").read_text())

    def record(self, **kw):
        self.state.update(kw, updated_utc=datetime.now(UTC).isoformat())
        write(self.path / "state.json", self.state)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in self.state.items()
                    if k not in ("results", "profile", "encoder", "jobs", "failures")
                }
            ),
            flush=True,
        )

    def register(self, job):
        path = self.path / "budget.json"
        b = json.loads(path.read_text())
        if job not in b["jobs"]:
            b["jobs"].append(job)
            write(path, b)

    def spent(self):
        p = self.path / "guard/status.json"
        if not p.exists():
            raise RuntimeError("Compute guard has no observation")
        status = json.loads(p.read_text())
        age = (
            datetime.now(UTC) - datetime.fromisoformat(status["observed_utc"])
        ).total_seconds()
        if age > 300:
            raise RuntimeError("Compute guard observation is stale")
        return status["enforcement_h100_hours"]

    def submit(self, key, gpus, args, workspace=None, timeout=7200):
        job = self.prefix + key
        if key not in self.state["jobs"]:
            self.state["jobs"][key] = job
            self.register(job)
            self.record(active=key)
        self.backend.submit(job, workspace or self.path / "base", gpus, timeout, args)
        return job

    def retrieve(self, key, uri):
        job = self.state["jobs"][key]
        status = self.backend.status(job)
        if status["state"] == "succeeded":
            return self.backend.result(uri)
        if status["state"] in TERMINAL:
            self.state["failures"][key] = status
            write(self.path / "failures" / f"{key}.json", status)
            raise RuntimeError("Terminal failed job: " + key)
        return None

    def profile_step(self):
        for key in ("profile_job", "prepare_job"):
            status = self.backend.status(self.state[key])
            if status["state"] != "succeeded":
                if status["state"] in TERMINAL:
                    raise RuntimeError("Failed prerequisite " + key)
                return
        profile = self.backend.result(PREFIX + "/crop256/profile.json")
        encoder = self.backend.result(PREFIX + "/encoder.json")
        if not profile or not encoder:
            return
        if not profile.get("complete"):
            raise ValueError("Incomplete profile")
        feasible = [a for a in ARMS if "selected" in profile["arms"][a]]
        if "C0" not in feasible or not any(a in feasible for a in ("A1", "J1")):
            raise RuntimeError("Required families infeasible")
        # The design allows staging families after measured costs. Retain core
        # C0/C2 and the joint hypothesis before spending on expensive controls.
        priority = [
            a for a in ("C0", "C2", "J1", "A1", "D1", "T1", "C1") if a in feasible
        ]
        selected = []
        estimate = 0
        for arm in priority:
            hours = cost(profile["arms"][arm]["selected"], 25000)
            if estimate + hours <= 350:
                selected.append(arm)
                estimate += hours
        if not {"C0", "J1"}.issubset(selected):
            raise RuntimeError("Screen allowance cannot support C0 and J1")
        self.record(
            profile=profile,
            encoder=encoder,
            screen_arms=selected,
            deferred_arms=[a for a in ARMS if a not in selected],
            estimated_screen_hours=estimate,
            stage="preflight",
        )
        write(self.path / "profile.json", profile)
        write(self.path / "encoder.json", encoder)

    def preflight_step(self):
        key = self.state.get("preflight_key", "preflight")
        if self.state.get("preflight_verify_from"):
            self.submit(
                key,
                0,
                [
                    "-m",
                    "scripts.pair_smoke",
                    "--verify-only",
                    "--out",
                    self.state["preflight_verify_from"],
                    "--result",
                    PREFIX + "/" + key + "/result.json",
                ],
            )
        else:
            self.submit_preflight_training(key)
        result = self.retrieve(key, PREFIX + "/" + key + "/result.json")
        if result:
            if not result.get("passed"):
                raise ValueError("Failed preflight")
            write(self.path / "preflight.json", result)
            self.record(stage="screening", active=None)

    def submit_preflight_training(self, key):
        self.submit(
            key,
            8,
            [
                "-m",
                "scripts.pair_smoke",
                "--out",
                PREFIX + "/" + key,
                "--encoder",
                PREFIX + "/encoder.json",
                "--batch-size",
                str(self.state["profile"]["arms"]["J1"]["selected"]["batch_size"]),
                "--profile",
                PREFIX + "/crop256/profile.json",
            ],
            timeout=7200,
        )

    def trial(self, arm, seed, step):
        key = f"{arm.lower()}-s{seed}-{step}"
        if key in self.state["results"]:
            return True
        profile = self.state["profile"]["arms"][arm]["selected"]
        cfg = training_config(arm, seed, step, profile) | {
            "init_sha256": self.state["encoder"]["sha256"]
        }
        if key not in self.state["jobs"]:
            additional = step - (25000 if step == 100000 and seed == 17 else 0)
            estimate = cost(profile, additional)
            if self.spent() + estimate > 925:
                self.record(stage="budget_limited", unadmitted=key)
                return False
        workspace = self.path / "work" / key
        if not workspace.exists():
            shutil.copytree(self.path / "base", workspace)
            (workspace / "pair_trial.yaml").write_text(yaml.safe_dump(cfg))
            write(
                workspace / "pair_contract.json",
                self.contract
                | {"config_sha256": digest(workspace / "pair_trial.yaml")},
            )
        run = PREFIX + f"/runs/{arm.lower()}-s{seed}"
        uri = PREFIX + f"/results/{key}.json"
        timeout = min(172800, max(7200, int(cost(profile, step) * 3600 / 8 * 2)))
        self.submit(
            key,
            8,
            [
                "-m",
                "scripts.pair_job",
                "--config",
                "pair_trial.yaml",
                "--out",
                run,
                "--encoder",
                self.state["encoder"]["checkpoint"],
                "--result",
                uri,
            ],
            workspace,
            timeout,
        )
        result = self.retrieve(key, uri)
        if result is None:
            return False
        expected_source = {
            k: v
            for k, v in self.contract["files"].items()
            if k.startswith("sparsa/") and k.endswith(".py")
        }
        reference = self.state["results"].get(f"c0-s{seed}-{step}")
        validate_result(result, arm, seed, step, expected_source, cfg, reference)
        write(self.path / "results" / f"{key}.json", result)
        self.state["results"][key] = result
        # Replace optimistic single-device timings with real distributed
        # training/validation/checkpoint time before admitting confirmations.
        observed = result.get("observed_seconds_per_step", 0)
        profile["estimated_seconds_per_step"] = max(
            profile["estimated_seconds_per_step"], observed
        )
        self.record(active=None)
        return True

    def tick(self):
        if self.state["stage"] == "profiling":
            self.profile_step()
            return
        if self.state["stage"] == "preflight":
            self.preflight_step()
            return
        if self.state["stage"] == "screening":
            for arm in self.state["screen_arms"]:
                if not self.trial(arm, 17, 25000):
                    return
            results = {
                a: self.state["results"][f"{a.lower()}-s17-25000"]
                for a in self.state["screen_arms"]
            }
            candidates = shortlist(results)
            profiles = self.state["profile"]["arms"]

            def total(arms):
                return sum(cost(profiles[a]["selected"], 175000) for a in ["C0", *arms])

            while candidates and self.spent() + total(candidates) > 925:
                # Keep the highest validation-scoring affordable single design;
                # no partially completed two-seed comparison is presented as confirmed.
                affordable = [a for a in candidates if self.spent() + total([a]) <= 925]
                candidates = (
                    [
                        max(
                            affordable,
                            key=lambda a: results[a]["validation"]["r_precision"],
                        )
                    ]
                    if affordable
                    else []
                )
                if len(candidates) <= 1:
                    break
            if not candidates:
                self.record(
                    stage="budget_limited", reason="No full two-seed confirmation fits"
                )
                return
            self.record(
                stage="confirmation",
                finalists=candidates,
                estimated_confirmation_hours=total(candidates),
            )
            return
        if self.state["stage"] == "confirmation":
            for arm in ["C0", *self.state["finalists"]]:
                for seed in (17, 37):
                    if not self.trial(arm, seed, 100000):
                        return
            base = {str(s): self.state["results"][f"c0-s{s}-100000"] for s in (17, 37)}
            comparisons = {
                a: comparison(
                    {
                        str(s): self.state["results"][f"{a.lower()}-s{s}-100000"]
                        for s in (17, 37)
                    },
                    base,
                )
                for a in self.state["finalists"]
            }
            winner = max(
                self.state["finalists"],
                key=lambda a: np.mean(
                    [
                        self.state["results"][f"{a.lower()}-s{s}-100000"]["validation"][
                            "r_precision"
                        ]
                        for s in (17, 37)
                    ]
                ),
            )
            self.record(
                stage="complete",
                comparisons=comparisons,
                validation_winner=winner,
                promotion=winner if comparisons[winner]["passes_target"] else None,
            )
            self.finish()

    def finish(self):
        b = json.loads((self.path / "budget.json").read_text())
        b["keep_watching"] = False
        write(self.path / "budget.json", b)
        self.backend.close()
        lines = [
            "# Pair-trunk campaign results",
            "",
            f"Status: {self.state['stage']}. Validation only; no held-out evaluation.",
            "",
            "| Arm | Seed | Steps | R | Long R |",
            "|---|---:|---:|---:|---:|",
        ]
        for key, r in self.state["results"].items():
            arm, seed, step = key.split("-")
            v = r["validation"]
            lines.append(
                f"| {arm} | {seed} | {step} | {v['r_precision']:.6f} | {v['long_r_precision']:.6f} |"
            )
        lines += [
            "",
            json.dumps(self.state.get("comparisons", {}), indent=2),
            "",
            f"Deferred arms: {self.state.get('deferred_arms', [])}.",
            "Intervals are descriptive after adaptive validation reuse. No claim about final 1.8B scaling is made.",
        ]
        (self.path / "REPORT.md").write_text("\n".join(lines) + "\n")
        dest = ROOT / "reports/pair_trunk_v1/campaign"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("REPORT.md", "profile.json", "encoder.json", "preflight.json"):
            if (self.path / name).exists():
                shutil.copy2(self.path / name, dest / name)
        write(
            dest / "summary.json",
            {k: v for k, v in self.state.items() if k not in ("profile", "results")},
        )
        for p in (
            (self.path / "results").glob("*.json")
            if (self.path / "results").exists()
            else []
        ):
            shutil.copy2(p, dest / p.name)
        subprocess.run(
            ["git", "add", "reports/pair_trunk_v1/campaign"], cwd=ROOT, check=True
        )
        staged = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True
        ).splitlines()
        if any(not p.startswith("reports/pair_trunk_v1/campaign/") for p in staged):
            raise RuntimeError(
                "Unrelated staged changes prevent automatic report commit"
            )
        if staged:
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "Record validation-only pair-trunk campaign results",
                ],
                cwd=ROOT,
                check=True,
            )
            subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
        self.record(published=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["init", "run"])
    p.add_argument("--campaign", default="outputs/" + NAME)
    a = p.parse_args()
    path = Path(a.campaign).resolve()
    if a.mode == "init":
        initialize(path)
        return
    lock = (path / "coordinator.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    c = Campaign(path)
    while True:
        if c.state["stage"] == "needs_inspection":
            raise RuntimeError(
                "Resolve the recorded failure before resuming this campaign"
            )
        if c.state["stage"] in ("complete", "budget_limited"):
            if not c.state.get("published"):
                c.finish()
            return
        try:
            c.tick()
        except (subprocess.SubprocessError, OSError) as error:
            print(
                json.dumps(
                    {
                        "event": "infrastructure_retry",
                        "error_type": type(error).__name__,
                    }
                ),
                flush=True,
            )
        except Exception as error:
            c.record(stage="needs_inspection", error=str(error))
            raise
        time.sleep(30)


if __name__ == "__main__":
    main()
