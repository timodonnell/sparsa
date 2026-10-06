"""Resumable 16-H100 Protenix pilot: warmup, shared-parent branches, fixed-val.

One gang runs the two branches in sequence so queued work holds no idle GPUs.
Every process stage is fail-closed. Iris retries recover the last durable stage.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime

from scripts.run_multinode_diffusion import iris_identity, rendezvous, torchrun_command
from sparsa.train import storage
from sparsa.train_diffusion import inherited_counts, load_checkpoint, write_json


def stage_plan(out, frozen_config, finetune_config, warmup=1000, final=6000):
    def train(config, name, stop, parent=None):
        cmd = [
            "-m",
            "sparsa.train_diffusion",
            "--config",
            config,
            "--out",
            out + "/" + name,
            "--auto-resume",
            "--stop-after",
            str(stop),
        ]
        if parent:
            cmd += ["--branch-from", parent]
        return cmd

    warm = out + f"/warmup/checkpoints/step-{warmup}.pt"
    gate = out + "/warmup/checkpoints/step-4.pt"
    stages = [
        ("warmup-2", train(frozen_config, "warmup", 2)),
        ("warmup-4", train(frozen_config, "warmup", 4)),
    ]
    for arm, config in [("frozen", frozen_config), ("finetune", finetune_config)]:
        for step in [5, 6]:
            stages.append(
                (f"gate-{arm}-{step}", train(config, "preflight/" + arm, step, gate))
            )
    stages += [
        ("warmup", train(frozen_config, "warmup", warmup)),
        (
            "warmup-validation",
            [
                "-m",
                "sparsa.eval_diffusion_checkpoint",
                "--checkpoint",
                warm,
                "--out",
                out + "/warmup-validation",
                "--n-rollouts",
                "100",
                "--rollout-batch",
                "4",
                "--rollout-seed",
                "20260925",
            ],
        ),
        ("finetune", train(finetune_config, "finetune", final, warm)),
        ("frozen", train(frozen_config, "frozen", final, warm)),
    ]
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frozen-config", default="configs/protenix_v1_frozen.yaml")
    parser.add_argument(
        "--finetune-config", default="configs/protenix_v1_finetune.yaml"
    )
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    args = parser.parse_args()
    info = iris_identity(os.environ)
    if info["nodes"] != args.nodes or args.nodes * args.gpus_per_node != 16:
        raise ValueError("Pilot configurations require 16 GPUs (global batch 128)")
    os.environ.setdefault("SPARSA_CHECKPOINT_CACHE", "/tmp/sparsa-checkpoint-cache")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    fs, root = storage(args.out)
    # Freeze decisions once per attempt; all nodes execute exactly the same stages.
    stages = stage_plan(args.out, args.frozen_config, args.finetune_config)
    job_key = hashlib.sha256(info["job"].encode()).hexdigest()[:16]
    plan_uri = args.out + f"/control/{job_key}-{info['attempt']}/plan.json"
    _, plan_path = storage(plan_uri)
    if info["rank"] == 0:
        write_json(
            plan_uri,
            {
                "model_preflight": not fs.exists(root + "/preflight/model.json"),
                "stages": [
                    (name, cmd)
                    for name, cmd in stages
                    if not fs.exists(root + "/completed/" + name + ".json")
                ],
            },
        )
    deadline = time.monotonic() + 1800
    while not fs.exists(plan_path):
        if time.monotonic() > deadline:
            raise TimeoutError("Missing pilot stage plan")
        time.sleep(2)
    plan = json.loads(fs.cat_file(plan_path))

    def run_stage(name, cmd):
        endpoint = rendezvous(info, args.out, name)
        if info["rank"] == 0:
            write_json(
                args.out + "/status.json",
                {
                    "stage": name,
                    "started_utc": datetime.now(UTC).isoformat(),
                    "job": info["job"],
                    "attempt": info["attempt"],
                },
            )
        print("PILOT_STAGE " + name, flush=True)
        started = time.monotonic()
        subprocess.run(
            torchrun_command(args.nodes, args.gpus_per_node, info["rank"], endpoint)
            + cmd,
            check=True,
        )
        if info["rank"] == 0:
            write_json(
                args.out + "/completed/" + name + ".json",
                {
                    "completed_utc": datetime.now(UTC).isoformat(),
                    "seconds": time.monotonic() - started,
                    "gpus": 16,
                },
            )

    run_stage("collective", ["scripts/check_distributed.py", "--world-size", "16"])
    if plan["model_preflight"]:
        if info["rank"] == 0:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.check_protenix",
                    "--config",
                    args.frozen_config,
                    "--out",
                    args.out + "/preflight/model.json",
                ],
                check=True,
            )
        # A distributed barrier also propagates a node-zero preflight failure.
        run_stage("model-ready", ["scripts/check_distributed.py", "--world-size", "16"])
    for name, cmd in plan["stages"]:
        if name == "warmup" and info["rank"] == 0:
            evidence = {}
            for arm in ["frozen", "finetune"]:
                checkpoint = args.out + "/preflight/" + arm + "/checkpoints/step-6.pt"
                state = load_checkpoint(checkpoint)
                assert state["step"] == 6 and state["world_size"] == 16
                assert inherited_counts(state)["proteins"] == 6 * 128
                assert state["branch_metadata"]["parent_step"] == 4
                evidence[arm] = {
                    "checkpoint": checkpoint,
                    "data_digests": [r["data_digest"] for r in state["rng"]],
                    "optimizer_states": len(state["optimizer"]["state"]),
                }
                del state
            assert (
                evidence["frozen"]["data_digests"]
                == evidence["finetune"]["data_digests"]
            )
            assert (
                evidence["finetune"]["optimizer_states"]
                > evidence["frozen"]["optimizer_states"]
            )
            write_json(args.out + "/preflight/branch-resume.json", evidence)
        run_stage(name, cmd)
    if info["rank"] == 0:
        write_json(
            args.out + "/status.json",
            {
                "stage": "complete",
                "completed_utc": datetime.now(UTC).isoformat(),
                "held_out_used": False,
            },
        )


if __name__ == "__main__":
    main()
