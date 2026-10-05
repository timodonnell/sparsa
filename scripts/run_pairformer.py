"""Gate a long Pairformer run on real DDP resume and full validation checks.

The same training configuration and 300k schedule are used throughout. A
supervisor retry resumes the current stage; failed validation cannot silently
fall through to the long training run. Preflight metrics are kept separately.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sparsa.data import benchmark
from sparsa.train import storage
from sparsa.train_diffusion import load_checkpoint, write_json


def verify_preflight(summary, per, config):
    expected = {(r["dataset"], r["stem"]) for r in benchmark()}
    if (
        summary.get("metric_version") != "fixed-r-v2"
        or summary.get("split") != "eval-val"
        or summary.get("held_out_used") is not False
        or summary.get("proteins") != 97
        or summary.get("n_rollouts") != 100
        or summary.get("rollout_batch") != config["rollout_batch"]
        or summary.get("rollout_seed") != config["rollout_seed"]
        or len(per) != 97
        or set(zip(per.dataset, per.stem, strict=True)) != expected
        or not per.n_rollouts.eq(100).all()
        or not per.metric_version.eq("fixed-r-v2").all()
        or not per.eval_set.eq("eval-val").all()
    ):
        raise ValueError("Incomplete or mismatched preflight validation")
    for metric in ["oracle_r_precision", "consensus_r_precision"]:
        np.testing.assert_allclose(per[metric].mean(), summary[metric])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10000)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if config["model"].get("pairformer_layers") != 48:
        raise ValueError("This preflight is for the 48-block Pairformer campaign")
    fs, root = storage(args.out)
    preflight = args.out + "/preflight"
    launch = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.gpus}",
    ]
    train = launch + [
        "-m",
        "sparsa.train_diffusion",
        "--config",
        args.config,
        "--out",
        args.out,
        "--auto-resume",
        "--eval-every",
        str(args.eval_every),
    ]

    def status(phase):
        write_json(
            preflight + "/stage.json",
            {"phase": phase, "observed_utc": datetime.now(UTC).isoformat()},
        )
        print("PREFLIGHT_STAGE " + phase, flush=True)

    if not fs.exists(root + "/preflight/complete.json"):
        for boundary in (2, 4):
            current = (
                json.loads(fs.cat_file(root + "/latest.json"))["step"]
                if fs.exists(root + "/latest.json")
                else 0
            )
            if current < boundary:
                status("train_to_2" if boundary == 2 else "resume_to_4")
                subprocess.run(train + ["--stop-after", str(boundary)], check=True)
        latest = json.loads(fs.cat_file(root + "/latest.json"))
        state = load_checkpoint(latest["checkpoint"])
        if state["world_size"] != args.gpus or state["training_config"] != config:
            raise ValueError("Preflight checkpoint configuration mismatch")
        if len(state["rng"]) != args.gpus or state["step"] < 4:
            raise ValueError("Preflight did not resume through four steps")
        counts = [r["data_counts"] for r in state["rng"]]
        if (
            sum(r["proteins"] for r in counts)
            != state["step"] * args.gpus * config["logical_batch_size"]
        ):
            raise ValueError("Resumed data exposure mismatch")
        evidence = {
            "checkpoint": latest,
            "world_size": args.gpus,
            "all_rank_data_counts": counts,
            "all_rank_data_digests": [r["data_digest"] for r in state["rng"]],
            "optimizer_parameters": len(state["optimizer"]["state"]),
        }
        del state
        write_json(preflight + "/resume.json", evidence)
        status("full_validation_97x100")
        if not fs.exists(root + "/preflight/validation/summary.json"):
            subprocess.run(
                launch
                + [
                    "-m",
                    "sparsa.eval_diffusion_checkpoint",
                    "--checkpoint",
                    latest["checkpoint"],
                    "--out",
                    preflight + "/validation",
                    "--n-rollouts",
                    "100",
                    "--rollout-batch",
                    str(config["rollout_batch"]),
                    "--rollout-seed",
                    str(config["rollout_seed"]),
                ],
                check=True,
            )
        summary = json.loads(fs.cat_file(root + "/preflight/validation/summary.json"))
        if (
            summary["step"] != latest["step"]
            or summary["source_checkpoint"] != latest["checkpoint"]
        ):
            raise ValueError("Preflight validation used the wrong checkpoint")
        with fs.open(root + "/preflight/validation/per_protein.csv") as handle:
            per = pd.read_csv(handle)
        verify_preflight(summary, per, config)
        write_json(
            preflight + "/complete.json",
            {
                "completed_utc": datetime.now(UTC).isoformat(),
                "config_sha256": hashlib.sha256(
                    Path(args.config).read_bytes()
                ).hexdigest(),
                "resumed_through_step": latest["step"],
                "full_validation_proteins": 97,
                "rollouts": 100,
                "held_out_used": False,
                "metric_version": "fixed-r-v2",
            },
        )
    status("long_training")
    subprocess.run(train, check=True)
    status("complete")


if __name__ == "__main__":
    main()
