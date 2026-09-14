"""Eight-rank real-data recovery equivalence using the full joint architecture."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from scripts.pair_trials import training_config
from sparsa.train import load_checkpoint, storage, write_json


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--profile")
    a = p.parse_args()
    fs, path = storage(a.encoder)
    encoder = json.loads(fs.cat_file(path))
    profile = {"batch_size": a.batch_size}
    if a.profile:
        pf, pp = storage(a.profile)
        profile = json.loads(pf.cat_file(pp))["arms"]["J1"]["selected"]
    cfg = training_config("J1", 17, 2, profile)
    cfg.update(
        init_sha256=encoder["sha256"], eval_every=1000, checkpoint_every=2, log_every=1
    )
    config = Path("/tmp/pair-smoke.yaml")

    def train(run, steps, resume=False):
        cfg["steps"] = steps
        config.write_text(yaml.safe_dump(cfg))
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
            "sparsa.train",
            "--config",
            str(config),
            "--out",
            run,
            "--smoke-validation-limit",
            "2",
        ]
        cmd += (
            ["--resume", run + "/checkpoints/step-2.pt"]
            if resume
            else ["--init-from", encoder["checkpoint"]]
        )
        subprocess.run(cmd, check=True)

    train(a.out + "/resumed", 2)
    train(a.out + "/resumed", 4, True)
    train(a.out + "/continuous", 4)
    x = load_checkpoint(a.out + "/resumed/checkpoints/step-4.pt")
    y = load_checkpoint(a.out + "/continuous/checkpoints/step-4.pt")
    for field in ("model", "ema", "optimizer", "rng"):
        equal(x[field], y[field])
    result = {
        "passed": True,
        "scope": "Full J1 eight-H100 real-data save/resume equivalence; not architecture scoring",
        "compared": ["model", "ema", "optimizer", "all_rank_rng_and_data_cursors"],
        "bitwise_equal": True,
        "steps": 4,
    }
    write_json(result, a.out + "/result.json")
    print("PAIR_PREFLIGHT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
