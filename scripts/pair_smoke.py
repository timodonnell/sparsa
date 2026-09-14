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


def equal(a, b, *, atol=0, rtol=0, stats=None):
    if isinstance(a, torch.Tensor):
        if stats is not None:
            stats["elements"] += a.numel()
            if not torch.equal(a, b):
                stats["unequal_elements"] += int((a != b).sum())
                stats["max_absolute_difference"] = max(
                    stats["max_absolute_difference"], float((a - b).abs().max())
                )
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k], atol=atol, rtol=rtol, stats=stats)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y, atol=atol, rtol=rtol, stats=stats)
    else:
        assert a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--encoder")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--result")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--profile")
    a = p.parse_args()
    if a.verify_only:
        verify(a.out, a.result)
        return
    if not a.encoder:
        p.error("Training preflight requires --encoder")
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
    verify(a.out, a.result)


def verify(out, result_uri=None):
    torch.set_num_threads(8)
    x = load_checkpoint(out + "/resumed/checkpoints/step-4.pt")
    y = load_checkpoint(out + "/continuous/checkpoints/step-4.pt")
    equal(x["rng"], y["rng"])
    differences = {}
    for field in ("model", "ema", "optimizer"):
        stats = {"elements": 0, "unequal_elements": 0, "max_absolute_difference": 0.0}
        equal(x[field], y[field], atol=1e-8, rtol=1e-6, stats=stats)
        differences[field] = stats
    result = {
        "passed": True,
        "scope": "Full J1 eight-H100 real-data save/resume equivalence; not architecture scoring",
        "compared": ["model", "ema", "optimizer", "all_rank_rng_and_data_cursors"],
        "bitwise_equal": all(s["unequal_elements"] == 0 for s in differences.values()),
        "rng_and_data_bitwise_equal": True,
        "tensor_tolerance": {"atol": 1e-8, "rtol": 1e-6},
        "tensor_differences": differences,
        "source_runs": out,
        "steps": 4,
    }
    write_json(result, result_uri or out + "/result.json")
    print("PAIR_PREFLIGHT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
