"""Immutable eight-GPU trial followed by fixed-endpoint validation collection."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.pair_trials import collect


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--result", required=True)
    a = p.parse_args()
    contract = json.loads(Path("pair_contract.json").read_text())
    for path, sha in contract["files"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen source changed: " + path)
    cfg = yaml.safe_load(Path(a.config).read_text())
    if (
        hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
        != contract["config_sha256"]
    ):
        raise ValueError("Frozen config changed")
    if cfg.get("smoke_validation_limit"):
        raise ValueError("Smoke config is not a production trial")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
            "sparsa.train",
            "--config",
            a.config,
            "--out",
            a.out,
            "--init-from",
            a.encoder,
            "--auto-resume",
        ],
        check=True,
    )
    collect(a.out, cfg["steps"], a.result)


if __name__ == "__main__":
    main()
