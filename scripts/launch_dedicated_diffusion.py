"""Launch one resumable eight-GPU diffusion run as a supervised user service.

Run on the dedicated node from its source snapshot. Node connection details are
managed outside this repository. User lingering must already be enabled.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--node-label", required=True)
    parser.add_argument("--smoke-validation-limit", type=int, default=0)
    args = parser.parse_args()
    if not re.fullmatch(r"sparsa-[a-z0-9-]+", args.unit):
        raise ValueError("Expected a project-scoped service name")
    root = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    cfg = yaml.safe_load(config.read_text())
    if cfg.get("execution_backend") != "dedicated":
        raise ValueError("Expected a dedicated-node training config")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run",
        "--user",
        f"--unit={args.unit}",
        f"--property=WorkingDirectory={root}",
        "--property=Restart=on-failure",
        "--property=RestartSec=60",
        "--property=StartLimitIntervalSec=1800",
        "--property=StartLimitBurst=3",
        f"--property=StandardOutput=append:{args.out}/train.log",
        f"--property=StandardError=append:{args.out}/train.log",
        "--setenv=OMP_NUM_THREADS=4",
        "--setenv=PYTHONUNBUFFERED=1",
        "--setenv=NCCL_DEBUG=WARN",
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "sparsa.train_diffusion",
        "--config",
        str(config),
        "--out",
        str(args.out),
        "--auto-resume",
        "--eval-every",
        "10000",
    ]
    if args.smoke_validation_limit:
        command += ["--smoke-validation-limit", str(args.smoke_validation_limit)]
    subprocess.run(command, check=True)
    record = {
        "node_label": args.node_label,
        "service": args.unit,
        "config": str(config.relative_to(root))
        if config.is_relative_to(root)
        else str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "source_snapshot": str(root),
        "output": str(args.out),
        "gpus": 8,
        "priority": "dedicated",
        "submitted_utc": datetime.now(UTC).isoformat(),
        "command": command,
    }
    (args.out / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
