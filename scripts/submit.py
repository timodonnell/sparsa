"""Submit a root GPU job; enforce batch priority without implicit child jobs."""

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--out", help="Existing run URI for a replacement job; defaults to the job name"
    )
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=43200)
    parser.add_argument(
        "--memory-gb", type=int, help="Host RAM; defaults to 32 GiB per GPU"
    )
    parser.add_argument("--resume")
    parser.add_argument("--init-from")
    parser.add_argument(
        "--iris", default=os.environ.get("IRIS_BIN", ".tools/iris/bin/iris")
    )
    parser.add_argument(
        "--cluster-config",
        default="/home/bizon/git/marin-freshiris/lib/iris/config/cw-rno2a.yaml",
    )
    args = parser.parse_args()
    if args.gpus not in (1, 2, 4, 8):
        raise ValueError("Single-node training supports 1, 2, 4, or 8 GPUs")
    if args.memory_gb is not None and args.memory_gb <= 0:
        raise ValueError("Host memory must be positive")
    root = Path(__file__).resolve().parents[1]
    out = (
        args.out
        or f"s3://marin-us-east-02a/marin/protein-structure/sparsa/runs/{args.name}"
    )
    cmd = [
        args.iris,
        "--config",
        args.cluster_config,
        "job",
        "run",
        "--priority",
        "batch",
        "--enable-extra-resources",
        "--gpu",
        f"H100x{args.gpus}",
        "--cpu",
        str(args.gpus * 6),
        "--memory",
        f"{args.memory_gb or args.gpus * 32}GB",
        "--disk",
        "40GB",
        "--timeout",
        str(args.timeout),
        "--max-retries",
        "5",
        "--job-name",
        "sparsa-" + args.name,
        "--no-wait",
        "--",
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.gpus}",
        "-m",
        "sparsa.train",
        "--config",
        args.config,
        "--out",
        out,
        "--auto-resume",
    ]
    if args.resume:
        cmd.extend(["--resume", args.resume])
    if args.init_from:
        cmd.extend(["--init-from", args.init_from])
    result = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    record = {
        "name": args.name,
        "output": out,
        "command": cmd,
        "priority": "batch",
        "submitted_utc": datetime.now(UTC).isoformat(),
        "stdout": result.stdout,
    }
    path = root / "reports" / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{args.name}.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
