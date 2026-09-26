"""Submit a root CoreWeave diffusion job with explicit batch priority."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--smoke-validation-limit", type=int, default=0)
    parser.add_argument("--iris", default=".tools/iris/bin/iris")
    parser.add_argument(
        "--cluster-config",
        default="/home/bizon/git/marin-freshiris/lib/iris/config/cw-rno2a.yaml",
    )
    args = parser.parse_args()
    if args.gpus not in (1, 2, 4, 8):
        raise ValueError("Expected a single-node GPU count")
    root = Path(__file__).resolve().parents[1]
    out = args.out or (
        "s3://marin-us-east-02a/marin/protein-structure/sparsa/diffusion-v1/"
        + args.name
    )
    command = [
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
        f"{args.gpus * 64}GB",
        "--disk",
        "40GB",
        "--timeout",
        str(args.timeout),
        "--max-retries",
        str(args.max_retries),
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
        "sparsa.train_diffusion",
        "--config",
        args.config,
        "--out",
        out,
        "--auto-resume",
    ]
    if args.smoke_validation_limit:
        command += ["--smoke-validation-limit", str(args.smoke_validation_limit)]
    result = subprocess.run(
        command, cwd=root, check=True, capture_output=True, text=True
    )
    record = {
        "name": args.name,
        "output": out,
        "command": command,
        "priority": "batch",
        "submitted_utc": datetime.now(UTC).isoformat(),
        "stdout": result.stdout,
    }
    path = root / "reports" / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{args.name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(result.stdout)


if __name__ == "__main__":
    main()
