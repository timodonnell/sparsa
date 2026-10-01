"""Submit a calibrated diffusion oracle evaluation at batch priority."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--rollout-batch", type=int, default=2)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--rollout-seed", type=int, default=20260925)
    parser.add_argument("--self-condition-guidance", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--iris", default=".tools/iris/bin/iris")
    parser.add_argument(
        "--cluster-config",
        default="/home/bizon/git/marin-freshiris/lib/iris/config/cw-rno2a.yaml",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
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
        f"{args.gpu_type}x{args.gpus}",
        "--cpu",
        str(args.gpus * 4),
        "--memory",
        f"{args.gpus * 32}GB",
        "--disk",
        "20GB",
        "--timeout",
        str(args.timeout),
        "--max-retries",
        "2",
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
        "sparsa.eval_diffusion_checkpoint",
        "--checkpoint",
        args.checkpoint,
        "--out",
        args.out,
        "--n-rollouts",
        str(args.n_rollouts),
        "--rollout-batch",
        str(args.rollout_batch),
        "--rollout-seed",
        str(args.rollout_seed),
        "--self-condition-guidance",
        str(args.self_condition_guidance),
    ]
    if args.validation_limit:
        command += ["--validation-limit", str(args.validation_limit)]
    result = subprocess.run(
        command, cwd=root, check=True, capture_output=True, text=True
    )
    record = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "output": args.out,
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
