"""Start a training phase from a completed run's validation-selected weights."""

import argparse
import json
import sys

from sparsa.train import main as train_main
from sparsa.train import storage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()
    fs, root = storage(args.source_run)
    if not fs.exists(root + "/complete.json"):
        raise RuntimeError("The source training run must be complete")
    completion = json.loads(fs.cat_file(root + "/complete.json"))
    selected = json.loads(fs.cat_file(root + "/best.json"))
    if (
        abs(
            selected["validation"]["r_precision"]
            - completion["best_validation_r_precision"]
        )
        > 1e-12
    ):
        raise RuntimeError("Source checkpoint pointer is stale")
    sys.argv = [
        sys.argv[0],
        "--config",
        args.config,
        "--out",
        args.out,
        "--init-from",
        selected["checkpoint"],
    ]
    if args.auto_resume:
        sys.argv.append("--auto-resume")
    train_main()


if __name__ == "__main__":
    main()
