"""Distributed calibrated oracle evaluation of a saved diffusion checkpoint."""

import argparse
import json
import os

import torch
import torch.distributed as dist

from sparsa.data import benchmark
from sparsa.diffusion import (
    BinaryDiffusion,
    DiffusionModelConfig,
    TriangleDiffusionModel,
)
from sparsa.evaluate_diffusion import evaluate_records, summarize
from sparsa.train_diffusion import load_checkpoint, write_frame, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--rollout-batch", type=int, default=4)
    parser.add_argument("--rollout-seed", type=int, default=20260925)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--validation-limit", type=int, default=0)
    args = parser.parse_args()

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint evaluation requires an Iris GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)

    state = load_checkpoint(args.checkpoint)
    config = DiffusionModelConfig(**state["model_config"])
    model = TriangleDiffusionModel(config).to(device).eval()
    model.load_state_dict(state["ema"])
    training = state["training_config"]
    pos_weight = float(training.get("pos_weight", 4.0))
    schedule = BinaryDiffusion(
        config.diffusion_steps,
        tuple(training.get("contact_priors", (0.025, 0.016, 0.005))),
    ).to(device)
    records = benchmark()
    if args.validation_limit:
        records = sorted(records, key=lambda row: row["L"])[: args.validation_limit]
    local_proteins, local_rollouts = evaluate_records(
        model,
        schedule,
        records[rank::world],
        device,
        n_rollouts=args.n_rollouts,
        rollout_batch=args.rollout_batch,
        seed=args.rollout_seed,
        temperature=args.temperature,
        pos_weight=pos_weight,
    )
    proteins_by_rank = [None] * world if rank == 0 else None
    rollouts_by_rank = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local_proteins, proteins_by_rank, dst=0)
        dist.gather_object(local_rollouts, rollouts_by_rank, dst=0)
    else:
        proteins_by_rank, rollouts_by_rank = [local_proteins], [local_rollouts]
    if rank == 0:
        proteins = sorted(
            [row for shard in proteins_by_rank for row in shard],
            key=lambda row: (row["dataset"], row["stem"]),
        )
        rollouts = [row for shard in rollouts_by_rank for row in shard]
        result = summarize(proteins) | {
            "step": int(state["step"]),
            "split": "eval-val",
            "n_rollouts": args.n_rollouts,
            "rollout_seed": args.rollout_seed,
            "temperature": args.temperature,
            "pos_weight_logit_correction": pos_weight,
            "source_checkpoint": args.checkpoint,
            "world_size": world,
            "held_out_used": False,
        }
        write_json(args.out + "/summary.json", result)
        write_frame(args.out + "/per_protein.csv", proteins)
        write_frame(args.out + "/rollouts.csv", rollouts)
        print("CALIBRATED_ORACLE " + json.dumps(result), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
