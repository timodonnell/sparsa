"""Resumable DDP training for the triangle discrete-diffusion experiments."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from sparsa.data import ROOT, TeacherBatches, benchmark, file_sha256, inventory
from sparsa.diffusion import (
    BinaryDiffusion,
    DiffusionModelConfig,
    TriangleDiffusionModel,
)
from sparsa.evaluate_diffusion import evaluate_records, summarize
from sparsa.experiment import advance_data_digest, learning_rate_scale, microbatches
from sparsa.model import ALPHABET
from sparsa.train import contact_loss, contact_ranking_loss, storage


def write_bytes(uri, content):
    fs, path = storage(uri)
    if "://" not in uri:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    fs.pipe_file(path, content)


def write_json(uri, value):
    write_bytes(uri, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def save_checkpoint(uri, state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    write_bytes(uri, buffer.getvalue())


def load_checkpoint(uri):
    fs, path = storage(uri)
    with fs.open(path, "rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=False)


def write_frame(uri, rows):
    write_bytes(uri, pd.DataFrame(rows).to_csv(index=False).encode())


def prune_checkpoints(out, keep=2):
    fs, root = storage(out)
    paths = []
    for path in fs.glob(root + "/checkpoints/step-*.pt"):
        try:
            step = int(Path(path).stem.removeprefix("step-"))
        except ValueError:
            continue
        paths.append((step, path))
    for _, path in sorted(paths, reverse=True)[keep:]:
        fs.rm(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--smoke-validation-limit", type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke_validation_limit:
        cfg["smoke_validation_limit"] = args.smoke_validation_limit
    # Validate a fixed training horizon before creating any durable run marker.
    learning_rate_scale(cfg, 0)

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Diffusion training requires an Iris GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    if world > 1:
        dist.init_process_group("nccl", device_id=device)

    fs, out_path = storage(args.out)
    latest_exists = fs.exists(out_path + "/latest.json")
    provenance_exists = fs.exists(out_path + "/provenance.json")
    if args.auto_resume and latest_exists:
        args.resume = json.loads(fs.cat_file(out_path + "/latest.json"))["checkpoint"]
    elif args.auto_resume and provenance_exists:
        # An Iris retry before the first checkpoint may safely restart only the
        # exact same run.  This avoids poisoning a URI with a startup failure.
        existing = json.loads(fs.cat_file(out_path + "/provenance.json"))
        if existing.get("training_config") != cfg:
            raise ValueError("Existing pre-checkpoint run has a different configuration")
    elif provenance_exists and not args.resume:
        raise ValueError("Output run already exists; use auto-resume or a new name")

    seed = int(cfg.get("seed", 17))
    torch.manual_seed(seed)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    model_config = DiffusionModelConfig(**cfg["model"])
    model = TriangleDiffusionModel(model_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    schedule = BinaryDiffusion(
        model_config.diffusion_steps,
        tuple(cfg.get("contact_priors", (0.025, 0.016, 0.005))),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=cfg.get("weight_decay", 0.01),
        fused=True,
    )
    diffusion_rng = torch.Generator(device=device).manual_seed(seed + 7919 * rank + 101)
    step = 0
    data_states = {}
    data_digest = "00" * 32
    data_counts = {
        "residues": 0,
        "positive_pairs": 0,
        "proteins": 0,
        "afdb": 0,
        "esm": 0,
    }
    frozen_shards = None
    history = []

    if args.resume:
        state = load_checkpoint(args.resume)
        if state.get("format_version") != 3:
            raise ValueError("Not a resumable diffusion checkpoint")
        if state["model_config"] != asdict(model_config):
            raise ValueError("Resume architecture mismatch")
        if state["training_config"] != cfg:
            raise ValueError("Resume training configuration mismatch")
        if state["world_size"] != world:
            raise ValueError("Resume must preserve world size")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        local_rng = state["rng"][rank]
        data_states = local_rng["data_states"]
        data_digest = local_rng["data_digest"]
        data_counts = local_rng["data_counts"]
        torch.set_rng_state(local_rng["cpu"])
        torch.cuda.set_rng_state(local_rng["cuda"])
        diffusion_rng.set_state(local_rng["diffusion"])
        provenance_uri = args.resume.rsplit("/checkpoints/", 1)[0] + "/provenance.json"
        pfs, ppath = storage(provenance_uri)
        provenance = json.loads(pfs.cat_file(ppath))
        frozen_shards = provenance["source_files"]
        if fs.exists(out_path + "/training_log.json"):
            history = json.loads(fs.cat_file(out_path + "/training_log.json"))

    shards = (
        (frozen_shards or inventory(cfg.get("data_root", ROOT))) if rank == 0 else None
    )
    if world > 1:
        objects = [shards]
        dist.broadcast_object_list(objects, src=0)
        shards = objects[0]

    if rank == 0 and not args.resume:
        provenance = {
            "model_config": asdict(model_config),
            "training_config": cfg,
            "world_size": world,
            "parameters": sum(p.numel() for p in model.parameters()),
            "gpu": torch.cuda.get_device_name(),
            "torch_version": str(torch.__version__),
            "source_files": shards,
            "benchmark_sha256": {
                path.name: file_sha256(path)
                for path in Path("data/benchmark").glob("*")
            },
            "code_sha256": {
                str(path): file_sha256(path)
                for path in sorted(Path("sparsa").rglob("*.py"))
            },
            "iris_job": os.getenv("IRIS_JOB_ID"),
            "priority": "batch",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selection_split": "eval-val",
            "held_out_used": False,
        }
        write_json(args.out + "/provenance.json", provenance)
        print(
            "PROVENANCE "
            + json.dumps({k: v for k, v in provenance.items() if k != "source_files"}),
            flush=True,
        )

    logical_size = int(cfg["logical_batch_size"])
    micro_size = int(cfg["batch_size"])
    if logical_size % micro_size:
        raise ValueError("Logical batch must divide into equal microbatches")
    accumulation = logical_size // micro_size
    dataset = TeacherBatches(
        shards,
        logical_size,
        int(cfg["crop"]),
        seed,
        rank,
        world,
        consumed=step,
        limit_shards=cfg.get("limit_shards", 0),
        states=data_states,
        total_batches=int(cfg["steps"]),
        afdb_probability=cfg.get("afdb_probability", 0.5),
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=cfg.get("workers", 2),
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed + rank),
        multiprocessing_context="spawn" if cfg.get("workers", 2) else None,
    )
    batches = iter(loader)
    wrapped = (
        DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    )
    start_time = last_log = time.monotonic()
    steps = int(cfg["steps"])
    checkpoint_every = int(cfg.get("checkpoint_every", 1000))

    while step < steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scale = learning_rate_scale(cfg, step)
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * scale
        logical = next(batches)
        data_states[logical["worker_id"]] = logical["data_state"]
        data_digest = advance_data_digest(data_digest, logical)
        data_counts["residues"] += int((logical["tokens"] != 0).sum())
        data_counts["positive_pairs"] += int(
            (logical["targets"] * logical["mask"]).sum()
        )
        data_counts["proteins"] += len(logical["ids"])
        for source in ("afdb", "esm"):
            data_counts[source] += logical["sources"].count(source)
        positive_proteins = max(
            1,
            int(
                (((logical["targets"] > 0.5) & logical["mask"]).flatten(1).any(1)).sum()
            ),
        )
        total_loss = torch.zeros((), device=device)
        timestep_counts = torch.zeros(model_config.diffusion_steps, device=device)
        for micro, batch in enumerate(microbatches(logical, micro_size)):
            tokens, target, mask = [
                batch[key].to(device, non_blocking=True)
                for key in ("tokens", "targets", "mask")
            ]
            timestep = torch.randint(
                1,
                model_config.diffusion_steps + 1,
                (tokens.shape[0],),
                device=device,
                generator=diffusion_rng,
            )
            timestep_counts.scatter_add_(
                0, timestep - 1, torch.ones_like(timestep, dtype=torch.float)
            )
            noisy = schedule.sample_forward(target, tokens, timestep, diffusion_rng)
            if world > 1:
                wrapped.require_backward_grad_sync = micro == accumulation - 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = wrapped(tokens, noisy, timestep)
                loss = contact_loss(logits, target, mask, cfg.get("pos_weight", 4.0))
                if cfg.get("ranking_weight", 0.0):
                    loss = loss + cfg["ranking_weight"] * contact_ranking_loss(
                        logits,
                        target,
                        mask,
                        positive_proteins / accumulation,
                    )
                loss = loss / accumulation
            loss.backward()
            total_loss += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.get("grad_clip", 1.0)
        )
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradients at step {step}")
        optimizer.step()
        step += 1
        decay = min(cfg.get("ema_decay", 0.999), (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_lerp_(
                list(ema.parameters()), list(model.parameters()), 1 - decay
            )

        if step % cfg.get("log_every", 50) == 0:
            if world > 1:
                dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(timestep_counts, op=dist.ReduceOp.SUM)
            if rank == 0:
                now = time.monotonic()
                row = {
                    "step": step,
                    "loss": float(total_loss),
                    "grad_norm": float(norm),
                    "lr": cfg["lr"] * scale,
                    "seconds": now - start_time,
                    "seconds_per_step": (now - last_log) / cfg.get("log_every", 50),
                    "examples": step * logical_size * world,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "timestep_counts": timestep_counts.tolist(),
                    "rank0_data_digest": data_digest,
                    "rank0_data_counts": dict(data_counts),
                }
                history.append(row)
                print(json.dumps(row), flush=True)
                last_log = now

        save_now = step % checkpoint_every == 0 or step == steps
        if save_now:
            optimizer.zero_grad(set_to_none=True)
            local_rng = {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
                "diffusion": diffusion_rng.get_state(),
                "data_states": data_states,
                "data_digest": data_digest,
                "data_counts": data_counts,
            }
            states = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(local_rng, states, dst=0)
            else:
                states = [local_rng]
            if rank == 0:
                uri = args.out + f"/checkpoints/step-{step}.pt"
                save_checkpoint(
                    uri,
                    {
                        "format_version": 3,
                        "alphabet": ALPHABET,
                        "step": step,
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": asdict(model_config),
                        "training_config": cfg,
                        "world_size": world,
                        "rng": states,
                    },
                )
                write_json(args.out + "/latest.json", {"step": step, "checkpoint": uri})
                write_json(args.out + "/training_log.json", history)
                prune_checkpoints(args.out, cfg.get("keep_recovery_checkpoints", 2))
            if world > 1:
                dist.barrier()

    validation = benchmark()
    if cfg.get("smoke_validation_limit"):
        validation = sorted(validation, key=lambda row: row["L"])[
            : int(cfg["smoke_validation_limit"])
        ]
    local_proteins, local_rollouts = evaluate_records(
        ema,
        schedule,
        validation[rank::world],
        device,
        n_rollouts=int(cfg.get("validation_rollouts", 100)),
        rollout_batch=int(cfg.get("rollout_batch", 4)),
        seed=int(cfg.get("rollout_seed", 20260925)),
        temperature=float(cfg.get("rollout_temperature", 1.0)),
    )
    gathered_proteins = [None] * world if rank == 0 else None
    gathered_rollouts = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local_proteins, gathered_proteins, dst=0)
        dist.gather_object(local_rollouts, gathered_rollouts, dst=0)
    else:
        gathered_proteins, gathered_rollouts = [local_proteins], [local_rollouts]
    if rank == 0:
        proteins = sorted(
            [row for shard in gathered_proteins for row in shard],
            key=lambda row: (row["dataset"], row["stem"]),
        )
        rollouts = [row for shard in gathered_rollouts for row in shard]
        result = summarize(proteins) | {
            "step": step,
            "split": "eval-val",
            "n_rollouts": int(cfg.get("validation_rollouts", 100)),
            "seed": seed,
            "world_size": world,
            "held_out_used": False,
        }
        write_json(args.out + f"/validation/step-{step}.json", result)
        write_frame(args.out + f"/validation/step-{step}-per_protein.csv", proteins)
        write_frame(args.out + f"/validation/step-{step}-rollouts.csv", rollouts)
        print("ORACLE_VALIDATION " + json.dumps(result), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
