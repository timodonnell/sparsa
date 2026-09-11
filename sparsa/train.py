"""Resumable single-node DDP training; every Iris job is explicitly batch priority."""

import argparse
import copy
import io
import json
import math
import os
import random
import re
import time
from dataclasses import asdict
from pathlib import Path

import fsspec
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from sparsa.data import ROOT, TeacherBatches, benchmark, file_sha256, inventory
from sparsa.evaluate import evaluate
from sparsa.model import ALPHABET, ContactModel, ModelConfig


def storage(uri):
    """Bound storage retries so an unavailable endpoint cannot hang a GPU gang."""
    options = {}
    if uri.startswith("s3://"):
        options = json.loads(os.getenv("FSSPEC_S3", "{}"))
        options["config_kwargs"] = dict(
            options.get("config_kwargs", {}),
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": 3, "mode": "standard"},
        )
    return fsspec.core.url_to_fs(uri, **options)


def write_json(value, uri):
    fs, path = storage(uri)
    if "://" not in uri:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    fs.pipe_file(path, json.dumps(value, indent=2).encode())


def load_checkpoint(uri):
    fs, path = storage(uri)
    with fs.open(path, "rb") as f:
        return torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)


def save_checkpoint(uri, state):
    data = io.BytesIO()
    torch.save(state, data)
    fs, path = storage(uri)
    if "://" not in uri:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with fs.open(path, "wb") as f:
        f.write(data.getbuffer())


def prune_recovery_checkpoints(out, keep, protected=()):
    """Keep all validated weights and the newest recovery-only checkpoints."""
    if not isinstance(keep, int) or keep < 1:
        raise ValueError("Keep at least one recovery checkpoint")
    fs, root = storage(out)
    validated = {
        int(Path(p).stem.removeprefix("step-"))
        for p in fs.glob(root + "/validation/step-*.json")
    }
    candidates = []
    protected_paths = {storage(p)[1] for p in protected if p}
    for path in fs.glob(root + "/checkpoints/step-*.pt"):
        match = re.search(r"/step-(\d+)\.pt$", path)
        if match and int(match[1]) not in validated:
            candidates.append((int(match[1]), path))
    removed = []
    for _, path in sorted(candidates, reverse=True)[keep:]:
        if path not in protected_paths:
            fs.rm(path)
            removed.append(path)
    return removed


def contact_loss(logits, targets, mask, pos_weight=1.0):
    per_pair = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets,
        reduction="none",
        pos_weight=logits.new_tensor(pos_weight).float(),
    )
    counts = mask.sum((1, 2))
    if (counts == 0).any():
        raise ValueError("A training protein has no supervised pairs")
    return ((per_pair * mask).sum((1, 2)) / counts).mean()


def contact_ranking_loss(logits, targets, mask):
    """Per-protein KL from uniform true contacts to a softmax over valid pairs.

    Positive gradients are normalized by true-contact count rather than L²;
    negative gradients focus on highly ranked false contacts. Proteins without
    positive contacts contribute only to the accompanying BCE objective.
    """
    values = logits.float()
    positive = mask & (targets > 0.5)
    count = positive.sum((1, 2))
    valid = count > 0
    normalizer = values.masked_fill(~mask, -torch.inf).flatten(1).logsumexp(1)
    mean_positive = (values * positive).sum((1, 2)) / count.clamp_min(1)
    kl = normalizer - mean_positive - count.clamp_min(1).float().log()
    return torch.where(valid, kl, 0.0).sum() / valid.sum().clamp_min(1)


def architecture_config(values):
    return {
        k: v
        for k, v in asdict(ModelConfig(**values)).items()
        if k != "gradient_checkpointing"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument(
        "--init-from",
        help="Start a new curriculum phase from Sparsa weights, resetting optimizer and data stream",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("compile_pair_blocks", False):
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = cfg.get("compile_cache_limit", 128)
    eval_every = cfg.get("eval_every", 1000)
    checkpoint_every = cfg.get("checkpoint_every", eval_every)
    if eval_every <= 0 or checkpoint_every <= 0:
        raise ValueError("Evaluation and checkpoint intervals must be positive")
    if args.resume and args.init_from:
        raise ValueError("Choose resume or init-from")
    fs, out_path = storage(args.out)
    if args.auto_resume and fs.exists(out_path + "/latest.json"):
        args.resume = json.loads(fs.cat_file(out_path + "/latest.json"))["checkpoint"]
        args.init_from = None
    if fs.exists(out_path + "/provenance.json") and not (
        args.resume or args.auto_resume
    ):
        raise ValueError("Output run already exists; use a new name or explicit resume")
    rank, local_rank, world = (
        int(os.getenv("RANK", "0")),
        int(os.getenv("LOCAL_RANK", "0")),
        int(os.getenv("WORLD_SIZE", "1")),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Production training requires an Iris GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    seed = cfg.get("seed", 17)
    torch.manual_seed(seed)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    config = ModelConfig(**cfg["model"])
    model = ContactModel(config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=cfg.get("weight_decay", 0.01),
        fused=True,
    )
    step, best, best_checkpoint = 0, -1.0, None
    data_states = {}
    frozen_shards = None
    execution_changes = {}
    if cfg.get("initialize_max_distance") is not None and not (
        args.init_from or args.resume
    ):
        raise ValueError("Distant-bucket initialization requires a source checkpoint")
    if cfg.get("grow_from_ema", False) and not (args.init_from or args.resume):
        raise ValueError("Model growth requires a source checkpoint")
    if args.init_from and not args.resume:
        state = load_checkpoint(args.init_from)
        if cfg.get("grow_from_ema", False):
            from sparsa.grow import grow_from_ema

            growth = grow_from_ema(model, state)
            ema.load_state_dict(model.state_dict())
            if rank == 0:
                print("GROWTH " + json.dumps(growth), flush=True)
        elif architecture_config(state["model_config"]) != architecture_config(
            asdict(config)
        ):
            raise ValueError("Initialization architecture mismatch")
        else:
            model.load_state_dict(state.get("model", state["ema"]))
            ema.load_state_dict(state["ema"])
        edge = cfg.get("initialize_max_distance")
        if edge is not None:
            if edge >= state["training_config"]["crop"]:
                raise ValueError("Initialization edge exceeds the source training crop")
            model.initialize_distant_buckets(edge)
            ema.initialize_distant_buckets(edge)
        del state
    if args.resume:
        state = load_checkpoint(args.resume)
        if state.get("format_version") != 2:
            raise ValueError(
                "Use init-from for pilot checkpoints without resumable data cursors"
            )
        if architecture_config(state["model_config"]) != architecture_config(
            asdict(config)
        ):
            raise ValueError("Resume architecture mismatch")
        for field, before, after in (
            (
                "gradient_checkpointing",
                state["model_config"].get("gradient_checkpointing", True),
                config.gradient_checkpointing,
            ),
            (
                "compile_pair_blocks",
                state["training_config"].get("compile_pair_blocks", False),
                cfg.get("compile_pair_blocks", False),
            ),
            (
                "compile_cache_limit",
                state["training_config"].get("compile_cache_limit", 128),
                cfg.get("compile_cache_limit", 128),
            ),
        ):
            if before != after:
                execution_changes[field] = {"before": before, "after": after}
        for field in (
            "batch_size",
            "accumulation",
            "crop",
            "workers",
            "seed",
            "data_root",
            "limit_shards",
            "lr",
            "weight_decay",
            "pos_weight",
            "ema_decay",
            "afdb_probability",
            "grow_from_ema",
            "ranking_weight",
        ):
            if state["training_config"].get(field) != cfg.get(field):
                raise ValueError(f"Resume changes data stream: {field}")
        if state["world_size"] != world:
            raise ValueError("Resume must preserve world size for stream replay")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        step, best = state["step"], state["best_val"]
        best_checkpoint = state["best_checkpoint"]
        data_states = state["rng"][rank]["data_states"]
        source_provenance = (
            args.resume.rsplit("/checkpoints/", 1)[0] + "/provenance.json"
        )
        source_fs, source_path = storage(source_provenance)
        frozen_shards = json.loads(source_fs.cat_file(source_path))["source_files"]
        torch.set_rng_state(state["rng"][rank]["cpu"])
        torch.cuda.set_rng_state(state["rng"][rank]["cuda"])
        del state
    if rank == 0:
        shards = frozen_shards or inventory(cfg.get("data_root", ROOT))
    else:
        shards = None
    if world > 1:
        objects = [shards]
        dist.broadcast_object_list(objects, src=0)
        shards = objects[0]
    if rank == 0:
        provenance = {
            "model_config": asdict(config),
            "training_config": cfg,
            "world_size": world,
            "parameters": sum(p.numel() for p in model.parameters()),
            "gpu": torch.cuda.get_device_name(),
            "torch_version": str(torch.__version__),
            "source_files": shards,
            "benchmark_sha256": {
                p.name: file_sha256(p) for p in Path("data/benchmark").glob("*")
            },
            "resume": args.resume,
            "init_from": args.init_from,
            "code_sha256": {
                str(p): file_sha256(p) for p in sorted(Path("sparsa").rglob("*.py"))
            },
            "iris_job": os.getenv("IRIS_JOB_ID"),
            "priority": "batch",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "execution_changes": execution_changes,
        }
        provenance_file = (
            f"/resume-step-{step}-provenance.json"
            if args.resume and fs.exists(out_path + "/provenance.json")
            else "/provenance.json"
        )
        write_json(provenance, args.out + provenance_file)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in provenance.items()
                    if k not in ("source_files", "benchmark_sha256")
                }
            ),
            flush=True,
        )
    accumulation = cfg.get("accumulation", 1)
    dataset = TeacherBatches(
        shards,
        cfg["batch_size"],
        cfg["crop"],
        seed,
        rank,
        world,
        consumed=step * accumulation,
        limit_shards=cfg.get("limit_shards", 0),
        states=data_states,
        total_batches=cfg["steps"] * accumulation,
        afdb_probability=cfg.get("afdb_probability", 0.5),
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=cfg.get("workers", 4),
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed + rank),
        multiprocessing_context="spawn" if cfg.get("workers", 4) else None,
    )
    batches = iter(loader)
    if cfg.get("compile_pair_blocks", False):
        # Keep EMA eager and parameter names unchanged for portable checkpoints.
        for block in model.pairs:
            block.forward = torch.compile(block.forward, dynamic=True)
    wrapped = (
        DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    )
    val = benchmark()
    history, start_time, last_log = [], time.monotonic(), time.monotonic()
    if args.resume and rank == 0 and fs.exists(out_path + "/training_log.json"):
        history = json.loads(fs.cat_file(out_path + "/training_log.json"))
    initial_step = step
    steps = cfg["steps"]
    while step < steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        warmup = cfg.get("warmup", 500)
        progress = max(0, (step - warmup) / max(1, steps - warmup))
        scale = min(1.0, (step + 1) / max(1, warmup)) * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        )
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * scale
        total_loss = 0.0
        for micro in range(accumulation):
            batch = next(batches)
            data_states[batch["worker_id"]] = batch["data_state"]
            tokens, target, mask = [
                batch[k].to(device, non_blocking=True)
                for k in ("tokens", "targets", "mask")
            ]
            if world > 1:
                wrapped.require_backward_grad_sync = micro == accumulation - 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = wrapped(tokens)
                loss = contact_loss(logits, target, mask, cfg.get("pos_weight", 1.0))
                if cfg.get("ranking_weight", 0.0):
                    loss = loss + cfg["ranking_weight"] * contact_ranking_loss(
                        logits, target, mask
                    )
                loss = loss / accumulation
            loss.backward()
            total_loss += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradients at step {step}")
        optimizer.step()
        step += 1
        decay = min(cfg.get("ema_decay", 0.999), (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_lerp_(
                list(ema.parameters()), list(model.parameters()), 1 - decay
            )
        if step % cfg.get("log_every", 25) == 0:
            if world > 1:
                dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
            if rank == 0:
                now = time.monotonic()
                record = {
                    "step": step,
                    "loss": float(total_loss),
                    "grad_norm": float(norm),
                    "lr": cfg["lr"] * scale,
                    "seconds": now - start_time,
                    "seconds_per_step": (now - last_log) / cfg.get("log_every", 25),
                    "examples": step * cfg["batch_size"] * accumulation * world,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
                }
                if cfg.get("compile_pair_blocks", False):
                    from torch._dynamo.utils import counters

                    record["compiled_graphs"] = counters["stats"]["unique_graphs"]
                history.append(record)
                print(json.dumps(record), flush=True)
                # Emit to the task log immediately; flush history to S3 at
                # checkpoints, keeping network writes out of the training loop.
                last_log = now
        evaluate_now = step % eval_every == 0 or step == steps
        if evaluate_now or step % checkpoint_every == 0:
            if world > 1:
                dist.barrier()
            result = None
            improved = False
            if evaluate_now and rank == 0:
                result = evaluate(
                    ema,
                    val,
                    device,
                    label=f"sparsa-step-{step}",
                    pos_weight=cfg.get("pos_weight", 1.0),
                )
                result.update(step=step, seconds=time.monotonic() - start_time)
                print("VALIDATION " + json.dumps(result), flush=True)
                write_json(result, args.out + f"/validation/step-{step}.json")
            if evaluate_now and world > 1:
                objects = [result]
                dist.broadcast_object_list(objects, src=0)
                result = objects[0]
            if evaluate_now:
                improved = result["r_precision"] > best
                best = max(best, result["r_precision"])
            rng = {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
                "data_states": data_states,
            }
            states = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(rng, states, dst=0)
            else:
                states = [rng]
            if rank == 0:
                uri = args.out + f"/checkpoints/step-{step}.pt"
                if improved:
                    best_checkpoint = uri
                state = {
                    "format_version": 2,
                    "alphabet": ALPHABET,
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": asdict(config),
                    "training_config": cfg,
                    "world_size": world,
                    "rng": states,
                    "best_val": best,
                    "best_checkpoint": best_checkpoint,
                }
                save_checkpoint(uri, state)
                write_json(history, args.out + "/training_log.json")
                write_json(
                    {"checkpoint": uri, "step": step, "validation": result},
                    args.out + "/latest.json",
                )
                if improved:
                    write_json(
                        {"checkpoint": uri, "step": step, "validation": result},
                        args.out + "/best.json",
                    )
                print(
                    "CHECKPOINT "
                    + json.dumps(
                        {"step": step, "validated": evaluate_now, "checkpoint": uri}
                    ),
                    flush=True,
                )
                if cfg.get("keep_recovery_checkpoints") is not None:
                    try:
                        removed = prune_recovery_checkpoints(
                            args.out,
                            cfg["keep_recovery_checkpoints"],
                            (uri, best_checkpoint),
                        )
                        if removed:
                            print("PRUNED_RECOVERY " + json.dumps(removed), flush=True)
                    except OSError as error:
                        print(
                            "RECOVERY_PRUNE_RETRY " + type(error).__name__, flush=True
                        )
            if world > 1:
                dist.barrier()
    if rank == 0:
        write_json(
            {
                "step": step,
                "best_validation_r_precision": best,
                "elapsed_seconds": time.monotonic() - start_time,
                "initial_step": initial_step,
            },
            args.out + "/complete.json",
        )
    del batches, loader
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
