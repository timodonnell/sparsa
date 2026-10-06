"""Resumable DDP training for the triangle discrete-diffusion experiments."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import io
import json
import math
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
    build_diffusion_model,
)
from sparsa.evaluate_diffusion import (
    COLLECTIVE_TIMEOUT,
    METRIC_POLICY,
    METRIC_VERSION,
    VALIDATION_PARTITION,
    evaluate_records,
    summarize,
    validation_shards,
)
from sparsa.experiment import advance_data_digest, learning_rate_scale, microbatches
from sparsa.model import ALPHABET
from sparsa.train import contact_loss, contact_ranking_loss, storage


def write_bytes(uri, content):
    fs, path = storage(uri)
    if "://" not in uri or uri.startswith("file://"):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
        try:
            with temporary.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        fs.pipe_file(path, content)


def write_json(uri, value):
    write_bytes(uri, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def save_checkpoint(uri, state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    write_bytes(uri, buffer.getvalue())


def load_checkpoint(uri):
    fs, path = storage(uri)
    cache = os.getenv("SPARSA_CHECKPOINT_CACHE")
    if cache and uri.startswith("s3://"):
        directory = Path(cache)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (hashlib.sha256(uri.encode()).hexdigest() + ".pt")
        # Eight local ranks share one download. Checkpoints are immutable URIs.
        with (directory / ".download.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                partial = target.with_suffix(".partial")
                fs.get_file(path, str(partial))
                os.replace(partial, target)
            handle = target.open("rb")
            for old in sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime)[
                :-2
            ]:
                if old != target:
                    old.unlink(missing_ok=True)
        with handle:
            return torch.load(handle, map_location="cpu", weights_only=False)
    with fs.open(path, "rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=False)


def write_frame(uri, rows):
    write_bytes(uri, pd.DataFrame(rows).to_csv(index=False).encode())


def prune_checkpoints(out, keep=2, keep_every=0):
    fs, root = storage(out)
    paths = []
    for path in fs.glob(root + "/checkpoints/step-*.pt"):
        try:
            step = int(Path(path).stem.removeprefix("step-"))
        except ValueError:
            continue
        paths.append((step, path))
    for step, path in sorted(paths, reverse=True)[keep:]:
        if not keep_every or step % keep_every:
            fs.rm(path)


def validation_due(step, steps, every, complete=False):
    """Return whether this checkpoint still needs fixed-split evaluation."""
    milestone = step > 0 and (step == steps or (every > 0 and step % every == 0))
    return milestone and not complete


def training_configs_compatible(saved, current):
    """Allow recovery cadence changes without allowing a scientific recipe change."""
    saved, current = dict(saved), dict(current)
    saved.pop("checkpoint_every", None)
    current.pop("checkpoint_every", None)
    return saved == current


def validate_parallel_fork(state, cfg, world):
    """A new stochastic branch with unchanged model, optimizer recipe and batch."""
    saved = state["training_config"]
    runtime_keys = {
        "logical_batch_size",
        "data_root",
        "execution_backend",
        "stream_seed",
        "ranking_group_size",
    }
    if {k: v for k, v in saved.items() if k not in runtime_keys} != {
        k: v for k, v in cfg.items() if k not in runtime_keys
    }:
        raise ValueError("Parallel fork may change only runtime and stream settings")
    if (
        saved["logical_batch_size"] * state["world_size"]
        != cfg["logical_batch_size"] * world
    ):
        raise ValueError("Parallel fork must preserve global batch size")
    if saved["logical_batch_size"] * saved.get("ranking_group_size", 1) != cfg[
        "logical_batch_size"
    ] * cfg.get("ranking_group_size", 1):
        raise ValueError("Parallel fork must preserve ranking normalization group size")
    if cfg.get("stream_seed", cfg["seed"]) == saved.get("stream_seed", saved["seed"]):
        raise ValueError("Parallel fork requires an explicit new stream seed")
    if len(state["rng"]) != state["world_size"]:
        raise ValueError("Incomplete parent checkpoint rank state")


def inherited_counts(state):
    counts = dict(state.get("inherited_data_counts", {}))
    for rank in state["rng"]:
        for key, value in rank["data_counts"].items():
            counts[key] = counts.get(key, 0) + value
    expected = (
        state["step"]
        * state["world_size"]
        * state["training_config"]["logical_batch_size"]
    )
    if counts.get("proteins") != expected:
        raise ValueError("Parent checkpoint data exposure mismatch")
    return counts


def relocate_sources(provenance, cfg):
    old = (
        provenance["training_config"]
        .get("data_root", ROOT)
        .removeprefix("file://")
        .rstrip("/")
    )
    new = cfg.get("data_root", ROOT).rstrip("/")
    result = {}
    for source, paths in provenance["source_files"].items():
        paths = [path.removeprefix("file://") for path in paths]
        if not all(path.startswith(old + "/") for path in paths):
            raise ValueError("Source file outside frozen parent data root")
        result[source] = [new + path[len(old) :] for path in paths]
    return result


def ranking_normalization(positive_counts, rank, group_size, accumulation):
    """Keep the original 16-protein ranking denominator after distributing micros."""
    start = rank // group_size * group_size
    count = sum(positive_counts[start : start + group_size])
    return max(1, count) / group_size / accumulation


def validation_prefix(out, step):
    """Keep corrected evaluations separate from legacy sparse precision."""
    return out + f"/validation-{METRIC_VERSION}/step-{step}"


def run_validation(ema, schedule, cfg, out, step, rank, world, device):
    """Run and durably record one distributed eval-val oracle evaluation."""
    validation = benchmark()
    if cfg.get("smoke_validation_limit"):
        validation = sorted(validation, key=lambda row: row["L"])[
            : int(cfg["smoke_validation_limit"])
        ]
    n_rollouts = int(cfg.get("validation_rollouts", 100))
    rollout_seed = int(cfg.get("rollout_seed", 20260925))
    temperature = float(cfg.get("rollout_temperature", 1.0))
    self_condition_guidance = float(cfg.get("rollout_self_condition_guidance", 1.0))
    pos_weight = float(cfg.get("pos_weight", 4.0))
    local_records = validation_shards(validation, world)[rank]
    print(
        "VALIDATION_START "
        + json.dumps({"step": step, "rank": rank, "proteins": len(local_records)}),
        flush=True,
    )

    def progress(completed, total, row):
        print(
            "VALIDATION_PROGRESS "
            + json.dumps(
                {
                    "step": step,
                    "rank": rank,
                    "completed": completed,
                    "total": total,
                    "dataset": row["dataset"],
                    "stem": row["stem"],
                    "length": row["L"],
                    "inference_seconds": row["inference_seconds"],
                }
            ),
            flush=True,
        )

    local_proteins, local_rollouts = evaluate_records(
        ema,
        schedule,
        local_records,
        device,
        n_rollouts=n_rollouts,
        rollout_batch=int(cfg.get("rollout_batch", 4)),
        seed=rollout_seed,
        temperature=temperature,
        pos_weight=pos_weight,
        self_condition_guidance=self_condition_guidance,
        progress_callback=progress,
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
            "n_rollouts": n_rollouts,
            "rollout_seed": rollout_seed,
            "rollout_batch": int(cfg.get("rollout_batch", 4)),
            "temperature": temperature,
            "self_condition_guidance": self_condition_guidance,
            "pos_weight_logit_correction": pos_weight,
            "source_checkpoint": out + f"/checkpoints/step-{step}.pt",
            "world_size": world,
            "validation_partition": VALIDATION_PARTITION,
            "held_out_used": False,
        }
        prefix = validation_prefix(out, step)
        # The summary is the completion marker and must be written last. A retry
        # after a partial evaluation will therefore repeat it rather than skip it.
        write_frame(prefix + "-per_protein.csv", proteins)
        write_frame(prefix + "-rollouts.csv", rollouts)
        write_json(prefix + ".json", result)
        print("ORACLE_VALIDATION " + json.dumps(result), flush=True)
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument(
        "--fork-from",
        help="Branch weights, EMA, optimizer and step; start fresh data/noise streams",
    )
    parser.add_argument("--validate-on-resume", action="store_true")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run fixed eval-val oracle evaluation every N steps; zero means final only",
    )
    parser.add_argument("--smoke-validation-limit", type=int, default=0)
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stop at this absolute step, preserving the configured training schedule",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke_validation_limit:
        cfg["smoke_validation_limit"] = args.smoke_validation_limit
    if args.eval_every < 0:
        raise ValueError("Evaluation cadence must be non-negative")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("Stop step must be positive")
    # Validate a fixed training horizon before creating any durable run marker.
    learning_rate_scale(cfg, 0)

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Diffusion training requires a CUDA GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    if world > 1:
        dist.init_process_group("nccl", device_id=device, timeout=COLLECTIVE_TIMEOUT)

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
            raise ValueError(
                "Existing pre-checkpoint run has a different configuration"
            )
        if (
            args.fork_from
            and (existing.get("fork_metadata") or {}).get("parent_checkpoint")
            != args.fork_from
        ):
            raise ValueError("Existing pre-checkpoint run has a different parent")
    elif provenance_exists and not args.resume:
        raise ValueError("Output run already exists; use auto-resume or a new name")

    seed = int(cfg.get("seed", 17))
    stream_seed = int(cfg.get("stream_seed", seed))
    torch.manual_seed(seed)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    model_config = DiffusionModelConfig(**cfg["model"])
    model = build_diffusion_model(model_config).to(device)
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
    diffusion_rng = torch.Generator(device=device).manual_seed(
        stream_seed + 7919 * rank + 101
    )
    conditioning_rng = torch.Generator(device=device).manual_seed(
        stream_seed + 7919 * rank + 303
    )
    self_condition_probability = float(cfg.get("self_condition_probability", 0.5))
    if not 0 <= self_condition_probability <= 1:
        raise ValueError("Self-conditioning probability must be in [0, 1]")
    self_condition_mode = cfg.get("self_condition_mode", "same_t")
    if self_condition_mode not in {"same_t", "rollout"}:
        raise ValueError("Self-conditioning mode must be same_t or rollout")
    if self_condition_mode == "rollout" and not model_config.self_conditioning:
        raise ValueError("Rollout self-conditioning requires self-conditioning")
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
    inherited_data_counts = {}
    fork_metadata = None
    history = []

    checkpoint = args.resume or args.fork_from
    if checkpoint:
        state = load_checkpoint(checkpoint)
        if state.get("format_version") != 3:
            raise ValueError("Not a resumable diffusion checkpoint")
        if asdict(DiffusionModelConfig(**state["model_config"])) != asdict(
            model_config
        ):
            raise ValueError("Resume architecture mismatch")
        if args.resume and not training_configs_compatible(
            state["training_config"], cfg
        ):
            raise ValueError("Resume training configuration mismatch")
        if args.resume and state["world_size"] != world:
            raise ValueError("Resume must preserve world size")
        if not args.resume:
            validate_parallel_fork(state, cfg, world)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        if args.resume:
            local_rng = state["rng"][rank]
            data_states = local_rng["data_states"]
            data_digest = local_rng["data_digest"]
            data_counts = local_rng["data_counts"]
            torch.set_rng_state(local_rng["cpu"])
            torch.cuda.set_rng_state(local_rng["cuda"])
            diffusion_rng.set_state(local_rng["diffusion"])
            if "conditioning" in local_rng:
                conditioning_rng.set_state(local_rng["conditioning"])
            inherited_data_counts = state.get("inherited_data_counts", {})
            fork_metadata = state.get("fork_metadata")
            if (
                args.fork_from
                and (fork_metadata or {}).get("parent_checkpoint") != args.fork_from
            ):
                raise ValueError("Resumed branch has a different parent checkpoint")
        else:
            inherited_data_counts = inherited_counts(state)
            fork_metadata = {
                "parent_checkpoint": checkpoint,
                "parent_world_size": state["world_size"],
                "parent_step": step,
                "stream_seed": stream_seed,
                "stochastic_streams_restarted": True,
                "inherited_data_counts": inherited_data_counts,
            }
        provenance_uri = checkpoint.rsplit("/checkpoints/", 1)[0] + "/provenance.json"
        pfs, ppath = storage(provenance_uri)
        provenance = json.loads(pfs.cat_file(ppath))
        frozen_shards = (
            provenance["source_files"]
            if args.resume
            else relocate_sources(provenance, cfg)
        )
        if (
            not args.resume
            and rank == 0
            and inventory(cfg.get("data_root", ROOT)) != frozen_shards
        ):
            raise ValueError(
                "Fork target inventory differs from frozen parent inventory"
            )
        if fs.exists(out_path + "/training_log.json"):
            history = json.loads(fs.cat_file(out_path + "/training_log.json"))
        del state

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
            "execution_backend": cfg.get("execution_backend", "iris"),
            "priority": "dedicated"
            if cfg.get("execution_backend") == "dedicated"
            else "batch",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selection_split": "eval-val",
            "held_out_used": False,
            "fork_metadata": fork_metadata,
        }
        write_json(args.out + "/provenance.json", provenance)
        print(
            "PROVENANCE "
            + json.dumps({k: v for k, v in provenance.items() if k != "source_files"}),
            flush=True,
        )

    if rank == 0:
        write_json(
            args.out + "/runtime.json",
            {
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "resumed_from": args.resume,
                "fork_metadata": fork_metadata,
                "resumed_step": step,
                "checkpoint_every": int(cfg.get("checkpoint_every", 1000)),
                "world_size": world,
                "collective_timeout_seconds": COLLECTIVE_TIMEOUT.total_seconds(),
                "validation_partition": VALIDATION_PARTITION,
                "code_sha256": {
                    str(path): file_sha256(path)
                    for path in sorted(Path("sparsa").rglob("*.py"))
                },
            },
        )
        write_json(
            args.out + "/evaluation_policy.json",
            {
                **METRIC_POLICY,
                "every_steps": args.eval_every,
                "split": "eval-val",
                "n_rollouts": int(cfg.get("validation_rollouts", 100)),
                "rollout_seed": int(cfg.get("rollout_seed", 20260925)),
                "validation_partition": VALIDATION_PARTITION,
                "collective_timeout_seconds": COLLECTIVE_TIMEOUT.total_seconds(),
                "self_condition_guidance": float(
                    cfg.get("rollout_self_condition_guidance", 1.0)
                ),
                "held_out_used": False,
            },
        )

    logical_size = int(cfg["logical_batch_size"])
    micro_size = int(cfg["batch_size"])
    if logical_size % micro_size:
        raise ValueError("Logical batch must divide into equal microbatches")
    accumulation = logical_size // micro_size
    ranking_group_size = int(cfg.get("ranking_group_size", 1))
    if ranking_group_size < 1 or world % ranking_group_size:
        raise ValueError("Ranking groups must partition the distributed world")
    dataset = TeacherBatches(
        shards,
        logical_size,
        int(cfg["crop"]),
        stream_seed,
        rank,
        world,
        consumed=step - (fork_metadata["parent_step"] if fork_metadata else 0),
        limit_shards=cfg.get("limit_shards", 0),
        states=data_states,
        total_batches=int(cfg["steps"])
        - (fork_metadata["parent_step"] if fork_metadata else 0),
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
    last_log_step = step
    steps = int(cfg["steps"])
    stop_step = min(steps, args.stop_after) if args.stop_after is not None else steps
    checkpoint_every = int(cfg.get("checkpoint_every", 1000))
    if args.eval_every and args.eval_every % checkpoint_every:
        raise ValueError("Evaluation cadence must be divisible by checkpoint cadence")

    if validation_due(step, steps, args.eval_every) or (
        args.validate_on_resume and args.resume and step > 0
    ):
        validation_marker = validation_prefix(out_path, step) + ".json"
        complete = fs.exists(validation_marker) if rank == 0 else None
        if world > 1:
            completed = [complete]
            dist.broadcast_object_list(completed, src=0)
            complete = completed[0]
        if not complete:
            run_validation(ema, schedule, cfg, args.out, step, rank, world, device)
            last_log = time.monotonic()

    while step < stop_step:
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
        positive_proteins = int(
            (((logical["targets"] > 0.5) & logical["mask"]).flatten(1).any(1)).sum()
        )
        if ranking_group_size > 1:
            counts = torch.tensor([positive_proteins], device=device, dtype=torch.long)
            all_counts = torch.empty(world, device=device, dtype=torch.long)
            dist.all_gather_into_tensor(all_counts, counts)
            rank_normalizer = ranking_normalization(
                all_counts.tolist(), rank, ranking_group_size, accumulation
            )
        else:
            rank_normalizer = max(1, positive_proteins) / accumulation
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
            self_condition = None
            use_self_condition = model_config.self_conditioning and bool(
                torch.rand((), device=device, generator=conditioning_rng)
                < self_condition_probability
            )
            if use_self_condition and self_condition_mode == "rollout":
                previous_timestep = (timestep + 1).clamp_max(
                    model_config.diffusion_steps
                )
                previous_noisy = schedule.sample_forward(
                    target, tokens, previous_timestep, diffusion_rng
                )
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    preliminary = model(tokens, previous_noisy, previous_timestep)
                    self_condition = (
                        preliminary.float() - math.log(cfg.get("pos_weight", 4.0))
                    ).sigmoid()
                rolled_noisy = schedule.sample_previous(
                    previous_noisy,
                    self_condition,
                    tokens,
                    previous_timestep,
                    diffusion_rng,
                )
                # At the terminal timestep there is no preceding reverse step;
                # retain standard same-t self-conditioning for those examples.
                noisy = torch.where(
                    (timestep < model_config.diffusion_steps)[:, None, None],
                    rolled_noisy,
                    previous_noisy,
                )
            else:
                noisy = schedule.sample_forward(target, tokens, timestep, diffusion_rng)
                if use_self_condition:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        preliminary = model(tokens, noisy, timestep)
                        self_condition = (
                            preliminary.float() - math.log(cfg.get("pos_weight", 4.0))
                        ).sigmoid()
            if world > 1:
                wrapped.require_backward_grad_sync = micro == accumulation - 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = wrapped(tokens, noisy, timestep, self_condition)
                loss = contact_loss(logits, target, mask, cfg.get("pos_weight", 4.0))
                if cfg.get("ranking_weight", 0.0):
                    loss = loss + cfg["ranking_weight"] * contact_ranking_loss(
                        logits,
                        target,
                        mask,
                        rank_normalizer,
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

        if step % cfg.get("log_every", 50) == 0 or step == stop_step:
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
                    "seconds_per_step": (now - last_log) / (step - last_log_step),
                    "examples": step * logical_size * world,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "timestep_counts": timestep_counts.tolist(),
                    "rank0_data_digest": data_digest,
                    "rank0_data_counts": dict(data_counts),
                }
                history.append(row)
                print(json.dumps(row), flush=True)
                last_log = now
                last_log_step = step

        save_now = (
            step % checkpoint_every == 0
            or step == steps
            or step == stop_step
            or step in cfg.get("extra_checkpoint_steps", ())
        )
        if save_now:
            optimizer.zero_grad(set_to_none=True)
            local_rng = {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
                "diffusion": diffusion_rng.get_state(),
                "conditioning": conditioning_rng.get_state(),
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
                        "inherited_data_counts": inherited_data_counts,
                        "fork_metadata": fork_metadata,
                    },
                )
                write_json(args.out + "/latest.json", {"step": step, "checkpoint": uri})
                write_json(args.out + "/training_log.json", history)
                prune_checkpoints(
                    args.out,
                    cfg.get("keep_recovery_checkpoints", 2),
                    cfg.get("keep_checkpoint_every", 0),
                )
            if world > 1:
                dist.barrier()

        if validation_due(step, steps, args.eval_every):
            marker = validation_prefix(out_path, step) + ".json"
            complete = fs.exists(marker) if rank == 0 else None
            if world > 1:
                completed = [complete]
                dist.broadcast_object_list(completed, src=0)
                complete = completed[0]
            if not complete:
                run_validation(ema, schedule, cfg, args.out, step, rank, world, device)
                last_log = time.monotonic()
                last_log_step = step

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
