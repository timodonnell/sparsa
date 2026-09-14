"""GPU profiling and durable fixed-endpoint validation for pair-trunk trials."""

import argparse
import copy
import gc
import hashlib
import io
import json
import time
from dataclasses import asdict

import pandas as pd
import torch

from sparsa.experiment import ENCODER_PREFIXES
from sparsa.model import ALPHABET, ContactModel, ModelConfig
from sparsa.train import (
    contact_loss,
    contact_ranking_loss,
    storage,
    write_json,
)

SOURCE = "s3://marin-us-east-02a/marin/protein-structure/sparsa/runs/scale-rank456m50-20260911/checkpoints/step-3000.pt"
ARMS = ("C0", "C1", "C2", "D1", "T1", "A1", "J1")


def model_config(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    cfg = ModelConfig(
        sequence_dim=1024,
        sequence_layers=36,
        heads=16,
        pair_dim=96,
        pair_layers=16,
        triangle_every=4,
        triangle_dim=48,
        gradient_checkpointing=True,
    )
    if arm == "C1":
        cfg.pair_dim = 256
        cfg.pair_layers = 24
        cfg.triangle_dim = 128
    if arm in ("C2", "D1", "T1", "A1", "J1"):
        cfg.pair_dim = 128
        cfg.triangle_dim = 64
    if arm in ("D1", "T1", "A1", "J1"):
        cfg.pair_directional = True
    cfg.pair_family = {"T1": "triangle", "A1": "attention", "J1": "joint"}.get(
        arm, "conv"
    )
    cfg.pair_attention_chunk = 64
    return asdict(cfg)


def training_config(arm, seed, steps, profile):
    b = profile["batch_size"]
    cfg = {
        "seed": seed,
        "model": model_config(arm),
        "batch_size": b,
        "accumulation": 8 // b,
        "logical_batch_size": 8,
        "crop": profile.get("crop", 512),
        "workers": 2,
        "steps": steps,
        "schedule": "wsd",
        "schedule_steps": 100000,
        "decay_start": 80000,
        "warmup": 1000,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "pos_weight": 4.0,
        "ema_decay": 0.999,
        "ranking_weight": 0.05,
        "afdb_probability": 0.5,
        "eval_every": 5000,
        "checkpoint_every": 1000,
        "log_every": 100,
        "keep_recovery_checkpoints": 2,
        "encoder_only": True,
        "save_validation_rows": True,
    }
    cfg["model"]["gradient_checkpointing"] = profile.get("gradient_checkpointing", True)
    cfg["compile_pair_blocks"] = profile.get("compile_pair_blocks", False)
    cfg["compile_pair_kernels"] = profile.get("compile_pair_kernels", False)
    cfg["compile_cache_limit"] = 128
    return cfg


def prepare(out):
    fs, path = storage(SOURCE)
    with fs.open(path, "rb") as f:
        content = f.read()
    checksum = hashlib.sha256(content).hexdigest()
    source = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    del content
    state = {
        "alphabet": ALPHABET,
        "model_config": source["model_config"],
        "source_checkpoint": SOURCE,
        "source_checkpoint_sha256": checksum,
        "ema": {
            k: v for k, v in source["ema"].items() if k.startswith(ENCODER_PREFIXES)
        },
    }
    buffer = io.BytesIO()
    torch.save(state, buffer)
    content = buffer.getvalue()
    fs, path = storage(out + "/encoder.pt")
    fs.pipe_file(path, content)
    report = {
        "checkpoint": out + "/encoder.pt",
        "sha256": hashlib.sha256(content).hexdigest(),
        "source": SOURCE,
        "source_sha256": checksum,
        "parameters": sum(v.numel() for v in state["ema"].values()),
    }
    write_json(report, out + "/encoder.json")
    print("ENCODER_READY " + json.dumps(report), flush=True)


def benchmark_training(arm, batch, checkpointing, compiled, crop=512):
    torch.cuda.empty_cache()
    cfg = ModelConfig(**model_config(arm))
    cfg.gradient_checkpointing = checkpointing
    model = ContactModel(cfg).cuda().train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
    if compiled:
        from torch._dynamo import config as dynamo_config

        dynamo_config.recompile_limit = 128
        for block in model.pairs:
            block.forward = torch.compile(block.forward, dynamic=True)
    kernels = arm in ("T1", "A1", "J1")
    if kernels:
        from sparsa.experiment import compile_pair_kernels

        compile_pair_kernels(model)
    tokens = torch.randint(1, 22, (batch, crop), device="cuda")
    targets = (torch.rand(batch, crop, crop, device="cuda") < 0.02).float()
    mask = torch.ones_like(targets, dtype=torch.bool).triu(6)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens)
            loss = contact_loss(logits, targets, mask, 4) + 0.05 * contact_ranking_loss(
                logits, targets, mask
            )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite profile gradients")
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(
                list(ema.parameters()), list(model.parameters()), 0.001
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    result = {
        "arm": arm,
        "batch_size": batch,
        "crop": crop,
        "gradient_checkpointing": checkpointing,
        "compile_pair_blocks": compiled,
        "compile_pair_kernels": kernels,
        "seconds_per_microbatch": sum(times[2:]) / 3,
        "estimated_seconds_per_step": sum(times[2:]) / 3 * 8 / batch,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss": float(loss.detach()),
        "grad_norm": float(norm),
    }
    del model, ema, opt, tokens, targets, mask, logits, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def profile(out, arms=ARMS, crop=512):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    fs, path = storage(out + "/profile.json")
    report = (
        json.loads(fs.cat_file(path))
        if fs.exists(path)
        else {"gpu": torch.cuda.get_device_name(), "arms": {}}
    )
    for arm in arms:
        if arm in report["arms"]:
            continue
        rows = []
        compiled = arm in ("C0", "C1", "C2", "D1")
        shapes = (
            ((4, True), (2, False), (4, False))
            if crop == 512
            else ((8, True), (4, False), (8, False))
        )
        for batch, checkpointing in shapes:
            try:
                rows.append(
                    benchmark_training(arm, batch, checkpointing, compiled, crop)
                )
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                rows.append(
                    {
                        "batch_size": batch,
                        "gradient_checkpointing": checkpointing,
                        "error": "CUDA OOM",
                    }
                )
        feasible = [r for r in rows if "error" not in r and r["peak_reserved_gib"] < 75]
        entry = {"training": rows}
        if feasible:
            entry["selected"] = min(
                feasible, key=lambda r: r["estimated_seconds_per_step"]
            )
            model = ContactModel(ModelConfig(**model_config(arm))).cuda().eval()
            entry["inference"] = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for length in (256, 512, 920, 1024):
                    tokens = torch.randint(1, 22, (1, length), device="cuda")
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    logits = model(tokens)
                    assert torch.isfinite(logits).all()
                    torch.testing.assert_close(logits, logits.transpose(1, 2))
                    torch.cuda.synchronize()
                    entry["inference"].append(
                        {
                            "length": length,
                            "seconds": time.perf_counter() - start,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated()
                            / 2**30,
                        }
                    )
                # Real GPU padding isolation; avoid requiring bitwise bf16 identity.
                tokens = torch.randint(1, 22, (2, 128), device="cuda")
                tokens[0, 81:] = 0
                mixed = model(tokens)[0, :81, :81]
                alone = model(tokens[:1, :81])[0]
                torch.testing.assert_close(mixed, alone, atol=0.08, rtol=0.015)
                entry["padding_max_error"] = float((mixed - alone).abs().max())
            del model, logits, tokens
            gc.collect()
            torch.cuda.empty_cache()
        report["arms"][arm] = entry
        write_json(report, out + "/profile.json")
        print("PAIR_PROFILE " + json.dumps({arm: entry}), flush=True)
    report["complete"] = True
    write_json(report, out + "/profile.json")


def collect(run, step, out):
    fs, path = storage(run)
    complete = json.loads(fs.cat_file(path + "/complete.json"))
    if complete["step"] != step:
        raise ValueError("Wrong training endpoint")
    val = json.loads(fs.cat_file(path + f"/validation/step-{step}.json"))
    frame = pd.read_csv(
        io.BytesIO(fs.cat_file(path + f"/validation/step-{step}-per_protein.csv"))
    )
    from sparsa.data import benchmark

    stems = {r["stem"] for r in benchmark()}
    if set(frame.eval_set) != {"eval-val"}:
        raise ValueError("Nonvalidation trial result")
    values = {}
    for region in ("all", "long"):
        rows = frame[(frame["range"] == region) & (frame["cut"] == "R")]
        if set(rows.stem) != stems or len(rows) != len(stems):
            raise ValueError("Validation coverage mismatch")
        values[region] = dict(zip(rows.stem, rows.precision, strict=True))
    provenance_path = path + "/provenance.json"
    resumes = fs.glob(path + "/resume-step-*-provenance.json")
    if resumes:
        provenance_path = max(
            resumes, key=lambda p: int(p.rsplit("/", 1)[1].split("-")[2])
        )
    result = {
        "run": run,
        "step": step,
        "observed_seconds_per_step": complete["elapsed_seconds"]
        / max(1, step - complete["initial_step"]),
        "validation": val,
        "per_protein": values,
        "exposure": json.loads(
            fs.cat_file(path + f"/validation/step-{step}-exposure.json")
        ),
        "provenance": json.loads(fs.cat_file(provenance_path)),
    }
    # Preserve hashes, not the full 3,500-shard inventory, in the compact result.
    result["source_files_sha256"] = hashlib.sha256(
        json.dumps(result["provenance"].pop("source_files"), sort_keys=True).encode()
    ).hexdigest()
    write_json(result, out)
    print("PAIR_RESULT " + json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "profile", "collect"])
    p.add_argument("--out", required=True)
    p.add_argument("--run")
    p.add_argument("--step", type=int)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--crop", type=int, choices=(256, 512), default=512)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a.out)
    elif a.mode == "profile":
        profile(a.out, a.arms, a.crop)
    else:
        collect(a.run, a.step, a.out)


if __name__ == "__main__":
    main()
