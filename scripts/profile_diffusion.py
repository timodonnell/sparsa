"""Measure large diffusion training memory/throughput on one dedicated GPU."""

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from sparsa.diffusion import (
    BinaryDiffusion,
    DiffusionModelConfig,
    build_diffusion_model,
)
from sparsa.train import contact_loss, contact_ranking_loss


def profile(cfg, batch, crop):
    torch.cuda.empty_cache()
    model = build_diffusion_model(DiffusionModelConfig(**cfg["model"])).cuda().train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), fused=True
    )
    schedule = BinaryDiffusion(
        cfg["model"]["diffusion_steps"], cfg["contact_priors"]
    ).cuda()
    rng = torch.Generator(device="cuda").manual_seed(20261002)
    tokens = torch.randint(1, 22, (batch, crop), device="cuda", generator=rng)
    target = torch.triu(
        torch.rand(batch, crop, crop, device="cuda", generator=rng) < 0.01, 6
    ).float()
    target = target + target.transpose(1, 2)
    mask = torch.ones_like(target, dtype=torch.bool).triu(6)
    torch.cuda.reset_peak_memory_stats()
    for step in range(6):
        if step == 2:
            torch.cuda.synchronize()
            start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        t = torch.randint(1, schedule.steps + 1, (batch,), device="cuda", generator=rng)
        noisy = schedule.sample_forward(target, tokens, t, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens, noisy, t)
            loss = contact_loss(logits, target, mask, cfg["pos_weight"]) + cfg[
                "ranking_weight"
            ] * contact_ranking_loss(logits, target, mask, batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite profile gradients")
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(
                list(ema.parameters()), list(model.parameters()), 0.001
            )
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / 4
    return {
        "batch_size": batch,
        "crop": crop,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds_per_microbatch": seconds,
        "estimated_seconds_per_optimizer_step": seconds
        * cfg["logical_batch_size"]
        / batch,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "loss": float(loss.detach()),
        "grad_norm": float(norm),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--crops", type=int, nargs="+", default=[384, 512])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(23)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "scope": "Synthetic single-GPU optimizer steps including AdamW and EMA; estimated logical steps exclude DDP and data I/O.",
        "config": cfg,
        "results": [],
    }
    for batch, crop in [(b, c) for c in args.crops for b in args.batches]:
        try:
            result = profile(cfg, batch, crop)
        except torch.cuda.OutOfMemoryError:
            result = {"batch_size": batch, "crop": crop, "error": "CUDA OOM"}
        report["results"].append(result)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
