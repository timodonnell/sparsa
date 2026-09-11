"""Profile larger sequence-only models on a batch Iris H100."""

import argparse
import copy
import gc
import json
import time
from dataclasses import asdict

import torch

from sparsa.model import ContactModel, ModelConfig
from sparsa.train import contact_loss, write_json


def profile(config, batch, length, transform=None):
    torch.cuda.empty_cache()
    model = ContactModel(config).cuda().train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if transform is not None:
        transform(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, fused=True)
    tokens = torch.randint(1, 22, (batch, length), device="cuda")
    target = (torch.rand(batch, length, length, device="cuda") < 0.01).float()
    mask = torch.ones_like(target, dtype=torch.bool).triu(6)
    torch.cuda.reset_peak_memory_stats()
    for step in range(8):
        if step == 3:
            torch.cuda.synchronize()
            start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = contact_loss(model(tokens), target, mask, 4.0)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite gradients during profiling")
        optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(
                list(ema.parameters()), list(model.parameters()), 0.001
            )
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / 5
    return {
        "model_config": asdict(config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": batch,
        "crop": length,
        "seconds_per_microbatch": seconds,
        "crops_per_second_per_gpu": batch / seconds,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "loss": float(loss.detach()),
        "grad_norm": float(norm),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--grown-fast", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    results = []
    architectures = (
        [(1024, 36, 16)] if args.grown_fast else [(768, 24, 12), (1024, 32, 16)]
    )
    shapes = (
        [(4, 512), (8, 512), (2, 768), (2, 1024)]
        if args.grown_fast
        else [(4, 384), (8, 384), (4, 512), (2, 1024)]
    )
    for dim, layers, heads in architectures:
        for batch, length in shapes:
            cfg = ModelConfig(
                sequence_dim=dim,
                sequence_layers=layers,
                heads=heads,
                pair_dim=96 if args.grown_fast else 128,
                pair_layers=16 if args.grown_fast else 24,
                triangle_every=4,
                triangle_dim=48 if args.grown_fast else 64,
                gradient_checkpointing=not args.grown_fast,
            )
            try:
                result = profile(cfg, batch, length)
            except torch.cuda.OutOfMemoryError:
                result = {
                    "model_config": asdict(cfg),
                    "batch_size": batch,
                    "crop": length,
                    "error": "CUDA OOM",
                }
            gc.collect()
            torch.cuda.empty_cache()
            results.append(result)
            print(json.dumps(result), flush=True)
            write_json(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "results": results,
                    "scope": "Synthetic full-length training, including EMA and AdamW; excludes DDP and teacher I/O",
                },
                args.out,
            )


if __name__ == "__main__":
    main()
