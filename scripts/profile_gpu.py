"""Measure training throughput/memory before selecting the production batch size."""

import gc
import json
import time

import torch

from sparsa.model import ContactModel, ModelConfig
from sparsa.train import contact_loss, write_json


def run(batch, length, checkpointing):
    torch.cuda.empty_cache()
    model = (
        ContactModel(
            ModelConfig(
                sequence_dim=512,
                sequence_layers=12,
                pair_dim=96,
                pair_layers=16,
                triangle_every=4,
                triangle_dim=48,
                gradient_checkpointing=checkpointing,
            )
        )
        .cuda()
        .train()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, fused=True)
    tokens = torch.randint(1, 22, (batch, length), device="cuda")
    labels = (torch.rand(batch, length, length, device="cuda") < 0.01).float()
    mask = torch.ones(batch, length, length, device="cuda", dtype=torch.bool).triu(6)
    torch.cuda.reset_peak_memory_stats()
    start = None
    for step in range(13):
        if step == 3:
            torch.cuda.synchronize()
            start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = contact_loss(model(tokens), labels, mask, 4)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / 10
    result = {
        "batch_size": batch,
        "crop": length,
        "checkpointing": checkpointing,
        "seconds_per_step": seconds,
        "proteins_per_second": batch / seconds,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    del optimizer, model, tokens, labels, mask, loss
    gc.collect()
    return result


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    results = []
    for batch, crop, checkpointing in [
        (4, 384, True),
        (8, 384, True),
        (16, 384, True),
        (8, 384, False),
        (8, 512, True),
    ]:
        try:
            result = run(batch, crop, checkpointing)
        except torch.cuda.OutOfMemoryError:
            result = {
                "batch_size": batch,
                "crop": crop,
                "checkpointing": checkpointing,
                "error": "CUDA OOM",
            }
            gc.collect()
            torch.cuda.empty_cache()
        print(json.dumps(result), flush=True)
        results.append(result)
        write_json(
            results,
            "s3://marin-us-east-02a/marin/protein-structure/sparsa/data/gpu_profile.json",
        )
