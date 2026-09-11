"""Check compiled pair-block numerics and measure whether compilation pays off."""

import argparse
import copy
import json
import time

import torch

from scripts.profile_scaling import profile
from sparsa.model import ModelConfig, PairBlock, SequenceBlock
from sparsa.train import write_json


def compile_pairs(model):
    # Compile bound forwards so serialization keeps the ordinary parameter keys.
    for block in model.pairs:
        block.forward = torch.compile(block.forward, dynamic=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--very-large", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--sequence", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(17)
    cfg = ModelConfig(
        sequence_dim=2048 if args.very_large else 1024,
        sequence_layers=36,
        heads=32 if args.very_large else 16,
        pair_dim=96,
        pair_layers=16,
        triangle_every=4,
        triangle_dim=48,
        gradient_checkpointing=args.long,
    )
    original = PairBlock(cfg, 3).cuda().train()
    with torch.no_grad():
        original.triangle.out.weight.normal_(std=0.02)
    compiled = copy.deepcopy(original)
    compiled.forward = torch.compile(compiled.forward, dynamic=True)
    results = {
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "checks": [],
    }
    started = time.monotonic()
    for length in (37, 53):
        z = torch.randn(2, length, length, 96, device="cuda")
        valid = torch.ones(2, length, device="cuda", dtype=torch.bool)
        valid[0, -7:] = False
        mask = valid[:, :, None] & valid[:, None, :]
        z *= mask[..., None]
        original.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b = original(z, mask), compiled(z, mask)
        a.square().mean().backward()
        b.square().mean().backward()
        torch.testing.assert_close(a, b, atol=0.03, rtol=0.01)
        assert original.state_dict().keys() == compiled.state_dict().keys()
        max_grad_error = 0.0
        for p, q in zip(original.parameters(), compiled.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=0.001, rtol=0.05)
            max_grad_error = max(max_grad_error, float((p.grad - q.grad).abs().max()))
        results["checks"].append(
            {
                "length": length,
                "max_output_error": float((a - b).detach().abs().max()),
                "max_gradient_error": max_grad_error,
            }
        )
        write_json(results, args.out)
        print("COMPILE_CHECK " + json.dumps(results["checks"][-1]), flush=True)
    del original, compiled, a, b, z
    torch.cuda.empty_cache()
    if args.sequence:
        original = SequenceBlock(cfg).cuda().train()
        compiled = copy.deepcopy(original)
        compiled.forward = torch.compile(compiled.forward, dynamic=True)
        for length in (37, 53):
            x = torch.randn(2, length, cfg.sequence_dim, device="cuda")
            valid = torch.ones(2, length, device="cuda", dtype=torch.bool)
            valid[0, -7:] = False
            original.zero_grad(set_to_none=True)
            compiled.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, b = original(x, valid), compiled(x, valid)
            a.square().mean().backward()
            b.square().mean().backward()
            torch.testing.assert_close(a, b, atol=0.03, rtol=0.01)
            max_error = 0.0
            for p, q in zip(original.parameters(), compiled.parameters(), strict=True):
                torch.testing.assert_close(p.grad, q.grad, atol=0.001, rtol=0.05)
                max_error = max(max_error, float((p.grad - q.grad).abs().max()))
            results["checks"].append(
                {
                    "block": "sequence",
                    "length": length,
                    "max_output_error": float((a - b).detach().abs().max()),
                    "max_gradient_error": max_error,
                }
            )
        del original, compiled, a, b, x
        torch.cuda.empty_cache()

    def transform(model):
        compile_pairs(model)
        if args.sequence:
            for block in model.sequence:
                block.forward = torch.compile(block.forward, dynamic=True)

    batch = 1 if args.long else 2 if args.very_large else 4
    crop = 1024 if args.long else 512
    try:
        result = profile(cfg, batch, crop, transform=transform)
    except torch.cuda.OutOfMemoryError:
        result = {
            "error": "CUDA OOM",
            "sequence_dim": cfg.sequence_dim,
            "batch_size": batch,
            "crop": crop,
        }
    results["profile"] = result
    results["total_probe_seconds"] = time.monotonic() - started
    write_json(results, args.out)
    print("COMPILE_PROFILE " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
