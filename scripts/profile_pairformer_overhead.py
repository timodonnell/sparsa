"""Compare original and optimized checkpoint/chunk bookkeeping on one GPU.

Uses the unchanged full architecture, microbatch, precision, AdamW and EMA.
Synthetic timings exclude DDP and data I/O; they are not live training timings.
The original attention forward is retained as a numerical/benchmark reference.
"""

import argparse
import gc
import json
from pathlib import Path
from unittest.mock import patch

import fsspec
import torch
import yaml
from torch.utils.checkpoint import checkpoint

from scripts.profile_diffusion import profile
from sparsa.diffusion import DiffusionModelConfig
from sparsa.pair_trunk import TriangleAttention
from sparsa.pairformer import PairformerBlock


def original_attention_forward(self, z, mask):
    """Original implementation from 4162131, including nested checkpoints."""
    if self.ending:
        z, mask = z.transpose(1, 2), mask.transpose(1, 2)
    x = self.norm(z) * mask[..., None]
    bias = self.bias(x).permute(0, 3, 1, 2)
    chunks = []
    for start in range(0, z.shape[1], self.chunk):
        args = (
            x[:, start : start + self.chunk],
            bias,
            mask[:, start : start + self.chunk],
        )
        if self.training and torch.is_grad_enabled():
            y = checkpoint(self._chunk, *args, use_reentrant=False)
        else:
            y = self._chunk(*args)
        chunks.append(y)
    y = torch.cat(chunks, 1)
    z = (z + self.out(y * self.gate(x).sigmoid())) * mask[..., None]
    return z.transpose(1, 2) if self.ending else z


def verify_block(cfg, batch=2, length=384):
    """Require bitwise equality with deterministic kernels and nonzero weights."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(234)
        config = DiffusionModelConfig(**cfg["model"])
        model = PairformerBlock(config, checkpoint_chunks=False).cuda().train()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.Linear) and not module.weight.any():
                    torch.nn.init.normal_(module.weight, std=0.01)
        z = torch.randn(
            batch, length, length, config.pair_dim, device="cuda", requires_grad=True
        )
        s = torch.randn(
            batch, length, config.sequence_dim, device="cuda", requires_grad=True
        )
        lengths = length - torch.arange(batch, device="cuda") * 13
        valid = torch.arange(length, device="cuda")[None] < lengths[:, None]
        mask = valid[:, :, None] & valid[:, None, :]
        weights = (torch.randn_like(z), torch.randn_like(s))
        reference = None
        for forward in (original_attention_forward, TriangleAttention.forward):
            model.zero_grad(set_to_none=True)
            z.grad = s.grad = None
            with patch.object(TriangleAttention, "forward", forward):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = checkpoint(model, z, s, mask, use_reentrant=False)
                    loss = sum(
                        (x * w).mean() for x, w in zip(outputs, weights, strict=True)
                    )
                loss.backward()
            values = {f"output.{i}": x.detach().cpu() for i, x in enumerate(outputs)}
            values.update(
                {f"gradient.{n}": p.grad.cpu() for n, p in model.named_parameters()}
            )
            values.update(input_z_gradient=z.grad.cpu(), input_s_gradient=s.grad.cpu())
            if reference is None:
                reference = values
            else:
                for name, value in values.items():
                    torch.testing.assert_close(
                        value,
                        reference[name],
                        rtol=0,
                        atol=0,
                        msg=lambda message, name=name: f"{name}: {message}",
                    )
        return {
            "width": config.pair_dim,
            "batch": batch,
            "length": length,
            "deterministic": True,
            "bitwise_equal_outputs_and_gradients": True,
            "compared_tensors": len(reference),
        }
    finally:
        torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--widths", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "scope": (
            "Deterministic BF16 block outputs and gradients with nonzero residual weights"
            if args.verify_only
            else "Synthetic full-model microsteps; no DDP or data I/O"
        ),
        **(
            {"steps": args.steps, "warmup": args.warmup} if not args.verify_only else {}
        ),
        "results": [],
    }
    current = TriangleAttention.forward
    for width in args.widths:
        cfg = yaml.safe_load(Path(f"configs/pairformer_noisy_{width}.yaml").read_text())
        if args.verify_only:
            result = verify_block(cfg)
            report["results"].append(result)
            with fsspec.open(args.out, "w") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
            print(json.dumps(result), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        for label, forward in [
            ("original", original_attention_forward),
            ("optimized", current),
        ]:
            torch.manual_seed(23)
            with patch.object(TriangleAttention, "forward", forward):
                result = profile(
                    cfg,
                    cfg["batch_size"],
                    cfg["crop"],
                    steps=args.steps,
                    warmup=args.warmup,
                )
            result.update(width=width, variant=label)
            report["results"].append(result)
            with fsspec.open(args.out, "w") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
            print(json.dumps(result), flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
