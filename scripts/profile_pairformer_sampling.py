"""Check complete reverse sampling on the longest frozen validation protein."""

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from sparsa.data import benchmark, tokenize
from sparsa.diffusion import (
    BinaryDiffusion,
    DiffusionModelConfig,
    build_diffusion_model,
    sample_contact_maps,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    rec = max(benchmark(), key=lambda r: r["L"])
    tokens = tokenize(rec["sequence"])[None].cuda()
    schedule = BinaryDiffusion(
        cfg["model"]["diffusion_steps"], cfg["contact_priors"]
    ).cuda()
    rows = []
    for mode in ("cached", "noisy"):
        model = (
            build_diffusion_model(
                replace(DiffusionModelConfig(**cfg["model"]), pairformer_mode=mode)
            )
            .cuda()
            .eval()
        )
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model.encode(tokens)
            state, probability = sample_contact_maps(
                model,
                schedule,
                tokens,
                4,
                torch.Generator(device="cuda").manual_seed(23),
                encoded=encoded,
            )
        torch.cuda.synchronize()
        assert torch.isfinite(probability).all() and state.shape == (
            4,
            rec["L"],
            rec["L"],
        )
        torch.testing.assert_close(state, state.transpose(1, 2))
        rows.append(
            {
                "mode": mode,
                "length": rec["L"],
                "rollouts": 4,
                "wall_seconds": time.monotonic() - start,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "finite_probabilities": True,
            }
        )
        args.out.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "pair_width": cfg["model"]["pair_dim"],
                    "scope": "Operational sampling profile only; random initial weights, no accuracy claims.",
                    "results": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(rows[-1]), flush=True)
        del model, encoded, state, probability
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
