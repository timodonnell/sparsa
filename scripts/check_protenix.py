"""GPU transfer preflight: native weights, gradients, padding, longest rollout."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml

from sparsa.data import benchmark
from sparsa.diffusion import (
    BinaryDiffusion,
    DiffusionModelConfig,
    build_diffusion_model,
    sample_contact_maps,
)
from sparsa.model import TOKEN_IDS
from sparsa.train_diffusion import load_checkpoint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    device = torch.device("cuda")
    torch.manual_seed(cfg["seed"])
    torch.set_float32_matmul_precision("high")
    model = build_diffusion_model(DiffusionModelConfig(**cfg["model"])).to(device)
    metadata = model.initialize_backbone(load_checkpoint(cfg["pretrained_backbone"]))
    model.configure_trainability(False)
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 2
    tokens = torch.tensor([[TOKEN_IDS[x] for x in sequence]], device=device)
    schedule = BinaryDiffusion(
        cfg["model"]["diffusion_steps"], cfg["contact_priors"]
    ).to(device)
    rng = torch.Generator(device=device).manual_seed(42)
    noisy = schedule.sample_prior(tokens, rng)
    timestep = torch.tensor([4], device=device)
    evidence = {
        "initialization": metadata,
        "gpu": torch.cuda.get_device_name(),
        "held_out_used": False,
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(tokens, noisy, timestep)
    output.float().square().mean().backward()
    assert model.noisy_state.weight.grad.norm() > 0
    assert model.adapter_gate.grad.abs() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    evidence["adapter_gradient_norm"] = model.noisy_state.weight.grad.norm().item()
    evidence["gate_gradient"] = model.adapter_gate.grad.item()
    model.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        plain = model(tokens, noisy, timestep)
        padded = model(
            torch.nn.functional.pad(tokens, (0, 7)),
            torch.nn.functional.pad(noisy, (0, 7, 0, 7)),
            timestep,
        )
        assert torch.equal(plain, padded[:, : tokens.shape[1], : tokens.shape[1]])
        assert torch.equal(plain, plain.transpose(1, 2))
    evidence["padding_exact"] = True
    evidence["symmetric"] = True
    model.configure_trainability(True)
    # At maximum training length, check every trainable parameter participates.
    tokens = torch.randint(1, 21, (cfg["batch_size"], cfg["crop"]), device=device)
    noisy = schedule.sample_prior(tokens, rng)
    timestep = torch.full((cfg["batch_size"],), 4, device=device)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(tokens, noisy, timestep)
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    evidence["full_training_microbatch_seconds"] = time.monotonic() - started
    evidence["full_training_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    evidence["all_parameters_receive_gradients"] = True
    model.zero_grad(set_to_none=True)
    del output, noisy, tokens
    torch.cuda.empty_cache()
    # Exercise the true longest fixed-validation input and all eight reverse steps.
    record = max(benchmark(), key=lambda r: r["L"])
    tokens = torch.tensor(
        [[TOKEN_IDS.get(x, TOKEN_IDS["X"]) for x in record["sequence"]]], device=device
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        maps, scores = sample_contact_maps(
            model,
            schedule,
            tokens,
            cfg["rollout_batch"],
            rng,
            logit_correction=math.log(cfg["pos_weight"]),
        )
    assert torch.isfinite(scores).all()
    assert torch.equal(maps, maps.transpose(1, 2))
    torch.cuda.synchronize()
    evidence.update(
        longest_validation_length=record["L"],
        rollout_batch=cfg["rollout_batch"],
        longest_rollout_seconds=time.monotonic() - started,
        rollout_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
        parameters=sum(p.numel() for p in model.parameters()),
    )
    write_json(args.out, evidence)
    print("PROTENIX_PREFLIGHT " + json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
