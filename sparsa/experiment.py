"""Explicit initialization and fixed-horizon training controls for pair trials."""

import hashlib
import math

from sparsa.model import ALPHABET

ENCODER_PREFIXES = ("embedding.", "sequence.", "seqnorm.")


def compile_pair_kernels(model):
    """Fuse dense triangle/transition operations without compiling chunk loops."""
    import torch
    from torch._dynamo import config as dynamo_config

    dynamo_config.recompile_limit = 128
    for block in model.pairs:
        for name in ("outgoing", "incoming", "ff"):
            module = getattr(block, name, None)
            if module is not None:
                module.forward = torch.compile(module.forward, dynamic=True)


def initialize_encoder(model, state):
    if state.get("alphabet") != ALPHABET:
        raise ValueError("Encoder initialization vocabulary mismatch")
    for key in ("sequence_dim", "sequence_layers", "heads"):
        if state["model_config"][key] != getattr(model.config, key):
            raise ValueError("Encoder initialization architecture mismatch: " + key)
    source = {k: v for k, v in state["ema"].items() if k.startswith(ENCODER_PREFIXES)}
    target = {
        k: v for k, v in model.state_dict().items() if k.startswith(ENCODER_PREFIXES)
    }
    if source.keys() != target.keys() or any(
        source[k].shape != v.shape for k, v in target.items()
    ):
        raise ValueError("Encoder state keys/shapes mismatch")
    model.load_state_dict(source, strict=False)
    return {
        "kind": "encoder_only_ema",
        "copied_parameters": sum(v.numel() for v in source.values()),
        "source_checkpoint_sha256": state.get("source_checkpoint_sha256"),
    }


def learning_rate_scale(cfg, step):
    warmup = cfg.get("warmup", 500)
    horizon = cfg.get("schedule_steps", cfg["steps"])
    progress = max(0, (step - warmup) / max(1, horizon - warmup))
    if cfg.get("schedule", "cosine") == "cosine":
        decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
    elif cfg["schedule"] == "wsd":
        begin = cfg["decay_start"]
        if not 0 <= warmup < begin < horizon or cfg["steps"] > horizon:
            raise ValueError("Invalid fixed WSD horizon")
        decay = 1 - 0.9 * min(1, max(0, (step - begin) / (horizon - begin)))
    else:
        raise ValueError("Unknown LR schedule")
    return min(1, (step + 1) / max(1, warmup)) * decay


def microbatches(batch, size):
    n = len(batch["ids"])
    if n % size:
        raise ValueError("Logical batch must be divisible by microbatch")
    for start in range(0, n, size):
        yield {k: batch[k][start : start + size] for k in ("tokens", "targets", "mask")}


def advance_data_digest(previous, batch):
    import json

    payload = json.dumps(
        list(zip(batch["ids"], batch["crop_starts"], strict=True)),
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(bytes.fromhex(previous) + payload).hexdigest()
