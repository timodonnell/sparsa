"""Widen and deepen a trained contact encoder while preserving its function.

Only Sparsa's supervised contact weights are used. Repeated attention heads keep
their original head dimension and RoPE frequencies. New residual blocks start
as identities. The pair trunk is copied unchanged.
"""

from dataclasses import asdict

import torch

from sparsa.model import ALPHABET, ContactModel, ModelConfig


@torch.no_grad()
def grow_from_ema(model: ContactModel, state: dict, *, noise=0.01):
    old = ModelConfig(**state["model_config"])
    new = model.config
    if state.get("alphabet") != ALPHABET:
        raise ValueError("Growth requires the Sparsa amino-acid vocabulary")
    if state.get("inference_config", {}).get("relative_max_distance") is not None:
        raise ValueError("Growth requires an uncapped source checkpoint")
    factor = new.sequence_dim // old.sequence_dim
    depth = new.sequence_layers // old.sequence_layers
    if factor not in (1, 2) or new.sequence_dim != factor * old.sequence_dim:
        raise ValueError("Growth supports unchanged or doubled sequence width")
    if depth < 1 or new.sequence_layers != depth * old.sequence_layers:
        raise ValueError("Sequence depth must be an integer multiple of source depth")
    if new.heads != factor * old.heads:
        raise ValueError("Growth must preserve attention head dimension")
    allowed = {"sequence_dim", "sequence_layers", "heads", "gradient_checkpointing"}
    if any(asdict(old)[k] != v for k, v in asdict(new).items() if k not in allowed):
        raise ValueError("Growth must preserve the pair architecture and dropout")
    if noise < 0:
        raise ValueError("Growth noise must be nonnegative")
    source = state["ema"]
    # Validate the source architecture and keys before applying any transforms.
    with torch.device("meta"):
        expected = ContactModel(old).state_dict()
    if set(source) != set(expected) or any(
        source[k].shape != v.shape for k, v in expected.items()
    ):
        raise ValueError("Source EMA does not match its model configuration")
    target = model.state_dict()
    d = old.sequence_dim
    for key, value in source.items():
        name = key
        if key.startswith("sequence."):
            pieces = key.split(".")
            pieces[1] = str(int(pieces[1]) * depth)
            name = ".".join(pieces)
            if key.endswith("qkv.weight"):
                value = (
                    value.reshape(3, d, d)
                    .repeat(1, factor, factor)
                    .reshape(3 * d * factor, d * factor)
                    / factor
                )
            elif value.ndim == 2:
                value = value.repeat(factor, factor) / factor
            else:
                value = value.repeat(factor)
        elif key == "embedding.weight":
            value = value.repeat(1, factor)
        elif key.startswith("seqnorm."):
            value = value.repeat(factor)
        elif key in ("left.weight", "product.weight"):
            value = value.repeat(1, factor) / factor
            if factor == 2 and noise:
                # Antisymmetric perturbations cancel on duplicated inputs but
                # give the two channel copies different gradients during training.
                delta = torch.randn_like(value[:, :d]) * source[key].std() * noise
                value[:, :d] += delta
                value[:, d:] -= delta
        target[name].copy_(value)
    for index, block in enumerate(model.sequence):
        if index % depth:
            block.out.weight.zero_()
            block.out.bias.zero_()
            block.ff[3].weight.zero_()
            block.ff[3].bias.zero_()
    return {
        "source_config": asdict(old),
        "target_config": asdict(new),
        "width_factor": factor,
        "depth_factor": depth,
        "symmetry_breaking_noise": noise,
        "weights": "source EMA",
    }
