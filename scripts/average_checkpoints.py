"""Average compatible EMA weights into a single inference checkpoint."""

import copy

import torch

from sparsa.train import load_checkpoint


@torch.no_grad()
def average_ema(checkpoints):
    checkpoints = list(checkpoints)
    if len(checkpoints) < 2 or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Provide at least two distinct checkpoints")
    result = None
    sources = []
    for count, uri in enumerate(checkpoints, 1):
        state = load_checkpoint(uri)
        if result is None:
            result = {
                key: copy.deepcopy(state[key])
                for key in [
                    "format_version",
                    "alphabet",
                    "model_config",
                    "training_config",
                    "step",
                ]
            }
            result["ema"] = {k: v.clone() for k, v in state["ema"].items()}
        else:
            for key in ["format_version", "alphabet", "model_config"]:
                if result[key] != state[key]:
                    raise ValueError(f"Cannot average different {key}")
            for key in ["crop", "pos_weight"]:
                if result["training_config"].get(key) != state["training_config"].get(
                    key
                ):
                    raise ValueError(f"Cannot average different training {key}")
            if result["ema"].keys() != state["ema"].keys():
                raise ValueError("Checkpoint parameter names differ")
            for name, mean in result["ema"].items():
                value = state["ema"][name]
                if value.shape != mean.shape or value.dtype != mean.dtype:
                    raise ValueError(f"Incompatible parameter {name}")
                if mean.is_floating_point():
                    mean.lerp_(value, 1 / count)
                elif not torch.equal(mean, value):
                    raise ValueError(f"Non-floating buffer {name} differs")
            result["step"] = max(result["step"], state["step"])
        sources.append(
            {"checkpoint": uri, "step": state["step"], "weight": 1 / len(checkpoints)}
        )
    result["checkpoint_kind"] = "ema_average"
    result["averaged_checkpoints"] = sources
    return result
