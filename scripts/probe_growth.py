"""Verify growth on actual trained weights and profile the expanded model."""

import argparse
import json
from dataclasses import replace

import torch

from scripts.profile_scaling import profile
from sparsa.data import benchmark
from sparsa.evaluate import evaluate
from sparsa.grow import grow_from_ema
from sparsa.model import ContactModel, ModelConfig, tokenize
from sparsa.train import load_checkpoint, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(217)
    state = load_checkpoint(args.checkpoint)
    old_config = ModelConfig(**state["model_config"])
    new_config = replace(old_config, sequence_dim=1024, heads=16, sequence_layers=36)
    old = ContactModel(old_config).cuda().eval()
    old.load_state_dict(state["ema"])
    new = ContactModel(new_config).cuda().eval()
    growth = grow_from_ema(new, state)
    val = benchmark()
    rec = next(r for r in val if r["stem"] == "7pv5_A")
    tokens = tokenize(rec["sequence"])[None].cuda()
    with torch.inference_mode():
        a, b = old(tokens), new(tokens)
        maximum_error = float((a - b).abs().max())
        torch.testing.assert_close(a, b, atol=0.001, rtol=0.0001)
    before = evaluate(old, val, torch.device("cuda"), pos_weight=4)
    after = evaluate(new, val, torch.device("cuda"), pos_weight=4)
    report = {
        "source": args.checkpoint,
        "growth": growth,
        "fp32_max_abs_logit_error": maximum_error,
        "source_validation": before,
        "grown_validation": after,
    }
    print("GROWTH_CHECK " + json.dumps(report), flush=True)
    write_json(report, args.out)
    del old, new, state, a, b
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")
    report["profiles"] = []
    for batch, crop, checkpointing in [(8, 384, True), (4, 512, True), (4, 384, False)]:
        result = profile(
            replace(new_config, gradient_checkpointing=checkpointing), batch, crop
        )
        report["profiles"].append(result)
        print("GROWTH_PROFILE " + json.dumps(result), flush=True)
        write_json(report, args.out)


if __name__ == "__main__":
    main()
