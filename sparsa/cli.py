"""Predict a full contact probability matrix and positive-only Helico contacts."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from sparsa.data import benchmark
from sparsa.evaluate import evaluate
from sparsa.model import ALPHABET, ContactModel, ModelConfig, tokenize
from sparsa.train import load_checkpoint


def load_model(checkpoint, device):
    state = load_checkpoint(checkpoint)
    if state.get("format_version") == 2 and state["alphabet"] != ALPHABET:
        raise ValueError(
            "Checkpoint amino-acid vocabulary differs from this implementation"
        )
    model = ContactModel(ModelConfig(**state["model_config"])).to(device).eval()
    model.load_state_dict(state["ema"])
    model.relative_max_distance = state.get("inference_config", {}).get(
        "relative_max_distance"
    )
    return model, state


def export_contacts(scores, path, top_k):
    """Write 0-based positive pairs; unlisted pairs remain unknown in Helico."""
    if (
        scores.ndim != 2
        or scores.shape[0] != scores.shape[1]
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Expected a finite square contact matrix")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    i, j = np.triu_indices(len(scores), k=6)
    order = np.argsort(-scores[i, j], kind="stable")[:top_k]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(f"{i[k]} {j[k]}\n" for k in order))


def main():
    parser = argparse.ArgumentParser(prog="sparsa")
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--sequence", required=True)
    predict.add_argument(
        "--top-l",
        type=float,
        default=1.0,
        help="Helico positive contact budget as a multiple of sequence length",
    )
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--split", action="append", choices=["eval-val", "eval-test", "eval-denovo"]
    )
    for p in (predict, evaluate_parser):
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--out", required=True)
        p.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
    args = parser.parse_args()
    device = torch.device(args.device)
    started = time.perf_counter()
    model, state = load_model(args.checkpoint, device)
    model_load_seconds = time.perf_counter() - started
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate":
        result = evaluate(
            model,
            benchmark(splits=tuple(args.split or ["eval-val"])),
            device,
            out,
            label=f"sparsa-step-{state['step']}",
            pos_weight=state["training_config"].get("pos_weight", 1.0),
        )
    else:
        if not math.isfinite(args.top_l) or args.top_l < 0:
            raise ValueError("top-l must be finite and nonnegative")
        tokens = tokenize(args.sequence)[None].to(device)
        seq = "".join(args.sequence.split()).upper()
        started = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            logits = model(tokens)[0].float()
        weight = state["training_config"].get("pos_weight", 1.0)
        scores = (logits - math.log(weight)).sigmoid().cpu().numpy()
        elapsed = time.perf_counter() - started
        # The excluded near-diagonal band is invalid, not a confident noncontact.
        valid = np.abs(np.arange(len(seq))[:, None] - np.arange(len(seq))[None]) >= 6
        np.savez_compressed(
            out / "contacts.npz", score=scores, valid=valid, sequence=seq
        )
        export_contacts(scores, out / "helico_contacts.txt", int(len(seq) * args.top_l))
        result = {
            "sequence": seq,
            "length": len(seq),
            "elapsed_seconds": elapsed,
            "contact_budget_top_l": args.top_l,
        }
    result.update(
        checkpoint=args.checkpoint,
        step=state["step"],
        model_load_seconds=model_load_seconds,
        parameters=sum(p.numel() for p in model.parameters()),
        alphabet=ALPHABET,
        contact_definition="native-amino-acid ConFind degree >= 0.001, sequence separation >= 6",
        index_base=0,
        model_config=state["model_config"],
        inference_config=state.get("inference_config", {}),
    )
    (out / "prediction.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
