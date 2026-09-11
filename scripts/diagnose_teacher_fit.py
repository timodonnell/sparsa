"""Training-corpus fit diagnostic; this is not an independent held-out score."""

import argparse
import json
import random
from pathlib import Path

import torch

from sparsa.data import decode_document, inventory, load_shard
from sparsa.evaluate import evaluate
from sparsa.model import ContactModel, ModelConfig
from sparsa.train import load_checkpoint, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    state = load_checkpoint(args.checkpoint)
    model = ContactModel(ModelConfig(**state["model_config"])).cuda().eval()
    model.load_state_dict(state["ema"])
    rng = random.Random(251)
    shards = inventory()
    results = {
        "checkpoint": args.checkpoint,
        "seed": 251,
        "scope": "100 proteins per source, lengths 12–1024; sampled from the training corpus, potentially seen during training. Not a held-out generalization estimate.",
        "sources": {},
    }
    for source in ("afdb", "esm"):
        selected = rng.sample(shards[source], 5)
        records = []
        for path in selected:
            rows = [
                r
                for r in load_shard(path)
                if not r.get("truncated", False) and 12 <= int(r["seq_len"]) <= 1024
            ]
            for row in rng.sample(rows, 20):
                rec = decode_document(row)
                length = len(rec["sequence"])
                records.append(
                    {
                        "dataset": source,
                        "stem": rec["entry_id"],
                        "sequence": rec["sequence"],
                        "L": length,
                        "resolved": list(range(length)),
                        "contacts": [[int(i), int(j), 1.0] for i, j in rec["contacts"]],
                        "eval_set": "teacher-fit",
                    }
                )
        out = Path("/tmp/teacher-fit") / source
        summary = evaluate(
            model,
            records,
            torch.device("cuda"),
            out=out,
            pos_weight=state["training_config"]["pos_weight"],
        )
        results["sources"][source] = {
            "summary": summary,
            "shards": selected,
            "length_mean": sum(r["L"] for r in records) / len(records),
        }
        # Keep identities and scores for reproducibility, without uploading matrices.
        from sparsa.train import storage

        fs, prefix = storage(args.out)
        fs.pipe_file(
            prefix + f"/{source}_per_protein.csv",
            (out / "per_protein.csv").read_bytes(),
        )
        write_json(results, args.out + "/summary.json")
        print("TEACHER_FIT " + json.dumps(results["sources"][source]), flush=True)


if __name__ == "__main__":
    main()
