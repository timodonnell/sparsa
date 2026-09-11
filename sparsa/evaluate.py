"""Experimental scoring with the unmodified MarinFold exp89 metric functions."""

import json
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sparsa.model import tokenize
from sparsa.vendor.marinfold_metrics import (
    metric_rows,
    resolved_pairs,
    stamp,
    true_matrix,
)


@torch.inference_mode()
def evaluate(model, records, device, out=None, label="sparsa", pos_weight=1.0):
    model.eval()
    rows, timings = [], []
    if out is not None:
        out = Path(out)
        (out / "scores").mkdir(parents=True, exist_ok=True)
    for rec in records:
        tokens = tokenize(rec["sequence"])[None].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(tokens)[0].float()
        scores = (logits - np.log(pos_weight)).sigmoid().cpu().numpy()
        elapsed = time.perf_counter() - start
        if not np.isfinite(scores).all() or scores.shape != (rec["L"], rec["L"]):
            raise ValueError(f"Invalid prediction for {rec['stem']}")
        pi, pj, sep = resolved_pairs(np.asarray(rec["resolved"], dtype=np.int64))
        mrows = metric_rows(
            scores,
            true_matrix(rec["L"], rec["contacts"]),
            pi,
            pj,
            sep,
            rec["L"],
            with_precision=True,
        )
        rows.extend(
            dict(r, eval_set=rec["eval_set"])
            for r in stamp(
                mrows, rec=rec, model=label, mode="single_seq", predictor="pair_network"
            )
        )
        timings.append(
            {
                "stem": rec["stem"],
                "dataset": rec["dataset"],
                "eval_set": rec["eval_set"],
                "n_residues": rec["L"],
                "n_pairs": len(pi),
                "elapsed_seconds": elapsed,
                "model_nickname": label,
                "mode": "single_seq",
                "gpu_name": torch.cuda.get_device_name()
                if device.type == "cuda"
                else "cpu",
                "gpu_total_memory_gb": torch.cuda.get_device_properties(
                    device
                ).total_memory
                / 2**30
                if device.type == "cuda"
                else 0,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "torch_version": str(torch.__version__),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
        )
        if out is not None:
            np.savez_compressed(
                out / "scores" / f"{rec['dataset']}__{rec['stem']}.npz", score=scores
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["eval_set", "range", "cut"])
        .precision.agg(["mean", "count"])
        .reset_index()
    )
    if out is not None:
        frame.to_csv(out / "per_protein.csv", index=False)
        pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
        summary.to_csv(out / "summary.csv", index=False)
        (out / "summary.json").write_text(
            json.dumps(summary.to_dict("records"), indent=2)
        )
    r = frame[(frame["range"] == "all") & (frame["cut"] == "R")].precision.mean()
    lr = frame[(frame["range"] == "long") & (frame["cut"] == "R")].precision.mean()
    return {
        "r_precision": float(r),
        "long_r_precision": float(lr),
        "proteins": len(records),
        "inference_seconds": sum(t["elapsed_seconds"] for t in timings),
    }
