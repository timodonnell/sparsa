"""Regenerate validation curves from recorded diffusion milestones.

Run with a Python environment containing matplotlib, numpy, and pandas.
Existing pointwise bootstrap intervals are retained; new ones use 100,000
protein resamples with seed 20261001 + checkpoint step. No held-out data is read.
"""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "g1-long": ("G1-long", "#3973ac", "o"),
    "g2-wide": ("G2-wide", "#dc6173", "s"),
    "g2-wide-sc2": ("G2-wide-SC", "#258443", "D"),
    "g2-rsc": ("G2-RSC", "#c27a12", "^"),
    "g3-balanced-rsc": ("G3-balanced-RSC", "#764bb5", "P"),
}


def collect_rows():
    """Keep v1 pilot records and collect all full v2/v3 milestone evaluations."""
    previous = ROOT / "reports/diffusion_v2/figures/r_precision_by_step.csv"
    with previous.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {(r["model"], int(r["step"])): r for r in rows}
    combined = ROOT / "reports/diffusion_v3/figures/r_precision_by_step.csv"
    if combined.exists():
        with combined.open(newline="") as handle:
            by_key.update(
                {(r["model"], int(r["step"])): r for r in csv.DictReader(handle)}
            )
    for campaign in ("v2", "v3"):
        directory = ROOT / f"reports/diffusion_{campaign}/milestones"
        for path in sorted(directory.glob("*-summary.json")):
            label, rest = path.name.split("-step", 1)
            if label not in LABELS:
                continue
            step = int(rest.split("-", 1)[0])
            summary = json.loads(path.read_text())
            if summary.get("metric_version") is not None:
                raise ValueError("Do not mix corrected metrics into legacy curves")
            per_path = path.with_name(
                path.name.replace("-summary.json", "-per-protein.csv")
            )
            per = pd.read_csv(per_path).sort_values(["dataset", "stem"])
            if (
                len(per) != 97
                or not per.eval_set.eq("eval-val").all()
                or not per.n_rollouts.eq(100).all()
                or summary.get("held_out_used", False)
                or summary.get("rollout_seed", 20260925) != 20260925
            ):
                raise ValueError(
                    f"Not a comparable fixed-validation evaluation: {path}"
                )
            values = per.oracle_r_precision.to_numpy()
            np.testing.assert_allclose(values.mean(), summary["oracle_r_precision"])
            model = LABELS[label][0]
            key = (model, step)
            if key in by_key:
                lo, hi = [
                    float(by_key[key][k])
                    for k in ("bootstrap_95_low", "bootstrap_95_high")
                ]
            else:
                rng = np.random.default_rng(20261001 + step)
                indices = rng.integers(0, len(values), size=(100000, len(values)))
                lo, hi = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
            by_key[key] = {
                "campaign": campaign,
                "model": model,
                "step": step,
                "step_thousands": step / 1000,
                "proteins": 97,
                "rollouts": 100,
                "rollout_seed": 20260925,
                "oracle_r_precision": summary["oracle_r_precision"],
                "bootstrap_95_low": lo,
                "bootstrap_95_high": hi,
                "long_oracle_r_precision": summary["oracle_long_r_precision"],
                "mean_rollout_r_precision": summary["mean_r_precision"],
                "consensus_r_precision": summary["consensus_r_precision"],
                "summary_file": str(path.relative_to(ROOT)),
            }
    rows = sorted(
        by_key.values(), key=lambda r: (r["campaign"], r["model"], int(r["step"]))
    )
    return rows


def render(rows, destination, early_panel, as_of):
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "r_precision_by_step.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    data = pd.DataFrame(rows)
    for col in (
        "step_thousands",
        "oracle_r_precision",
        "bootstrap_95_low",
        "bootstrap_95_high",
    ):
        data[col] = data[col].astype(float)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.hashsalt": "sparsa-diffusion",
        }
    )
    if early_panel:
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 7.6), gridspec_kw={"width_ratios": [2.2, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(12.5, 7.6))
        axes = [ax]
    handles = []
    for axis_index, ax in enumerate(axes):
        for model, d in data[data.campaign == "v1"].groupby("model"):
            d = d.sort_values("step_thousands")
            ax.plot(
                d.step_thousands,
                d.oracle_r_precision,
                "--",
                color="#a6a9ad",
                alpha=0.55,
                lw=1.2,
            )
            ax.scatter(
                d.step_thousands,
                d.oracle_r_precision,
                color="#a6a9ad",
                alpha=0.55,
                s=16,
            )
        for model, color, marker in LABELS.values():
            d = data[data.model == model].sort_values("step_thousands")
            if d.empty:
                continue
            y = d.oracle_r_precision.to_numpy()
            x = d.step_thousands.to_numpy()
            err = [
                y - d.bootstrap_95_low.to_numpy(),
                d.bootstrap_95_high.to_numpy() - y,
            ]
            ax.errorbar(
                x,
                y,
                yerr=err,
                fmt="none",
                ecolor=color,
                alpha=0.20,
                lw=1.2,
                capsize=2,
                zorder=1,
            )
            (line,) = ax.plot(
                x, y, color=color, marker=marker, ms=5.5, lw=2.2, label=model, zorder=3
            )
            if axis_index == 0:
                handles.append(line)
            if axis_index == 0 and (
                not early_panel or model in ("G1-long", "G2-wide", "G2-wide-SC")
            ):
                ax.annotate(
                    f"{y[-1]:.3f}",
                    (x[-1], y[-1]),
                    xytext=(7, 4),
                    textcoords="offset points",
                    color=color,
                    fontsize=10,
                    weight="bold",
                )
            if axis_index == 1 and model in ("G2-RSC", "G3-balanced-RSC"):
                for xx, yy in zip(x, y, strict=True):
                    ax.annotate(
                        f"{yy:.3f}",
                        (xx, yy),
                        xytext=(7, 5),
                        textcoords="offset points",
                        color=color,
                        fontsize=10,
                        weight="bold",
                    )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#e3e7eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0.12, 0.565)
        ax.set_xlabel("Training steps (thousands)", labelpad=10)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_xlim(
            0, max(285, data.step_thousands.max() + 25) if axis_index == 0 else 40
        )
        ax.xaxis.set_major_locator(MultipleLocator(50 if axis_index == 0 else 10))
        ax.set_title(
            "Full training history"
            if axis_index == 0
            else "Early training: first 40k steps",
            loc="left",
            fontsize=11,
            pad=12,
        )
    axes[0].set_ylabel("Legacy oracle precision among emitted contacts", labelpad=10)
    handles.append(Line2D([0], [0], color="#a6a9ad", ls="--", label="v1 pilot models"))
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.071, 0.877),
        ncol=3 if early_panel else 4,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.08,
        0.965,
        "Legacy scores — not R-precision",
        fontsize=21,
        weight="bold",
        va="top",
    )
    fig.text(
        0.08,
        0.918,
        f"Fixed 97-protein validation split · 100 rollouts per protein · {as_of}",
        fontsize=12,
        color="#505760",
        va="top",
    )
    fig.text(
        0.08,
        0.025,
        "Bars: pointwise 95% protein-bootstrap intervals. Lines connect evaluated checkpoints; no extrapolation.\n"
        "Incorrect denominator for sparse maps. Retained for audit only; use reports/diffusion_fixed_r for corrected comparisons.",
        fontsize=9,
        color="#606872",
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.715, wspace=0.24)
    for ext in ("png", "svg"):
        path = destination / f"r_precision_by_step.{ext}"
        fig.savefig(path, dpi=200, metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    args = parser.parse_args()
    rows = collect_rows()
    render(
        [r for r in rows if r["campaign"] != "v3"],
        ROOT / "reports/diffusion_v2/figures",
        early_panel=False,
        as_of=args.as_of,
    )
    render(
        rows, ROOT / "reports/diffusion_v3/figures", early_panel=True, as_of=args.as_of
    )
    print(
        f"Plotted {len(rows)} validation checkpoints; refreshed v2 and combined v3 figures."
    )
