"""Plot fixed-R probability oracle and frequency consensus against matched references.

Legacy consensus values remain valid. Legacy oracle values are never imported.
Requires matplotlib, numpy and pandas; run from an environment with those installed.
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
from plot_diffusion_progress import LABELS

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/diffusion_fixed_r"
LABELS = LABELS | {
    "g4-u4": ("G4-U4", "#00868b", "v"),
    "g4-l4": ("G4-L4", "#65503e", "X"),
    "pf-c128": ("PF-C128", "#238b45", "o"),
    "pf-c256": ("PF-C256", "#006d2c", "s"),
    "pf-n128": ("PF-N128", "#6a51a3", "^"),
    "pf-n128-32gpu": ("PF-N128-32GPU", "#d95f0e", "P"),
    "pf-n256": ("PF-N256", "#3f007d", "D"),
}


def collect():
    rows = {}
    for r in csv.DictReader(
        (ROOT / "reports/diffusion_v3/figures/r_precision_by_step.csv").open()
    ):
        if r["campaign"] == "v1":
            continue
        key = (r["model"], int(r["step"]))
        rows[key] = {
            "model": r["model"],
            "step": int(r["step"]),
            "consensus_r_precision": float(r["consensus_r_precision"]),
            "oracle_r_precision": None,
            "probability_ensemble_r_precision": None,
            "sampled_oracle_r_precision": None,
            "metric_version": "legacy_consensus_only",
            "summary_file": r["summary_file"],
        }
    for path in sorted((REPORT / "milestones").glob("*-summary.json")):
        summary = json.loads(path.read_text())
        if (
            summary.get("metric_version") != "fixed-r-v2"
            or summary["split"] != "eval-val"
            or summary["held_out_used"]
            or summary["proteins"] != 97
            or summary["n_rollouts"] != 100
        ):
            raise ValueError(f"Not a full corrected validation: {path}")
        label = path.name.split("-step")[0]
        model = LABELS[label][0]
        per = pd.read_csv(
            path.with_name(path.name.replace("-summary.json", "-per-protein.csv"))
        )
        if len(per) != 97 or not per.metric_version.eq("fixed-r-v2").all():
            raise ValueError(f"Inconsistent per-protein records: {path}")
        for metric in ["oracle_r_precision", "consensus_r_precision"]:
            np.testing.assert_allclose(per[metric].mean(), summary[metric])
        rows[(model, summary["step"])] = {
            "model": model,
            "step": summary["step"],
            **{
                k: summary[k]
                for k in [
                    "consensus_r_precision",
                    "oracle_r_precision",
                    "probability_ensemble_r_precision",
                    "sampled_oracle_r_precision",
                ]
            },
            "metric_version": summary["metric_version"],
            "summary_file": str(path.relative_to(ROOT)),
        }
    return sorted(rows.values(), key=lambda r: (r["model"], r["step"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of", default=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    )
    parser.add_argument(
        "--models", nargs="+", help="Model labels, e.g. PF-C128 PF-C256 PF-N128 PF-N256"
    )
    parser.add_argument("--out", type=Path, default=REPORT / "figures")
    args = parser.parse_args()
    rows = collect()
    if args.models:
        rows = [r for r in rows if r["model"] in args.models]
    if not rows:
        raise SystemExit(
            "No completed scientific validation checkpoints for the selected models yet."
        )
    data = pd.DataFrame(rows)
    ref = json.loads((REPORT / "marinfold-iid100-reference.json").read_text())[
        "metrics"
    ]["all"]
    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    with (out / "r_precision_by_step.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.hashsalt": "sparsa-fixed-r-v2",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    for ax, metric, title, baseline in zip(
        axes,
        ["oracle_r_precision", "consensus_r_precision"],
        [
            "Oracle best-of-100 · denoiser probability ranking",
            "Consensus · contact frequency across 100 rollouts",
        ],
        [ref["oracle_r_precision"], ref["consensus_r_precision"]],
        strict=True,
    ):
        any_points = False
        for model, color, marker in LABELS.values():
            d = data[data.model.eq(model)].dropna(subset=[metric]).sort_values("step")
            if d.empty:
                continue
            any_points = True
            ax.plot(
                d.step / 1000,
                d[metric],
                color=color,
                marker=marker,
                ms=5,
                lw=1.8,
                label=model,
            )
        ax.axhline(baseline, color="#963547", ls="--", lw=1.5)
        ax.text(
            4,
            baseline + 0.012,
            f"MarinFold {'oracle' if metric.startswith('oracle') else 'consensus'}: {baseline:.4f}",
            color="#963547",
            fontsize=10,
        )
        if not any_points:
            ax.text(
                0.5,
                0.4,
                "Corrected evaluations running\nLegacy oracle scores omitted",
                transform=ax.transAxes,
                ha="center",
                color="#606872",
            )
        ax.set_title(title, fontsize=11, loc="left", pad=12)
        ax.set_xlim(0, max(325, data.step.max() / 1000 + 25))
        ax.set_ylim(0, 0.61)
        ax.set_xlabel("Training steps (thousands)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Fixed-R precision: true positives / ground-truth contacts")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=5,
        loc="upper left",
        bbox_to_anchor=(0.07, 0.895),
        frameon=False,
        fontsize=10,
    )
    title_artist = fig.text(
        0.08,
        0.97,
        "Corrected R-precision comparisons",
        fontsize=20,
        weight="bold",
        va="top",
    )
    fig.text(
        0.08,
        0.925,
        f"97 fixed validation proteins · 100 rollouts · {args.as_of}",
        color="#505760",
        va="top",
    )
    fig.text(
        0.08,
        0.025,
        "Separate matched MarinFold references. Historical consensus is valid; legacy oracle scores are excluded.\nOracle ranks each rollout before selecting its best score; consensus pools the rollouts into one ranking.",
        fontsize=9,
        color="#606872",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.77, wspace=0.13)
    for name, early in [
        ("r_precision_by_step", False),
        ("r_precision_early_by_step", True),
    ]:
        if early:
            title_artist.set_text("Early training · corrected R-precision")
            for ax in axes:
                ax.set_xlim(0, 105)
        for ext in ("png", "svg"):
            path = out / f"{name}.{ext}"
            fig.savefig(
                path, dpi=180, metadata={"Date": None} if ext == "svg" else None
            )
            if ext == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines())
                    + "\n"
                )
    plt.close(fig)
    print(
        f"Plotted {data.consensus_r_precision.notna().sum()} consensus and {data.oracle_r_precision.notna().sum()} corrected oracle checkpoints."
    )


if __name__ == "__main__":
    main()
