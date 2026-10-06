"""Visualize the actual training corruption sampler on fixed-validation proteins.

Each timestep is an independent draw from q(x_t | x_0), as in training. Outputs
include per-protein PNGs, an overview, a multipage PDF, raw binary maps, and
provenance. Run with matplotlib installed; no GPU/checkpoint is needed.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from sparsa.data import benchmark, file_sha256
from sparsa.diffusion import BinaryDiffusion
from sparsa.model import tokenize
from sparsa.vendor.marinfold_metrics import true_matrix

COLORS = ["#ffffff", "#edf0f3", "#155f9f", "#db751f"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-0.5, 4), CMAP.N)
DEFAULT_PROTEINS = ["7y54_A", "8baq_A", "8daj_A"]


def sample_levels(record, schedule, seed):
    """Use production forward sampling, retaining all full-sequence residue axes."""
    tokens = tokenize(record["sequence"])[None]
    resolved = np.zeros(record["L"], dtype=bool)
    resolved[np.asarray(record["resolved"], dtype=int)] = True
    eligible_upper = schedule.eligible(tokens)[0].numpy()
    resolved_upper = eligible_upper & resolved[:, None] & resolved[None, :]
    truth_upper = true_matrix(record["L"], record["contacts"]) & resolved_upper
    truth = truth_upper | truth_upper.T
    clean = torch.from_numpy(truth.copy())[None]
    maps, stats = [truth], []
    for step in range(schedule.steps + 1):
        key = f"{seed}:{record['dataset']}:{record['stem']}:{step}".encode()
        draw_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (
            2**63 - 1
        )
        if step:
            generator = torch.Generator().manual_seed(draw_seed)
            sampled = schedule.sample_forward(
                clean, tokens, torch.tensor([step]), generator
            )[0].numpy()
            maps.append(sampled)
        sampled = maps[step]
        assert np.array_equal(sampled, sampled.T)
        assert not np.any(sampled & ~(eligible_upper | eligible_upper.T))
        selected = sampled & resolved_upper
        shared = selected & truth
        stats.append(
            {
                "dataset": record["dataset"],
                "protein": record["stem"],
                "length": record["L"],
                "step": step,
                "alpha_bar": float(schedule.alpha_bar[step]),
                "contacts": int(selected.sum()),
                "shared_with_truth": int(shared.sum()),
                "added_contacts": int((selected & ~truth).sum()),
                "missing_truth_contacts": int((truth_upper & ~sampled).sum()),
                "ground_truth_contacts": int(truth_upper.sum()),
                "draw_seed": draw_seed,
            }
        )
    return np.stack(maps), resolved_upper | resolved_upper.T, stats


def show_panel(ax, sampled, truth, eligible, row, compact=False):
    colors = np.zeros(truth.shape, dtype=np.uint8)
    colors[~eligible] = 1
    colors[sampled & truth & eligible] = 2
    colors[sampled & ~truth & eligible] = 3
    length = truth.shape[0]
    ax.imshow(
        colors,
        cmap=CMAP,
        norm=NORM,
        interpolation="nearest",
        origin="upper",
        extent=(0.5, length + 0.5, length + 0.5, 0.5),
        rasterized=True,
    )
    step = row["step"]
    if step == 0:
        title = "Ground truth · t = 0"
    else:
        title = f"t = {step} · signal {100 * row['alpha_bar']:.1f}%"
    ax.set_title(title, fontsize=10 if compact else 12, weight="bold", pad=8)
    detail = (
        f"{row['contacts']} contacts"
        if step == 0
        else f"{row['shared_with_truth']} shared + {row['added_contacts']} new"
    )
    ax.set_xlabel(detail, fontsize=8 if compact else 10, color="#404953", labelpad=5)
    ticks = [1, length // 2, length]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=7 if compact else 9, length=2, color="#a0a8b0")
    for spine in ax.spines.values():
        spine.set_color("#c5cbd2")
        spine.set_linewidth(0.6)


def legend(fig, y):
    fig.legend(
        handles=[
            Patch(facecolor=COLORS[2], label="Contact also in ground truth"),
            Patch(facecolor=COLORS[3], label="Contact absent from ground truth"),
            Patch(facecolor=COLORS[1], edgecolor="#c5cbd2", label="Excluded pairs"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=3,
        frameon=False,
        fontsize=10,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/protenix_v1_frozen.yaml")
    )
    parser.add_argument("--proteins", nargs="+", default=DEFAULT_PROTEINS)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--out", type=Path, default=Path("reports/contact_noise"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    schedule = BinaryDiffusion(cfg["model"]["diffusion_steps"], cfg["contact_priors"])
    if schedule.steps != 8:
        raise ValueError("The 3 x 3 per-protein layout expects eight noise steps")
    records = {r["stem"]: r for r in benchmark()}
    if any(stem not in records for stem in args.proteins):
        raise ValueError("Choose proteins from the fixed validation split")
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    prepared, rows = [], []
    with PdfPages(args.out / "contact_noise.pdf") as pdf:
        for stem in args.proteins:
            record = records[stem]
            maps, eligible, stats = sample_levels(record, schedule, args.seed)
            prepared.append((record, maps, eligible, stats))
            rows.extend(stats)
            np.savez_compressed(
                args.out / f"{stem}_maps.npz",
                maps=maps,
                eligible=eligible,
                alpha_bar=schedule.alpha_bar.numpy(),
                priors=schedule.priors.numpy(),
            )
            fig, axes = plt.subplots(3, 3, figsize=(12.0, 13.1))
            fig.subplots_adjust(
                left=0.065, right=0.97, bottom=0.12, top=0.9, wspace=0.20, hspace=0.30
            )
            for step, ax in enumerate(axes.flat):
                show_panel(ax, maps[step], maps[0], eligible, stats[step])
            fig.suptitle(
                f"{stem} · {record['L']} residues\nGround truth and the eight training noise levels",
                fontsize=17,
                weight="bold",
                y=0.97,
            )
            legend(fig, 0.055)
            fig.text(
                0.5,
                0.035,
                "Independent draw at each level · “signal” is the original-map weight, not the fraction of contacts remaining.",
                ha="center",
                fontsize=9,
                color="#404953",
            )
            fig.text(
                0.5,
                0.018,
                "Axes: residue positions (1-based). Contacts counted once per pair. Excluded: separation < 6 or unresolved residues.",
                ha="center",
                fontsize=9,
                color="#404953",
            )
            fig.savefig(args.out / f"{stem}_noise.png", dpi=170)
            pdf.savefig(fig, dpi=200)
            plt.close(fig)
        fig, axes = plt.subplots(
            len(prepared), 9, figsize=(27, 3.15 * len(prepared) + 1.35), squeeze=False
        )
        fig.subplots_adjust(
            left=0.055, right=0.985, bottom=0.105, top=0.905, wspace=0.19, hspace=0.30
        )
        for axrow, (record, maps, eligible, stats) in zip(axes, prepared, strict=True):
            for step, ax in enumerate(axrow):
                show_panel(ax, maps[step], maps[0], eligible, stats[step], compact=True)
            axrow[0].set_ylabel(
                f"{record['stem']}\n{record['L']} residues",
                fontsize=12,
                weight="bold",
            )
        fig.suptitle(
            "Binary contact diffusion · ground truth → noise levels 1–8",
            fontsize=21,
            weight="bold",
            y=0.97,
        )
        legend(fig, 0.027)
        fig.text(
            0.5,
            0.016,
            "Each level is sampled independently using the training sampler. At t = 8, contacts depend only on the separation prior.",
            ha="center",
            fontsize=10,
        )
        fig.savefig(args.out / "contact_noise_overview.png", dpi=150)
        plt.close(fig)
    with (args.out / "contact_counts.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "seed": args.seed,
        "proteins": args.proteins,
        "split": "eval-val",
        "held_out_used": False,
        "selection": (
            "Three fully resolved validation proteins at approximately 100, 200 and 300 residues; no model predictions used."
            if args.proteins == DEFAULT_PROTEINS
            else "Explicitly selected fixed-validation proteins; no model predictions used."
        ),
        "sampling": "Independent production BinaryDiffusion.sample_forward draw from q(x_t | x_0) at each t; not a connected forward trajectory.",
        "alpha_bar": schedule.alpha_bar.tolist(),
        "priors": schedule.priors.tolist(),
        "prior_separations": ["6–11", "12–23", "24+"],
        "contact_definition": "Native-amino-acid ConFind degree >= 0.001 and separation >= 6.",
        "display": "Symmetric binary maps; gray excludes unresolved residues and separation < 6; blue shared with truth, orange absent from truth.",
        "benchmark_sha256": {
            p.name: file_sha256(p)
            for p in Path("data/benchmark").glob("*")
            if p.is_file()
        },
        "script_sha256": file_sha256(Path(__file__)),
        "diffusion_code_sha256": file_sha256(Path("sparsa/diffusion.py")),
        "torch_version": str(torch.__version__),
    }
    (args.out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "proteins": args.proteins,
                "levels_per_protein": len(schedule.alpha_bar),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
