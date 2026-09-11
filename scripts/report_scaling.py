"""Publish compact, auditable results after completed scaling evaluation."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def paired_release_comparison(current, baseline):
    rows = []
    rng = np.random.default_rng(17)
    for split, count in [("eval-val", 97), ("eval-test", 217), ("eval-denovo", 19)]:
        for distance in ("all", "long"):
            a = current[
                (current.eval_set == split)
                & (current["range"] == distance)
                & (current.cut == "R")
            ]
            b = baseline[
                (baseline.eval_set == split)
                & (baseline["range"] == distance)
                & (baseline.cut == "R")
            ]
            pair = a.merge(
                b,
                on=["dataset", "stem"],
                suffixes=("_new", "_old"),
                validate="one_to_one",
            )
            if len(pair) != count or len(a) != count or len(b) != count:
                raise ValueError("Paired protein coverage differs")
            for column in ("n_candidate", "n_true", "n_top"):
                if not np.array_equal(pair[column + "_new"], pair[column + "_old"]):
                    raise ValueError("Contact scoring universe differs")
            delta = (pair.precision_new - pair.precision_old).dropna().to_numpy()
            boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
            rows.append(
                {
                    "eval_set": split,
                    "range": distance,
                    "n": len(delta),
                    "new_mean": pair.precision_new.mean(),
                    "old_mean": pair.precision_old.mean(),
                    "delta": delta.mean(),
                    "ci_low": np.quantile(boot, 0.025),
                    "ci_high": np.quantile(boot, 0.975),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluation-uri", required=True)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifacts, out = args.artifacts, args.out
    manifest = json.loads((artifacts / "evaluation_manifest.json").read_text())
    if (
        manifest["test_used_for_selection"]
        or not json.loads((artifacts / "complete.json").read_text())["complete"]
    ):
        raise ValueError(
            "Expected completed evaluation after validation-only selection"
        )
    if (
        manifest["benchmark_sha256"]
        != json.loads((root / "reports/final/evaluation_manifest.json").read_text())[
            "benchmark_sha256"
        ]
    ):
        raise ValueError("Benchmark changed from the released reference")
    out.mkdir(parents=True, exist_ok=True)
    for name in [
        "summary.csv",
        "per_protein.csv",
        "paired_comparisons.csv",
        "paired_comparisons_step363000.csv",
        "evaluation_manifest.json",
        "candidate_run_selection.json",
        "model_config.json",
        "SHA256SUMS.json",
        "complete.json",
    ]:
        shutil.copyfile(artifacts / name, out / name)
    for stem in ["7pv5_A", "7znz_A"]:
        verification = artifacts.parent / f"release-{stem}" / "verification.json"
        data = json.loads(verification.read_text())
        if not data["helico"]["strict_parser_passed"]:
            raise ValueError("Helico parser verification failed")
        if (
            data["checkpoint_sha256"]
            != json.loads((artifacts / "SHA256SUMS.json").read_text())["sparsa.pt"]
        ):
            raise ValueError("Verified checkpoint differs from released artifact")
        shutil.copyfile(verification, out / f"helico-{stem}.json")
    paired = paired_release_comparison(
        pd.read_csv(artifacts / "per_protein.csv"),
        pd.read_csv(root / "reports/final/per_protein.csv"),
    )
    paired.to_csv(out / "paired_previous_release.csv", index=False)
    marin = pd.read_csv(artifacts / "paired_comparisons.csv")
    marin = marin[
        (marin.baseline == "marinfold-exp232-decontam-m2-p06-step145199")
        & (marin["range"] == "all")
        & (marin.cut == "R")
    ].set_index("eval_set")
    accounting = json.loads(args.accounting.read_text())
    summary = [
        "# Scaling campaign results",
        "",
        "Selection used the fixed 97-protein validation split. The held-out sets were evaluated after selection.",
        "",
        "| Split | Proteins | Selected Sparsa | Previous 40M | MarinFold¹ |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ["eval-val", "eval-test", "eval-denovo"]:
        row = paired[(paired.eval_set == split) & (paired["range"] == "all")].iloc[0]
        summary.append(
            f"| {split} | {row['n']} | {row.new_mean:.6f} | {row.old_mean:.6f} | {marin.loc[split, 'baseline_mean']:.6f} |"
        )
    test = paired[(paired.eval_set == "eval-test") & (paired["range"] == "all")].iloc[0]
    summary += [
        "",
        "¹ Decontaminated exp232 m2-p06, step 145199.",
        "",
        f"Test R-precision change versus the previous release: {test.delta:+.6f}; paired bootstrap 95% interval [{test.ci_low:+.6f}, {test.ci_high:+.6f}].",
        "Validation intervals are descriptive because validation was reused for adaptive selection.",
        "",
        f"Selected model: {manifest['model_parameters']:,} parameters from `{manifest['selected_training_run']}`.",
        f"Checkpoint: `{args.evaluation_uri}/sparsa.pt`.",
        f"SHA256: `{json.loads((artifacts / 'SHA256SUMS.json').read_text())['sparsa.pt']}`.",
        f"Campaign running GPU time, including retries: {accounting['running_gpu_hours']:.3f} H100-hours (see accounting timing qualifications).",
        "",
        "Actual CLI exports and the Helico contact parser passed for 81- and 761-residue validation chains. This verifies contact-input compatibility, not downstream structure quality.",
        "",
        "Full per-protein scores, paired comparisons, source hashes, model selection, and verification records are in [final/](final/). The original report remains in [../FINAL.md](../FINAL.md).",
    ]
    (out.parent / "FINAL.md").write_text("\n".join(summary) + "\n")
    readme = root / "README.md"
    old = (
        "A [scaling campaign](reports/scaling_v2/PLAN.md)\n"
        "is now testing 456M and 1.8B models, teacher mixtures, ranking loss, and longer\n"
        "training against continuation controls."
    )
    text = readme.read_text()
    if args.update_readme and old in text:
        readme.write_text(
            text.replace(
                old,
                "The larger/longer [scaling campaign results](reports/scaling_v2/FINAL.md) are also available.",
            )
        )


if __name__ == "__main__":
    main()
