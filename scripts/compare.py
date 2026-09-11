"""Paired bootstrap comparisons on identical benchmark scoring universes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare(predictions, baseline_path, manifest_path, seed=17, splits=None):
    predicted = pd.read_csv(predictions)
    if splits is not None:
        predicted = predicted[predicted.eval_set.isin(splits)]
        if set(predicted.eval_set) != set(splits):
            raise ValueError("Requested comparison split is absent")
    baseline = pd.read_csv(baseline_path)
    manifest = pd.read_csv(manifest_path)
    baseline = baseline.merge(
        manifest[["stem", "eval_set"]], on="stem", validate="many_to_one"
    )
    rng = np.random.default_rng(seed)
    result = []
    for (split, metric_range, cut), pred in predicted.groupby(
        ["eval_set", "range", "cut"]
    ):
        for name in baseline.model.unique():
            ref = baseline[
                (baseline.model == name)
                & (baseline.eval_set == split)
                & (baseline["range"] == metric_range)
                & (baseline.cut == cut)
            ]
            pair = pred.merge(
                ref,
                on=["dataset", "stem"],
                suffixes=("_sparsa", "_baseline"),
                validate="one_to_one",
            )
            if len(pair) != len(pred) or len(ref) != len(pred):
                raise ValueError(
                    f"Incomplete paired coverage: {split}/{name}/{metric_range}/{cut}"
                )
            for column in ("n_candidate", "n_true", "n_top"):
                if not np.array_equal(
                    pair[column + "_sparsa"], pair[column + "_baseline"]
                ):
                    raise ValueError(f"Different scoring universe: {column}")
            values = (
                pair[["precision_sparsa", "precision_baseline"]].dropna().to_numpy()
            )
            if not len(values):
                continue
            delta = values[:, 0] - values[:, 1]
            boot = delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(1)
            result.append(
                {
                    "eval_set": split,
                    "range": metric_range,
                    "cut": cut,
                    "baseline": name,
                    "n": len(delta),
                    "sparsa_mean": float(values[:, 0].mean()),
                    "baseline_mean": float(values[:, 1].mean()),
                    "paired_delta": float(delta.mean()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                }
            )
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--baseline", default="data/benchmark/marinfold_baselines.csv")
    parser.add_argument("--manifest", default="data/benchmark/eval_sets.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compare(args.predictions, args.baseline, args.manifest)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(
        result[(result["range"] == "all") & (result.cut == "R")].to_string(index=False)
    )


if __name__ == "__main__":
    main()
