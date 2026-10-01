"""Pin MarinFold exp321 iid100 references on Sparsa's frozen eval-val proteins."""

import argparse
import json
from pathlib import Path

import pandas as pd

from sparsa.data import benchmark, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marinfold-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/diffusion_fixed_r"))
    args = parser.parse_args()
    relative = Path(
        "experiments/exp321_evals_null_sequence_contrastive_guidance_for_contacts"
    )
    source = args.marinfold_root / relative
    records = benchmark()
    expected = {r["stem"]: r for r in records}
    targets = pd.read_csv(source / "data/targets.csv")
    targets = targets[targets.cohort.eq("eval-val")]
    assert len(targets) == len(expected) == 97 and set(targets.stem) == set(expected)
    for row in targets.itertuples():
        rec = expected[row.stem]
        assert (
            row.dataset == rec["dataset"]
            and row.sequence == rec["sequence"]
            and row.L == rec["L"]
        )
    # "heldout" in exp321 means the other 81 eval-val proteins, NOT eval-test.
    paths = [source / "data/full_dev_natural.csv", source / "data/heldout_natural.csv"]
    data = pd.concat([pd.read_csv(path) for path in paths])
    data = data[data["mode"].eq("full_iid_single") & data.N.eq(100)].copy()
    assert set(data.stem) == set(expected) and set(data["range"]) == {"all", "long"}
    assert not data.duplicated(["stem", "range"]).any() and len(data) == 194
    truth_path = (
        args.marinfold_root
        / "experiments/exp245_evals_foldbench_held_out_monomers/data/gt_universe_scored.jsonl"
    )
    upstream_truth = {
        r["stem"]: r
        for line in truth_path.open()
        if (r := json.loads(line))["stem"] in expected
    }
    for stem, rec in expected.items():
        upstream = upstream_truth[stem]
        assert (
            rec["resolved"] == upstream["resolved"]
            and rec["contacts"] == upstream["contacts"]
        )
    data["dataset"] = data.stem.map(
        {stem: rec["dataset"] for stem, rec in expected.items()}
    )
    data["eval_set"] = "eval-val"
    data["n_rollouts"] = 100
    data = data[
        [
            "dataset",
            "stem",
            "eval_set",
            "range",
            "n_rollouts",
            "consensus_r_precision",
            "validity_gated_oracle_r_precision",
            "oracle_r_precision",
            "mean_rollout_precision",
            "mean_contacts",
        ]
    ].sort_values(["dataset", "stem", "range"])
    args.out.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out / "marinfold-iid100-per-protein.csv", index=False)
    used = paths + [
        source / "data/targets.csv",
        source / "analyze_results.py",
        source / "README.md",
        truth_path,
    ]
    reference = {
        "model": "contacts-v1-exp277-m2-p06-full-epoch-1.5B",
        "checkpoint_step": 266344,
        "experiment": "MarinFold exp321",
        "arm": "full_iid_single",
        "split": "eval-val",
        "proteins": 97,
        "n_rollouts": 100,
        "held_out_used": False,
        "consensus_ranking": "sampled_contact_frequency_all_eligible_pairs",
        "oracle_ranking": "unique_contacts_in_emission_order",
        "precision_denominator": "ground_truth_contacts_in_resolved_range",
        "oracle_validity_gate": "unfinished or malformed rollout scores zero",
        "temperature": 1.0,
        "top_p": 0.95,
        "provenance": "Imported published per-protein values; verified identical sequences, ground truth and resolved residues for all 97 proteins; scorer source uses fixed R. Raw rollout artifacts not present locally.",
        "source_sha256": {
            str(p.relative_to(args.marinfold_root)): file_sha256(p) for p in used
        },
        "metrics": {
            region: {
                "consensus_r_precision": float(group.consensus_r_precision.mean()),
                "oracle_r_precision": float(
                    group.validity_gated_oracle_r_precision.mean()
                ),
                "ungated_oracle_r_precision": float(group.oracle_r_precision.mean()),
            }
            for region, group in data.groupby("range")
        },
    }
    (args.out / "marinfold-iid100-reference.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
    print(json.dumps(reference["metrics"], indent=2))


if __name__ == "__main__":
    main()
