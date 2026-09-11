"""Record hashes for the exact local upstream artifacts used in this experiment."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
marin = Path("/home/bizon/git/MarinFold")
helico = Path("/home/bizon/git/helico")
record = {
    "marinfold": {
        "repository": "https://github.com/Open-Athena/MarinFold",
        "revision": subprocess.check_output(
            ["git", "-C", str(marin), "rev-parse", "HEAD"], text=True
        ).strip(),
    },
    "helico": {
        "repository": "https://github.com/Open-Athena/helico",
        "revision": subprocess.check_output(
            ["git", "-C", str(helico), "rev-parse", "HEAD"], text=True
        ).strip(),
    },
    "metric_source": "experiments/exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py",
    "benchmark_source": "experiments/exp245_evals_foldbench_held_out_monomers/data",
    "teacher_source": "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/data",
    "teacher_counts_before_sparsa_filter": {"afdb": 3963003, "esm": 65553178},
    "teacher_shards": {"afdb": 2067, "esm": 3338},
    "decontamination": "MarinFold exp225, >=30% identity over >=50% of the shorter sequence against the benchmark reference",
    "contact_definition": {
        "native_amino_acids": True,
        "minimum_degree_in_metric": 0.001,
        "minimum_sequence_separation": 6,
    },
    "selection_split": "eval-val",
    "scored_split_sizes": {"eval-val": 97, "eval-test": 217, "eval-denovo": 19},
    "excluded_from_comparison": {
        "8uxt_A": "MarinFold exp245 context-budget exclusion retained for the paired comparison"
    },
    "baseline_checkpoints": [
        "exp232 decontaminated m2-p06 step145199",
        "exp232 decontaminated m1-p02 step145199",
        "exp199 contaminated cooldown step290400",
    ],
    "files": {},
}
for path in sorted(
    p for p in (root / "data/benchmark").glob("*") if p.name != "provenance.json"
) + [root / "sparsa/vendor/marinfold_metrics.py"]:
    record["files"][str(path.relative_to(root))] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
(root / "data/benchmark/provenance.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
