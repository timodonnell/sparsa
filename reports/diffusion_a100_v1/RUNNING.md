# Dedicated A100 run status

Both production runs launched **October 2, 2026 at 01:55:55 UTC**
(October 1 at 9:55:55 p.m. ET). At 01:58:43 UTC both services were active,
with zero restarts and durable step-10 checkpoints. All sixteen GPUs were
allocated to training. The [plan](PLAN.md) describes the scientific comparison.

| Arm | Parameters | Hardware | Service | Latest saved step |
|---|---:|---|---|---:|
| G4-U4 | 315,297,028 | a100-1: 8 A100-SXM4-80GB | `sparsa-g4-u4-s23` | 10 |
| G4-L4 | 309,359,620 | a100-2: 8 A100-SXM4-80GB | `sparsa-g4-l4-s23` | 10 |

U4 has four independent triangle blocks; L4 reuses one block four times.
Both use a 24-layer, width-1024 sequence encoder, width-256 pairs, eight reverse
diffusion steps, crop 384 and global batch 128. The fixed horizon is 300k steps
(38.4M crop presentations per arm), with EMA validation every 10k steps on
97 proteins × 100 rollouts. No self-conditioning or pretrained features.

Training uses the complete local teacher inventory: 2,067 AFDB and 3,338 ESM
Parquet files, totaling 139,371,683,675 bytes. Footer counts are 3,963,003 and
65,553,178 document rows, respectively; these are **not unique-protein counts**.
Size checks, download-time content hashes and readable footers agree on both
nodes. Production provenance confirms no shard limit. At step 10, each run had
consumed 1,280 crops, with matching per-rank data digests and source counts
between arms.

## Launch verification

- **82 local tests passed**, including gradients through four tied/untied pair
  updates, atomic checkpoint failure handling and milestone retention.
- Both real eight-GPU smoke runs completed four optimizer steps, checkpointed,
  and completed a fixed-R evaluation on one validation protein with two rollouts.
- Both restored step 2 and trained through step 4 again. Data digests and counts
  matched on all eight ranks. Resumed losses matched within 2.1e-6; bitwise
  floating-point reproducibility is not claimed.
- Both sampled the longest validation protein (761 residues), two complete
  eight-step rollouts each, with finite ranked scores and about 7.6 GiB peak
  allocation in this inference-only check.
- Production source hashes and frozen benchmark hashes match commit
  `6a95e8428364dc558a36eb9b2c817670cf7325a0`. Each run uses an immutable source
  snapshot. The first ~5 GB checkpoint was loaded on CPU to verify optimizer
  step, eight-rank state and exposure counts.
- User services remain supervised after SSH disconnects, with automatic
  checkpoint resume on process failure (three starts per 30-minute limit).
  Checkpoints are local and atomic; every 10k milestone is preserved in addition
  to the three newest recovery checkpoints. Existing Iris jobs were untouched.

Measured steady eight-GPU smoke steps were **9.82–9.85 s U4** and **9.80–9.81 s
L4**, excluding checkpoint pauses, with about 54.4 / 54.3 GiB peak allocation.
This implies roughly **27–28 hours to step 10k**, plus evaluation time, and
about **34 days to 300k before overhead**. These are initial throughput estimates,
not completed training results. The first full evaluation should follow step
10k around October 3 UTC if this throughput holds.

Neither arm has a full scientific evaluation yet. Smoke metrics are excluded
from learning curves and architecture selection. The primary measure is fixed-R
probability oracle best-of-100; frequency consensus has its own matched MarinFold
reference. Review points are 30k and 90k; no automated promotion has been made.

## Recovery and inspection

On each destination node, outputs are under
`/home/ubuntu/sparsa-runs/g4-{u4,l4}-s23-300k`. Inspect the corresponding user
service with `systemctl --user status sparsa-g4-u4-s23` (or `l4`) and read
`train.log`, `latest.json`, `evaluation_policy.json` and `provenance.json` in
the output directory. The launch records here contain exact commands and config
hashes; connection details are supplied separately.

Completed evaluations can be imported into the existing comparison plot:

```bash
uv run python scripts/fetch_fixed_r_evaluations.py \
  --host '<SSH destination for a100-1>' --model g4-u4 \
  --run-dir /home/ubuntu/sparsa-runs/g4-u4-s23-300k
uv run python scripts/fetch_fixed_r_evaluations.py \
  --host '<SSH destination for a100-2>' --model g4-l4 \
  --run-dir /home/ubuntu/sparsa-runs/g4-l4-s23-300k
python scripts/plot_fixed_r_progress.py
```

The importer verifies all 97 identities, rollout counts, metric version and
per-protein means. Recovery was exercised on both production directories, and
both smoke directories were correctly rejected. Plot collection still accepts
the existing 45 validation checkpoints and now recognizes both G4 labels.

Evidence: `g4-*-launch.json`, `g4-*-status.json`, `g4-*-smoke.json`,
`dataset-a100-*.json`, and `profiles/`. Status snapshots intentionally omit
connection addresses and credentials.
