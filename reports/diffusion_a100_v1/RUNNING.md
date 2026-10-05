# Dedicated A100 run status

**Retired October 5 at the user's request.** Both services were explicitly stopped
at logged step 10,750; their step-10k checkpoints are retained. The nodes are being
reassigned to the [Pairformer campaign](../pairformer_v1/PLAN.md). The status below
records the earlier G4 campaign and is historical.

Both production runs launched **October 2, 2026 at 01:55:55 UTC**
(October 1 at 9:55:55 p.m. ET). The October 5 check found both runs stuck in
repeated evaluation timeouts at step 10,000. The services had restarted 129 / 128
times without further optimizer progress. Both were stopped and resumed from
the intact checkpoints with the [evaluation repair](RECOVERY-20261005.md).
The [plan](PLAN.md) describes the scientific comparison.

| Arm | Parameters | Hardware | Service | Latest saved step |
|---|---:|---|---|---:|
| G4-U4 | 315,297,028 | a100-1: 8 A100-SXM4-80GB | `sparsa-g4-u4-s23` | 10,000 |
| G4-L4 | 309,359,620 | a100-2: 8 A100-SXM4-80GB | `sparsa-g4-l4-s23` | 10,000 |

U4 has four independent triangle blocks; L4 reuses one block four times.
Both use a 24-layer, width-1024 sequence encoder, width-256 pairs, eight reverse
diffusion steps, crop 384 and global batch 128. The fixed horizon is 300k steps
(38.4M crop presentations per arm), with EMA validation every 10k steps on
97 proteins × 100 rollouts. No self-conditioning or pretrained features.

Training uses the complete local teacher inventory: 2,067 AFDB and 3,338 ESM
Parquet files, totaling 139,371,683,675 bytes. Footer counts are 3,963,003 and
65,553,178 document rows, respectively; these are **not unique-protein counts**.
Size checks, download-time content hashes and readable footers agree on both
nodes. Production provenance confirms no shard limit. At saved step 10,000,
each run had consumed **1,280,000 crops**. The checkpoints have
matching per-rank data digests and source counts between arms.

## October 5 recovery

Recovery services started at 13:30 UTC from source snapshot `07af40f`. The
runtime now records a four-hour collective timeout, balanced validation work
and per-protein progress. Original model/data/optimizer provenance is preserved;
`runtime.json` identifies the active source separately. All 84 local tests passed.
Both full production evaluations completed: all 97 validation proteins with
100 rollouts each. By **14:01 UTC**, both runs had advanced to logged step
**10,050**, with finite losses/gradients, 9.783 seconds per step, zero service
restarts and no post-recovery errors. The latest durable checkpoint is still
10k; the next scheduled save is 11k. Evidence: `training-progress-20261005.json`.

| Arm | Evaluated step | Probability oracle @100 | Frequency consensus |
|---|---:|---:|---:|
| G4-U4 | 10,000 | 0.214326 | 0.211588 |
| G4-L4 | 10,000 | 0.217606 | 0.215978 |

L4 minus U4 is +0.003279 oracle (paired protein bootstrap 95% interval
[-0.003134, 0.010476]) and +0.004390 consensus [-0.002297, 0.010876]. Neither
metric establishes a winner. Both are still within the 15k-step warmup.
The [corrected comparison plots](../diffusion_fixed_r/README.md) now include
both first evaluations. Review points remain 30k and 90k.

## October 2 afternoon update

Both runs have trained for about sixteen hours without restarts or logged
errors, advancing 2,150 steps since the morning check. GPU allocation remains
eight A100s per model. Both are still in the 15k-step warmup.

| Arm | Latest logged step | Latest loss | Mean of last 10 logged losses | Median recent seconds / step |
|---|---:|---:|---:|---:|
| G4-U4 | 5,850 | 0.123013 | 0.146991 | 9.766 |
| G4-L4 | 5,850 | 0.121704 | 0.145045 | 9.766 |

Recent mean losses have fallen from 0.159904 / 0.159511 at the morning check.
L4 has the slightly lower teacher training loss on matched data, but no full
validation exists yet, so this is not evidence of better contact R-precision.
Step 10k is projected around **05:15 UTC / 1:15 a.m. ET on October 3**, roughly
11.3 hours after this check, followed by evaluation time. Evaluation remains
97 validation proteins × 100 rollouts; the 300k training horizon is unchanged.
Evidence: `training-progress-20261002-pm.json` and the current status snapshots.

## October 2 morning update

Both runs have trained uninterrupted for about ten hours, with no logged
tracebacks, OOMs or non-finite gradient failures. Both remain in the 15k-step
learning-rate warmup. Loss is the teacher training objective, not experimental
R-precision; it does not establish an architecture winner.

| Arm | Latest logged step | Latest loss | Mean of last 10 logged losses | Median recent seconds / step |
|---|---:|---:|---:|---:|
| G4-U4 | 3,700 | 0.159754 | 0.159904 | 9.759 |
| G4-L4 | 3,700 | 0.158821 | 0.159511 | 9.770 |

The first logged losses at step 50 were 0.385112 / 0.373878, respectively.
Neither run has a completed full validation. At current throughput they should
reach 10k around **05:10 UTC / 1:10 a.m. ET on October 3**, approximately 17 hours
after this check, followed by evaluation time. The 300k horizon is unchanged.
Evidence: `training-progress-20261002.json` and the refreshed status snapshots.

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
  checkpoint resume on process failure (the October 5 repair changes the limit
  to three starts per six hours).
  Checkpoints are local and atomic; every 10k milestone is preserved in addition
  to the three newest recovery checkpoints. Existing Iris jobs were untouched.

Measured steady eight-GPU smoke steps were **9.82–9.85 s U4** and **9.80–9.81 s
L4**, excluding checkpoint pauses, with about 54.4 / 54.3 GiB peak allocation.
This implies roughly **27–28 hours to step 10k**, plus evaluation time, and
about **34 days to 300k before overhead**. These are initial throughput estimates,
not completed training results. The original estimate put the first evaluation
on October 3; its repeated timeout delayed completion, as recorded above.

Smoke metrics are excluded from learning curves and architecture selection.
The primary measure is fixed-R
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
both smoke directories were correctly rejected. Plot collection recognizes both
G4 labels and includes their complete 10k evaluations.

Evidence: `g4-*-launch.json`, `g4-*-status.json`, `g4-*-smoke.json`,
`dataset-a100-*.json`, and `profiles/`. Status snapshots intentionally omit
connection addresses and credentials.
