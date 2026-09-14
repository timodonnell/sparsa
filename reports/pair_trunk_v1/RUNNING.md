# Pair-trunk campaign execution

2026-09-14. C0–J1 are implemented and GPU-profiled. The eight-H100 J1
save/resume preflight is running; the coordinator will start screening only if
it passes. No architecture-quality result is available yet.

## Executed protocol

This is the approved [design](PLAN.md) with one measured amendment: **crop 256
for every discovery arm**, including C0. Full triangle attention at crop 512
was too expensive for the intended longer paired comparisons within 1,000
running H100-hours. Validation still uses all 97 full sequences, without crops.
A discovery result at crop 256 does not establish performance at longer crops
or with the 1.8B encoder.

All arms inherit only the same supervised 453M encoder's EMA weights, then
reset their pair network, contact head, optimizer and EMA. Initialization and
source-checkpoint checksums are in [encoder.json](encoder.json). No PLM or MSA
features are used. The encoder is trained jointly with the new trunk.

- Screen C0, C2, J1, A1, D1, T1 and C1 at 25,000 updates, seed 17.
- Use global batch 64, fixed logical records/crop offsets, 50/50 AFDB/ESM
  sampling, the existing BCE plus ranking loss, and one fixed 100k-step
  warmup/constant/decay schedule. Screening supplies 1.6M crops per arm.
- Shortlist the best two feasible alternatives, reserving one place for A1/J1.
  Confirm affordable finalists and C0 at 100,000 updates on seeds 17 and 37
  (6.4M crops per seed). Seed 17 resumes its data, optimizer and schedule.
- Admission uses fresh attempt accounting and measured throughput; if two
  finalists cannot fit, select the highest-validation-scoring affordable one.
  If no complete paired confirmation fits, stop and report screening only.
- Target: mean absolute validation R gain >= 0.005, positive on both seeds,
  mean long-range regression <= 0.002. Protein bootstrap intervals are
  descriptive because validation is reused. No held-out scores guide selection.

## Profile

[Raw measurements](profile.json) include all attempted microbatches, memory,
full-sequence inference at 256/512/920/1024 residues and padding checks.
Times below extrapolate single-H100 synthetic training to a global-64 update
on eight H100s. They exclude DDP, data, checkpoint and validation overhead;
the runner replaces optimistic estimates with observed screening throughput.

| Arm | Total parameters | Microbatch/rank | Estimated s/update | Peak reserved GiB |
|---|---:|---:|---:|---:|
| C0 | 455.65M | 8 | 0.140 | 31.59 |
| C1 | 475.82M | 4 | 0.439 | 50.96 |
| C2 | 457.31M | 8 | 0.163 | 38.60 |
| D1 | 457.55M | 8 | 0.156 | 38.78 |
| T1 | 460.23M | 4 | 0.442 | 42.93 |
| A1 | 461.56M | 4 | 0.920 | 52.17 |
| J1 | 470.12M | 4 | 0.997 | 56.01 |

All seven passed finite forward/backward and full-length inference. Execution
uses no whole-model activation checkpointing at crop 256. Convolution arms
compile pair blocks; modern trunks compile dense triangle/transition kernels.
Triangle attention uses fused four-dimensional SDPA with chunk size 64 and
checkpoints each attention chunk during training.

The initial screen forecast is 254 H100-hours, including a 25% timing margin
and per-job overhead. This is a forecast, not actual usage or a guarantee all
confirmations fit. Each trial uses eight H100s at **batch priority**. Admissions
stop at 925 projected hours; an independent guard requests cancellation at
1,000 cumulative running H100-hours, counting retries. Cancellation latency can
cause a small overrun. Only one training trial runs at a time.

## Operation and recovery

Local ledger: `outputs/pair-trunk-v1-20260914/state.json`.
Remote artifacts:
`s3://marin-us-east-02a/marin/protein-structure/sparsa/pair-trunk-v1-20260914/`.
Worker source and benchmark contracts are frozen in the campaign's `base/` and
`contract.json`; each trial also verifies its configuration hash. Checkpoints
retain every rank's RNG, data cursor, crop-stream digest and exposure counts.
Validation per-protein rows are saved every 5k updates; recovery checkpoints
every 1k. Fixed endpoints, rather than the best intermediate checkpoint, decide
comparisons. Final reports are committed and pushed to main automatically.

The two detached local processes reconnect after restart; they are not a
machine-reboot service. To restart them from the repository, use the virtualenv
interpreter without resolving its symlink to the system Python:

```bash
.venv/bin/python -u -m scripts.guard_compute \
  --config outputs/pair-trunk-v1-20260914/budget.json \
  --out outputs/pair-trunk-v1-20260914/guard
.venv/bin/python -u -m scripts.pair_campaign run
```

Run these in separate supervised sessions, or background each with redirected
logs. File locks reject duplicate processes. Inspect `coordinator.log` and
`guard/status.json`; do not reinitialize an existing campaign. A terminal trial
failure or invalid result pauses in `needs_inspection`. Resolve the recorded
failure before changing stage; do not overwrite a completed trial. To stop new
submissions, stop the coordinator process; cancel an active job explicitly with
Iris if needed, while leaving the compute guard running until jobs terminate.

## Scope and remaining diagnostics

The first implementation records stream digests, sampled source counts,
unpadded residues, positive pairs, training loss/gradient norm and all scheduled
validation rows. It does **not** yet measure unique-protein coverage, modulewise
gate/rank statistics, a fixed teacher-fit panel, or compute-matched validation
curves. Those analyses and mechanism-specific ablations follow useful results.
Third-seed resolution of borderline results, recurrent R1, from-scratch
confirmation, and 1.8B transfer are deferred; none is claimed by this campaign.
No new held-out evaluation or Helico structure-generation benchmark is launched.
