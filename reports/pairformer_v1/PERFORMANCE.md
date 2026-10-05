# Pairformer bookkeeping optimization — 2026-10-05

A short single-H100 benchmark reduced full-model microstep time by 14.5% at
pair width 128 and 13.1% at width 256. The model, attention kernels, BF16
autocast, loss, optimizer, EMA, crop and microbatch remain unchanged.

| Pair width | Original microstep | Optimized microstep | Original peak | Optimized peak |
| --- | ---: | ---: | ---: | ---: |
| 128 | 9.734 s | 8.324 s | 28.22 GiB | 32.45 GiB |
| 256 | 14.617 s | 12.700 s | 55.12 GiB | 60.20 GiB |

These are synthetic full 48-block model forward/backward/AdamW/EMA timings,
batch 4, crop 384, with one warmup and three measured microsteps per variant.
They exclude data I/O, DDP, checkpoint writes and validation. Multiplying by
four accumulation microbatches gives approximately 38.9→33.3 s and
58.5→50.8 s per logical step; these are estimates, not observed production
speedups. A100 end-to-end improvement has not been measured.

The job `/bizon/sparsa-pairformer-overhead-20261005` used one H100 at **batch**
priority and completed. Raw results: [overhead-h100.json](overhead-h100.json).

## Changes

- Use `split` for disjoint attention chunks. This removes the repeated
  full-pair gradient buffers produced by slice backward while preserving
  chunk dimensions and attention calculations.
- Retain the outer Pairformer block checkpoint and remove inner attention
  chunk checkpoints when that outer checkpoint is enabled. This avoids an
  extra attention recomputation, at the measured additional memory cost.
  Models without outer block checkpointing retain chunk checkpointing.

The original attention forward remains in
`scripts/profile_pairformer_overhead.py` as a benchmark and regression reference.
Checkpoint parameters/configuration and state-dict keys are unchanged.

## Numerical checks

- 101 CPU tests passed, including exact logits, gradients and two AdamW
  updates against the original implementation for both cached/noisy modes,
  with and without outer checkpointing. Residual projections were made
  nonzero; padding and partial chunks were included.
- BF16 CUDA checks at both production pair widths, length 384, batch 2,
  compared all block outputs and parameter/input gradients (62 tensors).
  With deterministic kernels, every tensor matched bit-for-bit. These checks
  ran on the local RTX A5000; see
  [overhead-numerical-check.json](overhead-numerical-check.json).
- Ordinary training kernels are not deterministic. This does not promise
  identical future training trajectories between independent executions.
  Deterministic mode was used only for verification, not enabled in training.

Reproduce the numerical check:

```sh
python -m scripts.profile_pairformer_overhead --verify-only --out numerical-check.json
```

Reproduce the full-model benchmark on an 80 GB GPU:

```sh
python -m scripts.profile_pairformer_overhead --out overhead.json
```

## Deployment status

The optimization is in the repository. The four existing training services
still use their immutable `90aca06` snapshots. They were not interrupted for
this investigation; deploying this change requires a checkpointed restart
using the updated source. Inference calculations are unchanged.
