# Pairformer conditioning factorial

October 5, 2026. This campaign replaces the G2/G3/G4 series at the user's request.
Previous checkpoints and evaluations are retained; the three active Iris jobs and
both dedicated A100 services were stopped. See `retired-*.json`.

## Four matched arms

| Arm | Pair width | Noise/time enters | Parameters | Hardware |
|---|---:|---|---:|---|
| PF-C128 | 128 | After the 48-block trunk | 157,211,713 | 8 H100, Iris batch |
| PF-C256 | 256 | After the 48-block trunk | 244,826,689 | 8 H100, Iris batch |
| PF-N128 | 128 | Before the first trunk block | 157,211,713 | a100-1: 8 A100 80GB |
| PF-N256 | 256 | Before the first trunk block | 244,826,689 | a100-2: 8 A100 80GB |

Every arm has a four-layer, width-384, six-head bidirectional sequence Transformer
(7.09M parameters), 48 independent Pairformer blocks and four pair-update blocks
in the diffusion head. Every Pairformer block performs outgoing/incoming triangle
multiplication, starting/ending triangle attention and a SwiGLU pair transition,
then updates a width-384 single-residue stream with gated pair-biased attention
and a SwiGLU transition. The final single stream projects into the diffusion head,
so its parameters receive the contact loss. This is an adaptation of the
[AF3 Pairformer design](https://www.nature.com/articles/s41586-024-07487-w), with
an independent PyTorch implementation using Sparsa's existing triangle operators.
It is not AF3's coordinate diffusion model. Pair attention has four heads at both
widths; increasing width also increases head and triangle channel dimensions.

Most parameters now sit inside the interleaved Pairformer, which contains both
pair and single-residue updates; they are not all pair-only parameters. Exact
module counts are in `parameter-counts.json`. There are no convolutions, MSAs,
pretrained features or self-conditioning. The eight-step binary contact diffusion
schedule and corrected fixed-R readouts remain unchanged.

Within a width, C and N have identical parameter names, shapes and initialization.
The sole architectural difference is placement of the noise/time embedding. C
computes the deep trunk once per protein and caches its output across all reverse
steps and rollout batches. N caches only the shallow sequence features and runs
all 48 trunk blocks on every current noisy map at every reverse step. Both use
the same four-block output head. Gradients flow through the full trunk for a
single sampled training timestep; training does not backpropagate through an
eight-step sampling trajectory.

## Training and selection

- Seed 23, crop 384, logical batch 16 per GPU, eight GPUs, global batch 128.
- Full AFDB/ESMFold teacher pool and AFDB sampling probability 0.059522. Dedicated
  nodes use their verified local mirrors; Iris reads the same inventory from S3.
- 300k optimizer steps: 38.4M crop presentations per arm. AdamW lr 2e-4, 15k warmup,
  WSD decay from 240k, EMA 0.999, weighted contact loss plus ranking loss as before.
- Checkpoints every 1k; retain three newest and every 10k milestone. Initial
  checkpoints also cover steps 1, 10 and 100. No held-out set is used.
- Full frozen 97-protein validation, 100 rollouts, every 10k steps. Primary metric
  is dense-probability oracle best@100 R-precision. Frequency consensus is compared
  to MarinFold frequency consensus separately. All denominators use true R.
- Readouts, temperature and rollout seed/batch are matched across the four arms.
  Compare at matched optimizer steps/exposure; report GPU time separately because
  hardware and inference cost differ. One seed per arm is a screen, not evidence
  of seed-robust superiority. Review at 30k and 90k rather than promoting on the
  startup measurements.

## Launch gate and recovery

Each supervised launch first trains through step 2, restarts from its saved state
and advances through step 4 with the original 300k schedule. It verifies eight-rank
data exposure and checkpoint metadata, then runs **all 97 validation proteins
with 100 rollouts**. Identities, readout version, rollouts and per-protein means
must agree before a completion marker is written and long training resumes.
Startup metrics remain under `preflight/`, separate from scientific milestones.

Any failed stage fails the supervised job. A retry resumes the unfinished stage;
partial evaluations without a summary completion marker are repeated. Dedicated
services stop after three starts in six hours. Iris jobs use batch priority.
Validation work is balanced across ranks, with four-hour collective timeout and
per-protein progress logging. Stage status is in `preflight/stage.json`.

## Measured launch profiles

All 96 CPU tests passed, including every-parameter gradient connectivity, exact
seeded cached/uncached sampling, noise entry at the correct block, padding,
activation checkpointing, and rejection of incomplete validation by the gate.
At crop 384 and microbatch 4, single-A100 AdamW/EMA profiles peak at 28.17 GiB
(128 channels) and 55.08 GiB (256 channels). Estimated global-batch-128 optimizer
steps are 64.0 and 104.9 seconds before DDP/data overhead. These are substantially
slower than G4; on A100 the initial estimates put 10k steps at about 7.4 / 12.1
days before evaluation. The 300k horizon is a maximum schedule, with validation
reviews before committing the entire horizon. H100 throughput is measured after
launch rather than inferred from A100 results.
