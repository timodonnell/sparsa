# Dedicated A100 scaling experiments

Launch plan, October 2, 2026 UTC. Status and measured launch evidence are in
[RUNNING.md](RUNNING.md). Architecture and checkpoint decisions use only the
frozen 97-protein validation split; held-out sets are excluded.

## Question and choice

Can a substantially larger sequence encoder plus deeper global pair reasoning
improve fixed-R oracle best-of-100? Does sharing the pair block across repeated
updates help or hurt at approximately matched parameters and computation?

The corrected results currently favor plain G2-wide: at 260k it achieves
0.353881 probability oracle and 0.369172 frequency consensus. G2-RSC at 30k
achieves 0.255612 / 0.231885; G3-balanced-RSC at 25k achieves 0.271135 / 0.246599.
These are unequal training exposures, not evidence that RSC cannot work, but
they no longer justify preferentially scaling RSC. The formerly reported
RSC oracle advantage used an incorrect denominator. We therefore scale the
established diffusion objective without self-conditioning in both new arms.

| | G4-U4 | G4-L4 |
|---|---:|---:|
| Parameters | 315,297,028 | 309,359,620 |
| Sequence encoder | 24 layers, width 1024, 16 heads | identical |
| Pair / triangle width | 256 / 256 | identical |
| Pair attention | 8 heads; chunk 32 | identical |
| Global pair updates per denoising call | 4 independent blocks | 1 block repeated 4 times |
| Reverse diffusion steps | 8 | 8 |
| Hardware | a100-1: 8 A100-SXM4-80GB | a100-2: 8 A100-SXM4-80GB |

Both are initialized from scratch, with sequence as the only conditioning
input. There are no convolutions, MSAs or pretrained language model features.
Each pair update contains outgoing/incoming triangle multiplication and
starting/ending triangle attention, followed by a transition. Four pair
updates occur **inside** one denoising call. Training samples one noise time
per case and differentiates through these four updates; it does not unroll
all eight reverse diffusion steps during training. L4 retains gradients through
all four applications of its shared block.

The sequence encoder accounts for most parameters; this experiment scales
sequence capacity and pair width/depth together relative to earlier models.
U4 versus L4 isolates parameter sharing at the same update count, with a 1.9%
parameter difference. Comparisons to G2/G3 are scaling experiments, not a
single-variable architecture ablation. One seed per arm does not establish
seed-robust superiority.

Triangle operations are motivated by [AlphaFold2's pair reasoning](https://www.nature.com/articles/s41586-021-03819-2).
The shared-depth alternative is also motivated by [recurrent-depth transformer research](https://arxiv.org/abs/2502.05171),
which is not evidence of improved protein contacts. Here L4 uses a fixed four
iterations in both training and sampling; variable inference depth is not tested.

## Training protocol

- Full locally mirrored exp232 decontaminated teacher pool: 5,405 Parquet files,
  139,371,683,675 bytes. AFDB and ESMFold sources are sampled at protein level
  with AFDB probability 0.059522, matching the long-run recipe. File and row
  counts do not imply unique proteins; training draws may repeat.
- Crop 384; eight GPUs; microbatch 8; two gradient accumulation microbatches:
  **global batch 128**, matched across arms. Seed 23. No shard limit.
- 300,000 optimizer steps, or **38.4 million protein-crop presentations** per
  arm. This is a target horizon, not a claim of completed exposure or a
  coverage-guaranteed pass through the teacher pool.
- AdamW, learning rate 2e-4, weight decay 0.01, gradient clip 1, bf16,
  activation checkpointing, EMA 0.999. WSD: 15k warmup, decay starts 240k,
  fixed horizon 300k. Positive-class weight 4; ranking weight 0.05.
- Supervised user services, restart on failure, exact-state automatic resume,
  immutable source snapshots. These dedicated nodes do not use Iris scheduling;
  existing Iris jobs remain at batch priority.
- Atomic local checkpoints every 1k; retain the latest three and every 10k
  milestone. Additional launch checks save steps 10 and 100. Recovery includes
  optimizer, EMA, per-rank RNG state and data cursors. Each node trains one arm
  independently; no cross-node collective communication.

## Evaluation and decisions

Evaluate EMA checkpoints every **10,000 steps**, with 100 eight-step rollouts
per protein, on all 97 validation proteins. Use metric version `fixed-r-v2`,
rollout batch 1, seed 20260925 and temperature 1. Smaller rollout batches alter
random-number assignment relative to older runs, so seeds do not guarantee
identical sample banks across campaigns.

Primary metric: oracle best-of-100 R-precision using each rollout's final
denoiser probabilities to rank all eligible pairs, with ground-truth R as both
the top-k count and denominator. Report frequency consensus separately and
compare it with **MarinFold frequency consensus**, not its oracle reference.
Also retain mean-probability and sampled-map diagnostics and long-range scores.

Matched MarinFold validation references: 0.524309 validity-gated oracle and
0.552634 frequency consensus (all contacts, separation >= 6). MarinFold's
individual rollouts rank contacts by emission order; Sparsa's primary oracle
uses dense probabilities. This readout difference remains explicit. See the
[scoring policy and baseline provenance](../diffusion_fixed_r/README.md).

Review paired per-protein differences at 30k and 90k, then continue the promising
training trajectories. Compare matched steps and example counts between U4/L4,
and record GPU-hours. Early warmup scores are not grounds for declaring scaling
a failure. No arm is automatically promoted on a single checkpoint. Stop or
repair divergence, NaNs or data faults promptly. Before expanding a winning
architecture further, confirm the improvement with an independent training seed.

## Measured feasibility

Single-A100 synthetic training profiles include forward/backward, clipping,
AdamW and EMA, but exclude DDP, data loading, checkpointing and evaluation:

| Shape | U4 estimated seconds / optimizer step | L4 | Peak allocated GiB, U4 / L4 |
|---|---:|---:|---:|
| Crop 384, microbatch 4 | 9.93 | 9.91 | 28.41 / 28.32 |
| Crop 384, microbatch 8 | 9.73 | 9.73 | 52.08 / 51.99 |
| Crop 512, microbatch 4 | 20.78 | 20.77 | 46.70 / 46.61 |
| Crop 512, microbatch 8 | OOM | OOM | — |

We choose crop 384 / microbatch 8 for throughput and compatibility with prior
training exposure. Synthetic timing implies about 27 hours per 10k steps and
34 days for 300k, before distributed and evaluation overhead. Real eight-GPU
measurements supersede these estimates. This is a substantial long run; the
dedicated allocation avoids the frequent preemptions affecting the Iris jobs.

Configs: [U4](../../configs/diffusion_g4_u4_a100.yaml),
[L4](../../configs/diffusion_g4_l4_a100.yaml). The generic
[launcher](../../scripts/launch_dedicated_diffusion.py) runs on the destination
node inside its frozen source snapshot. Infrastructure connection details are
managed separately from the repository.
