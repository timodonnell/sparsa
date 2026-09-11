# Scaling campaign, 2026-09-11

Objective: materially improve validation R-precision over the released 40M model
(0.247716), using more capacity and training while preserving sequence-only
inputs and MarinFold's fixed evaluation contract. No held-out scores guide this
campaign. All GPU jobs run at Iris batch priority; push code directly to main.

## Evidence and hypotheses

- The original main run sampled 6.4M crops. MarinFold exp232 uses a 1.5B model
  and 152.25B document tokens; those include sequence and contact statements, so
  document tokens are not interchangeable with Sparsa residues or crops.
- MarinFold's strongest pinned reference uses its proportional m2 mixture,
  about 94% ESM tokens. Sparsa used 50/50 protein sampling. Test 5.7005% AFDB /
  94.2995% ESM by protein, proportional to eligible teacher row counts. This is
  analogous to m2, not a claim of identical packed-token sampling.
- GPU profiling finds 176M and 409M models have similar cost when the pair
  network is fixed: 1.105 and 1.133 s per batch of eight at crop 384. Increasing
  sequence capacity is comparatively cheap in this implementation.
- A 456M encoder can inherit the trained 40M model: double sequence width, keep
  head dimension 64, and triple depth with identity-initialized residual blocks.
  Preserve the pair trunk. Antisymmetric projection perturbations preserve the
  initial function on duplicate channels while breaking gradient symmetry.
  This uses only our supervised contact weights, not a pretrained PLM.
- The actual growth probe passes floating-point tolerance and starts at
  validation R 0.247906 versus 0.247716 for the source under bf16 evaluation.
  This tiny numerical difference is not a learned improvement. FP32 maximum
  logit error on the short validation chain is 0.001216.

Further diagnostics: the old model scores 0.244667 against 100 sampled AFDB
teachers and 0.334542 against 100 ESM teachers. These are different proteins and
may overlap past training, so this is not a controlled domain-gap estimate. It
does show that teacher-contact reconstruction is still far from saturated.

The 456M non-checkpointed profile fits batch 4 / crop 512 in 54.75 GiB and takes
0.423 s/microbatch, versus 0.621 s with activation checkpointing. The planned
long run can use this faster mode, subject to an actual DDP memory check.
Batch 8 / crop 512 and batch 2 / crop 1024 OOM in this mode; do not use them.

Verification: 26 local tests passed, including growth prediction equivalence,
new-layer/channel gradients, vocabulary and architecture rejection, teacher
mixture endpoints, exact worker-cursor replay, existing model/decoder/export
contracts, checkpoint averaging, and job accounting. The real GPU growth and
profiling checks passed; pilot loss and gradient telemetry are finite.

## Experiments

Three continuation pilots start from the identical released EMA weights:

| Pilot | Parameters | AFDB protein probability | Purpose |
|---|---:|---:|---|
| scale-control50-20260911 | 39,994,657 | 0.5 | Longer-training control |
| scale-control-prop-20260911 | 39,994,657 | 0.0570052468 | Teacher-mixture control |
| scale-grown456m-20260911 | 455,648,545 | 0.0570052468 | Capacity expansion |

All use seed 217, crop 512, global batch 64, 6,000 steps (384,000 additional
crops), LR 1e-4 with 300-step warmup/cosine decay, EMA 0.999, and validation every
1,000 steps. The two proportional arms share the deterministic data stream.
Each uses eight H100s and saves recovery checkpoints every 500 steps. This is
matched training exposure, not matched FLOPs or training time.

A separate 200-protein training-corpus diagnostic checks fit to each teacher
source. Those proteins may have been seen in training; its scores are diagnostic,
not held-out generalization estimates. Identity and score tables are retained.

After inspecting these curves, choose the mixture and run a larger model for
substantially longer, initially targeting 200,000 additional steps. Profile
non-checkpointed execution to reduce activation-recomputation overhead before
that run. If growth impairs learning, test a freshly initialized wider model;
do not dismiss scaling solely from a short continuation pilot.

The initial pilots should use tens of H100-hours. Bound the subsequent campaign
to approximately 600 H100-hours initially, including retries; reassess using
validation and measured throughput rather than automatically consuming it.
Keep resumable state and actual attempt-time accounting. Preserve the original
release and its held-out results as a fixed reference. Final evaluation and
Helico export follow only after model selection.

## Sources

MarinFold contracts were read from the pinned local checkout's
`experiments/exp232_sweep_cv1_decontam/{README.md,training_contract.py}`.
[AlphaFold 2](https://www.nature.com/articles/s41586-021-03819-2) motivates
explicit pair reasoning, but uses MSA features that remain excluded here.
[ESMFold](https://pubmed.ncbi.nlm.nih.gov/36927031/) reports improvements with
language-model scale; that does not establish scaling behavior for this
supervised-only contact model. This campaign must measure it directly.
