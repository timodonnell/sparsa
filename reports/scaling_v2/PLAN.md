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
0.423 s/microbatch, versus 0.621 s with activation checkpointing. The actual eight-GPU DDP smoke subsequently passed with this faster mode.
Batch 8 / crop 512 and batch 2 / crop 1024 OOM in this mode; do not use them.

Verification: the latest targeted suite has 30 passing local tests, including growth prediction equivalence,
new-layer/channel gradients, vocabulary and architecture rejection, teacher
mixture endpoints, exact worker-cursor replay, existing model/decoder/export
contracts, checkpoint averaging, and job accounting. The real GPU growth and
profiling checks passed; pilot loss and gradient telemetry are finite.

## Experiments

Eight continuation pilots start from the identical released EMA weights:

| Pilot | Parameters | AFDB protein probability | Purpose |
|---|---:|---:|---|
| scale-control50-20260911 | 39,994,657 | 0.5 | Longer-training control |
| scale-control-prop-20260911 | 39,994,657 | 0.0570052468 | Teacher-mixture control |
| scale-grown456m-20260911 | 455,648,545 | 0.0570052468 | Capacity expansion |
| scale-grown456m50-20260911 | 455,648,545 | 0.5 | Capacity × mixture interaction |
| scale-rank40m50-20260911 | 39,994,657 | 0.5 | BCE + 0.05 contact-ranking KL |
| scale-rank456m50-20260911 | 455,648,545 | 0.5 | Capacity with ranking loss |
| scale-grown18b50-20260911 | 1,815,192,865 | 0.5 | Parameter scale comparable to MarinFold |
| scale-grown456m50-lr4-20260911 | 455,648,545 | 0.5 | LR 4e-4, the original main-run peak |

All use seed 217, crop 512, global batch 64, 6,000 steps (384,000 additional
crops), LR 1e-4 (4e-4 in the LR pilot) with 300-step warmup/cosine decay, EMA 0.999, and validation every
1,000 steps. Each size comparison within a mixture shares the deterministic data stream.
Each uses eight H100s and saves recovery checkpoints every 500 steps (1,000 for
1.8B). The 1.8B arm uses batch 2 × accumulation 4 instead of batch 4 × 2, so its
worker grouping/data stream differs while total crop count and distribution
match. This is
matched training exposure, not matched FLOPs or training time.

A separate 200-protein training-corpus diagnostic checks fit to each teacher
source. Those proteins may have been seen in training; its scores are diagnostic,
not held-out generalization estimates. Identity and score tables are retained.

After inspecting these curves, choose the mixture and run a larger model for
substantially longer, initially targeting 200,000 additional steps. Use the verified compiled pair blocks and non-checkpointed main-training
execution for that run. If growth impairs learning, test a freshly initialized wider model;
do not dismiss scaling solely from a short continuation pilot.

The initial pilots should use tens of H100-hours. Bound the subsequent campaign
to approximately 600 H100-hours initially, including retries; reassess using
validation and measured throughput rather than automatically consuming it.
Keep resumable state and actual attempt-time accounting. A local
`scripts/guard_compute.py` process monitors the explicit job list in
`outputs/scaling-v2/budget.json`; it includes retries and requests cancellation
of active GPU jobs at 600 hours. It keeps watching between phases. Unknown
stopped-attempt durations are conservatively bounded by whole job lifetime for
enforcement. This is a polling guard, not a billing guarantee. Large production
checkpoints will retain all validated steps and roll recovery-only saves to the
latest two, reducing storage while preserving resumability. Preserve the original
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

## Measured execution improvements and additional hypotheses

Compiled pair-block forwards preserve ordinary checkpoint keys and leave EMA
inference eager. GPU checks cover outputs, gradients, masks and active triangle
updates. The initial DDP smoke hit Dynamo's default recompilation limit; raising
the per-code cache limit to 128 fixed that fallback. A second 200-step, eight-H100
smoke settled at 0.38–0.39 s/optimizer step, global batch 64, peak 55.85 GiB.
The four original pilots were stopped and fully resumed with this execution
change; optimizer, RNG, worker cursors, objective and LR schedule are retained.
Resume provenance records the change. Compiled arithmetic can alter rounding.
The ranking pilots use compilation from their start.

The 1.8B sequence encoder (2048 × 36, 32 heads, unchanged pair trunk) also passes
the actual growth probe: validation R 0.247789, FP32 max logit error 0.001391.
These small differences are numerical, not learned gains. A compiled synthetic
batch 2 / crop 512 takes 0.179 s/microbatch and peaks at 54.28 GiB. Its eight-GPU
pilot must establish real throughput before choosing a long-run length. Host
RAM is requested at 640 GiB for loading full eight-rank optimizer checkpoints.

The auxiliary ranking loss is the KL from a uniform distribution over true
contacts to softmax logits over eligible pairs. It normalizes positive learning
by the number of contacts instead of L²; zero-contact crops retain BCE alone.
This is a training-only use of true contact count, not a truth-based export
budget. Test its validation effect before selecting it for extended training.

The extended run remains a pending decision, targeting about 200,000 additional
steps in the best measured larger recipe, subject to measured throughput and
the 600-H100-hour campaign guard. No new test/de novo evaluation has occurred.

[PyTorch compilation documentation](https://docs.pytorch.org/docs/2.10/generated/torch.compile.html)
explains dynamic-shape compilation and fallback after the recompilation limit.

## Autonomous handoff

`scripts/scaling_campaign.py --manifest configs/scaling_campaign.json` waits for
all eight completed pilots, recomputes their best/final EMA validation scores and
paired intervals, then freezes its decision in `reports/scaling_v2/selection.json`.
The highest validation R chooses the starting EMA and recipe; the original release
remains a fallback if no pilot improves it. The target is 1.8B parameters and up
to 200,000 new steps (12.8M crops). Measured full-pilot elapsed time, a 10% margin,
and a 60-H100-hour reserve determine the exact affordable step count. If 1.8B
cannot support at least 100,000 steps, it uses 456M and a compatible source.
This is a compute-constrained decision, not a claim that size already helps.

The generated main config uses crop 512, global batch 64, the chosen pilot LR,
and 2,000 warmup steps. A subsequent 5,000-step phase uses crop 1024, global batch
32 and LR at most 5e-5. Existing long-distance embeddings are retained: the source
40M release already had 1024-residue finetuning. Synthetic 1.8B long-crop profiling
passed at batch 1 with activation checkpointing: 0.429 s/microbatch, 40.97 GiB.
Final validation-only readout selection includes the original completed baseline;
then one held-out evaluation, checksummed artifact recovery, and actual CLI/Helico
parser verification run through `scripts/finish_run.py`.

The local handoff and compute guard require this workstation to remain running.
Their journals are restartable; Iris training independently retries from durable
checkpoints. Terminal failures stop the handoff for inspection. Unit tests cover
budget bounds, capacity fallback without checkpoint shrinking, and rejection of
held-out recipe selection. Downstream phases are pending until their jobs finish.


Additional checks: on validation chains of at most 256 residues, the previous
release scores 0.276413 versus MarinFold's 0.492856 (51 proteins). The gap is not
limited to contacts beyond training crop length; the length-stratified table is
retained in `validation_gap_by_length.csv`.

Compiling sequence blocks as well as pair blocks improves synthetic 1.8B timing
from 0.179 to 0.163 s/microbatch. Output/gradient checks pass, but the first DDP
resume probe encountered a Dynamo graph-partition compiler error after loading
the full checkpoint. Its retries were stopped. Sequence compilation remains
optional and is not selected for production until a real DDP test passes.
Full-state eight-rank loading peaked at about 598 GiB host RAM; extended 1.8B jobs
will request 768 GiB for headroom. This does not change GPU priority or count.


The extended run retains the established pair-only compilation. Disabling
Dynamo's DDP graph optimizer allowed the optional sequence-compiled probe to
train 200 resumed steps at about 0.51 s/step, but its checkpoint collective
failed with an NCCL error. This is not accepted as a successful production
configuration. Its logs and failed outcome are retained. Production checkpoints
now release completed gradients and unused CUDA cache before validation and
checkpoint collectives. A separate pair-only full-state resume/checkpoint probe
must pass before the handoff can launch extended training.

Final selection includes the chosen pilot run as well as main training,
long-crop finetuning and the original release, preserving a successful pilot if
extended training fails to improve it. The automated final report includes paired
comparisons against the previous release; completed results are pushed to main
when the branch and staging index permit publishing campaign-owned outputs.


The production preflight passed: the 1.8B checkpoint restored model, EMA,
optimizer, RNG and worker cursors on eight H100s, trained 40 further steps,
evaluated validation, and durably saved step 1040. It is an execution test with a
shortened schedule, not a selection candidate. The 30-test local suite passes.
The completed-control GPU comparison also passed: seven model/readout entries,
42 paired comparisons, and exact 97-protein scoring-universe agreement. Its
final 40M 50/50 gain is +0.001493 R, paired interval [-0.001108, +0.004097];
these descriptive intervals do not establish a substantial improvement.

The restartable coordinator and budget guard are running locally. Remaining
pilots and the extended run are still pending; this report does not claim a
new held-out result or a closed MarinFold gap. After completion, the pipeline
writes and publishes `reports/scaling_v2/FINAL.md` and the verified model artifacts.
