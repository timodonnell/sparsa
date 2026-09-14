# Pair-trunk experiments

Design date: 2026-09-14. Status: proposed experiments; architectures and runner
extensions below are not implemented or launched. All eventual Iris GPU jobs
must use batch priority. This campaign selects on the fixed 97 validation
proteins only. Previously reported test/de novo scores do not guide experiments.

## What the current model actually does

The released 1.815B model contains **2,060,928 pair-block parameters (0.114%)**.
The sequence blocks contain 1,812,676,608 parameters. The pair state has only
96 channels; sixteen convolution/transition blocks contain four triangle
modules. Sequence features enter once, through additive and elementwise-product
projections. Pair reasoning never updates the sequence representation. Every
pair block symmetrizes its output. See [parameter audit](parameter_audit.json)
and [implementation](../../sparsa/model.py).

These observations motivate a bottleneck hypothesis; parameter share alone
does not establish it. Pair operations reuse weights over L² cells and can
dominate compute despite small parameter counts. Triangle modules have global
receptive fields already: the question is the quality and capacity of global
communication, not whether it exists.

The scaling run's validation R went 0.25734 → 0.26320 → 0.26642 at steps
50k/100k/150k, then 0.26621 at 200k. This occurred under cosine LR decay to
10% of peak. It is evidence of diminishing returns under that recipe, not proof
of architectural saturation or proof that longer training cannot help.

The previous architecture search used only 3k steps at crop 256. Attention-map
features and shared refinement each improved mean R by about 0.002, without
passing its promotion rule. Those results justify longer tests, not rejection
of the whole architectural families. See [previous search](../AUTORESEARCH.md).

## Architectural hypotheses and comparisons

All arms return symmetric contact logits from amino-acid tokens alone. Internal
pair states may be directed: z(i,j) need not equal z(j,i). Only the final
Bernoulli prediction must be symmetric. No MSA, PLM weights/features, templates,
sequence retrieval, or teacher contacts as inference inputs are introduced.

| ID | Architecture | Main comparison and interpretation |
|---|---|---|
| C0 | Existing 96-channel, 16-block trunk | Mandatory matched training control |
| C1 | Existing operators, 256 channels, 24 blocks, triangle hidden 128 every fourth block | C1 vs C0: more pair capacity/depth, without a new reasoning mechanism |
| C2 | Existing operators, 128 channels, 16 blocks, triangle hidden 64 every fourth block | Width control for the operator experiments |
| D1 | C2 with directed pair initialization and internal state; symmetric final logits | D1 vs C2: richer orientation information and delayed symmetrization |
| T1 | D1 with separate incoming/outgoing triangle multiplication in every block; pair transition expansion 4; retain local convolution | T1 vs D1: substantially more global pair computation |
| A1 | T1 plus starting-node and ending-node triangle attention every second block, 4 heads × 32 channels | A1 vs T1: content-dependent routing among triples of residues |
| J1 | A1 plus a 256-channel residue state, pair-biased residue attention and residue-to-pair updates every second block | J1 vs A1: iterative exchange between residue and pair representations |
| R1 | Eight J1-style blocks shared across three passes, carrying residue and pair state forward | Later: compare to eight unshared blocks (parameters) and 24 unshared blocks (block applications) |

**C1 is a substantial capacity control:** its pair blocks have 21,883,392
parameters, 10.6× C0. With the screening encoder its total is 475,820,289,
versus 455,648,545 for C0. C2 has 3,657,216 pair-block parameters. Counts for
new operators must be measured after implementation; no memory or speed claims
are inferred from these parameter counts.

### Directed pair state (D1)

Initialize z(i,j) with separate left/right projections, a rank-32 outer product
of projected residue features mapped into 128 channels, and signed relative
position buckets. Contract the outer product into pair channels without storing
a full [B,L,L,32,32] tensor. Preserve directional features between blocks and
symmetrize logits at the end. Padding stays masked after every spatial update.

This first comparison bundles orientation-preserving state and initialization.
If useful, ablate (a) per-block symmetrization restored, (b) the outer product
removed, and (c) signed positions replaced by absolute positions. Those tests
identify which component helps; D1 alone cannot attribute gains to one of them.

### Triangle reasoning and attention (T1, A1)

Use independently parameterized outgoing and incoming multiplication updates,
each with normalization, input/output gates, and residual output initialization.
The current implementation sums the two contractions before one output map;
the proposed block can learn different transformations for the two directions.

For starting-node attention, pair (i,j) attends over (i,k), with a learned bias
from (j,k); use the transposed orientation for ending-node attention. Preserve
directed hidden states. Chunk the fixed-node axis and benchmark the actual
attention kernel: arbitrary triangle bias may prevent a fused attention path.
Never construct a full [B,L,L,L,H] score tensor. Chunking reduces peak memory,
but exact triangle attention still has cubic arithmetic cost in sequence length.

If T1 succeeds, separate operator density from other changes by testing its
new multiplication modules only every fourth block and by restoring transition
expansion 2. If A1 succeeds, compare attention-only and multiplication-only
versions at the same measured GPU budget.

### Joint residue/pair reasoning (J1)

Project the common encoder output into residue state s of width 256. At each
feedback stage, update s with 8-head attention whose logits include a learned
projection of z(i,j), then a residue transition. Feed the updated s back through
a gated rank-32 outer-product projection into z. Run pair reasoning again.
This lets inferred pair patterns change subsequent residue interpretation.

Retain A1 as the no-feedback control. If J1 wins, disable the pair bias while
keeping the residue updater, and separately remove repeated residue-to-pair
updates. These distinguish extra residue capacity from actual feedback.

### Recurrent refinement (R1)

Reuse the same eight joint blocks for exactly three passes during the primary
comparison. Inject normalized initial residue/pair states through learned gates
on each pass; carry predicted states forward. Backpropagate through all passes
with activation checkpointing; supervise only final logits so the primary loss
is unchanged. Compare against an eight-block single pass and a 24-block untied
stack. Run all three with the same training exposure; separately compare at
matched GPU hours. Fixed three-pass inference is the primary endpoint. Any
one/two/four-pass analysis uses validation only and is reported as a separate
inference-compute experiment.

This tests refinement depth and weight sharing, not a coordinate recycling
system. Binary contacts do not obey a triangle inequality: do not impose
transitivity of contacts or fabricate distance labels from contact bits.

## Common training recipe and initialization

Screen with the existing 456M model's 1024-wide, 36-layer sequence encoder.
Copy only embedding, sequence blocks, and final sequence normalization from
the validation-selected supervised Sparsa checkpoint:

`s3://marin-us-east-02a/marin/protein-structure/sparsa/runs/scale-rank456m50-20260911/checkpoints/step-3000.pt`

Use its EMA encoder weights, with a recorded checkpoint checksum. Reset all
pair initialization, relative positions, pair blocks, contact head, optimizer,
and EMA state in **every** arm, including C0. New pair modules start randomly;
train the encoder and pair trunk jointly from the first step. This avoids
penalizing new trunks by comparing them against an already trained decoder.
Also retain the original trained C0 as an inference reference; it is not the
matched reset-trunk control. These are supervised-transfer experiments, not
from-scratch comparisons. Confirm a winning design against C0 from scratch
before making claims about from-scratch learning efficiency.

Freeze these settings before screening:

- Seeds 17 and 37; one shared encoder initialization, separate paired data and
  new-module initialization seeds. Repetitions measure conditional variability,
  not independence of the inherited encoder training history.
- Decontaminated AFDB/ESM data and 50/50 sampling by protein; same corpus hashes.
  Teacher-mixture experiments remain a separate axis.
- Crop 512, global batch 64, same logical records and crop offsets per update.
  Eight H100s per trial. Adjust microbatch/accumulation for memory while retaining
  the logical batch stream. A seed alone is insufficient: current worker-based
  crop sampling changes when microbatch grouping changes.
- Existing per-protein BCE (positive weight 4) + 0.05 ranking KL; weight decay
  0.01, EMA 0.999, bf16. Keep masking, loss normalization and evaluator fixed.
- Nominal 100k-step horizon: 1k warmup to LR 1e-4, constant to step 80k,
  linear decay to 1e-5 at step 100k. Restart neither data nor optimizer when
  extending a screened arm. All arms have the same schedule from step zero.
- Validation every 5k, recovery checkpoints every 1k. Compare final EMA at fixed
  25k and 100k endpoints with original readout; no unequal checkpoint sweeps.
- No long-crop curriculum, auxiliary loss, teacher-mixture change, or readout
  tuning during the primary architecture comparisons.

Use a small, paired LR sensitivity check if both a new trunk and reset C0
fail to recover useful teacher fit. A common but unsuitable LR is not evidence
against a whole architectural family. Report any retuned recipe as a new
comparison applied to both models.

## Stages and decisions

1. **Implementation, correctness and profiling.** Implement C0/C1/C2, D1,
   T1, A1 and J1, then profile forward/backward, AdamW, EMA, DDP, compilation and
   checkpointing. Measure crop 512 training and full-sequence inference at
   256/512/920/1024 residues, including padded mixed-length batches. Use the
   same hardware and record peak allocation, reserved memory, seconds/update,
   parameter count, and inference latency. Fit within 80 GiB H100s. Test one
   real save/resume with optimizer, RNG and logical-data position restored.
2. **Screen seven arms at 25k steps, seed 17.** That is 1.6M crops/arm, over
   16× the previous search exposure. Inspect fixed-endpoint validation R and
   long-range R, per-protein differences and teacher-fit curves. Advance the
   best two feasible designs and C0. Reserve one of those two places for A1
   or J1 as a longer-training hypothesis test even if early R trails; choose
   between them by validation R among stable, feasible implementations.
3. **Confirm at 100k, seeds 17 and 37.** Resume seed 17 on its original schedule;
   train seed 37 from the same stipulated starting state. Apply the same
   exposure to C0. This is 6.4M crops per seed. Rank on mean fixed-endpoint
   validation R; use long-range R as a guardrail. Candidate-specific ablations
   follow only for useful mechanisms. R1 and its matched controls are a later
   experiment after identifying a stable joint block.
4. **Transfer finalists to the 1.8B encoder and longer training.** Repeat the
   encoder-only transfer/reset-trunk comparison for C0 and the best design.
   Compare at matched exposure and inspect the accuracy/compute tradeoff before
   allocating 300k–500k steps. Test crop 1024 as a paired curriculum ablation
   only after selecting the core architecture. Reusing the trained production
   trunk for deployment is a separate continuation experiment.

An architectural improvement target is mean **+0.005 absolute validation R**
over matched C0 at 100k, positive gains on both seeds, and no mean long-range
regression greater than 0.002. The target for a major change is +0.01 or more;
neither number is a predicted outcome. Calculate paired bootstrap intervals
over proteins after averaging seeds; they are descriptive under adaptive
validation reuse. Borderline or seed-inconsistent results receive a third
paired seed before being described as robust. A failure to reach the target
means this implementation/recipe did not justify scale-up, not impossibility.

Record R at equal training exposure **and** equal cumulative running H100-hours.
The latter uses the most recent scheduled validation checkpoint within each
budget, identically for every arm. Report Pareto tradeoffs rather than claiming
that equal steps equal FLOPs. Checkpointing, chunking and recurrent passes can
materially change training and inference cost. A slow but substantially better
architecture can merit further engineering; a tiny gain at many times the cost
does not automatically merit production training.

## Diagnostics that make failures informative

- Freeze a small teacher-fit panel from each training source, disjoint from
  experimental validation/test/de novo records. Measure R, BCE and ranking
  loss at the same checkpoints; it is a training-distribution diagnostic, not
  an independent evaluation. Keep a separate tiny memorization sanity test.
- Log per-module gradient/update norms and gate activity. New zero-initialized
  residual outputs can initially suppress upstream gradients; check that they
  become active over several updates, not just that gradients exist.
- Measure effective pair-channel rank, representation variance and update
  magnitudes at early/middle/late blocks. Treat these as diagnostics, not
  evidence that a specific rank is intrinsically good or a promotion metric.
- Record unique teacher IDs, source counts, actual unpadded residues and positive
  pair counts. Earlier crop totals were sampling exposure, not measured unique
  protein coverage. Report per-source coverage rather than equating document
  tokens, residue positions and dense contact targets.
- Break down validation R by the existing separation ranges and fixed length
  bins (<=256, 257–512, >512). Preserve macro R over all 97 proteins as primary.
  Do not cherry-pick subsets for promotion.
- Verify final-logit symmetry, padding isolation, finite gradients, reload
  equivalence, directed-state survival and chunked/un-chunked attention
  agreement on small tensors. Re-run Helico export/parser checks for the final
  selected model; structure-generation accuracy requires a separate evaluation.

## Compute and runner work

Proposed initial envelope: **1,000 running H100-hours** for profiling, screening
and confirmation, including retries. Planning allocations are 50/350/500/100
hours for profiling/screening/confirmation/recovery reserve. These are proposed
allocations, not a launched budget or a claim all arms will fit. Freeze actual
stage admissions after profiling; if the plan exceeds the envelope, stage the
experiment families and retain matched controls rather than silently shortening
individual architectures. Long 1.8B training and R1 require separately recorded
allocations and are not included in this initial envelope.

At the old 456M throughput (~0.39 s/step on eight H100s), 25k steps cost ~22
H100-hours; 100k costs ~87. This is an anchor estimate only. Full triangle
attention and wider pair tensors can be much slower. Use profiled execution,
validation and checkpoint overhead to forecast the new arms. Do not reuse the
old 1.8B sequence-scaling throughput estimate for pair attention.

The current autoresearch runner cannot execute this protocol unchanged: its
defaults enforce fresh random initialization, <=100M parameters, 3k steps and
the existing cosine schedule. The growth loader also explicitly forbids pair
architecture changes. Required work before launch:

1. Explicit, hash-verified encoder-only initialization and reset rules; source
   snapshots accompanying every candidate and checkpoint.
2. Fixed logical sample/crop IDs independent of microbatch grouping, with exact
   replay through preemption. Same-seed equivalence must be verified, not assumed.
3. Fixed-horizon warmup/constant/decay scheduling and stage continuation;
   checkpoint/EMA/optimizer provenance preserved when a trial advances.
4. Registered architectural families and dependency/ablation relationships;
   larger parameter limits, real profiling and per-stage budget admission.
5. Immutable data/loss/evaluator contracts, per-protein validation metrics,
   actual attempt accounting and a persistent coordinator/compute guard.

No held-out evaluation runs during this search. Freeze the final architecture,
checkpoint and inference policy before a separate held-out report. Since the
held-out sets have previous campaign results, describe this as a new evaluation
of a validation-selected model, not a newly collected independent test set.

## Architectural references

The [AlphaFold implementation](https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/modules.py)
provides the reference for separate incoming/outgoing multiplication and the
two triangle-attention orientations. Its MSA machinery is excluded here.
The [ESMFold trunk](https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/trunk.py)
is a reference for repeated sequence/pair blocks and recycling organization.
We borrow computational patterns; its pretrained language model and coordinate
recycling are not part of these experiments. Neither source establishes that
these changes will improve Sparsa under contact-only supervision.
