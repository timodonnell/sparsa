# Implementation and training plan

The pre-existing `PLAN.md` is a research roadmap for contact completion. This
initial implementation prioritizes the user's sequence-only contact prediction
objective, with a strong one-pass model before iterative generative extensions.

1. Freeze MarinFold source revisions, contact semantics, teacher shard inventory,
   experimental targets, resolved masks, baseline results, and evaluation code.
2. Decode the existing decontaminated AFDB/ESM-Atlas contacts-v1 documents into
   sequence and sparse binary targets. Preserve ring-index mappings; reject
   truncated or malformed records. Use a 50/50 source mixture as in MarinFold.
3. Implement a randomly initialized bidirectional sequence encoder plus explicit
   pair reasoning (relative positions, outer-product features, dilated pair
   convolutions and triangle multiplication). No PLMs, MSA, templates, or
   inference-time evolutionary features.
4. Test decoding, masking, symmetry, loss gradients, checkpoint restoration,
   upstream metric parity, and Helico export; run an Iris H100 pilot at batch
   priority to measure throughput and establish learnability.
5. Train controlled architecture variants; use only experimental eval-val for
   selection. Save resumable optimizer/RNG checkpoints and compute/data budgets.
   Expand the best variant after inspecting validation trajectories.
   Follow the 100,000-step crop-384 run with a 3,000-step crop-1024 finetune
   (global batch 32, low learning rate, activation checkpointing). Select the
   checkpoint and readout across both phases using validation only, including
   distance capping and a uniform EMA weight average of each phase's two best
   original checkpoints. A validation-only averaging preview showed a small gain.
6. Evaluate the selected checkpoint on all fixed experimental splits; publish
   per-protein scores, timing, paired comparisons, limitations, reproducible
   commands, and a loadable checkpoint plus Helico contact export.

All accelerator jobs are submitted directly at Iris `--priority batch`, avoiding
implicit child jobs whose priority might default to interactive. Training data
and durable artifacts stay on CoreWeave S3 alongside compute. The implemented
architecture-search loop uses the same data and scoring contracts, with fixed
short training budgets and two-seed confirmation. Completion requires actual
training and measured evaluation results.

## Execution status (2026-09-11 17:05 UTC)

Stages 1–4 and the main 100,000-step training run are complete. Stage 5 continues
with a 3,000-step crop-1024 finetune on eight batch H100s, initialized from the
main validation best at step 96000. Held-out evaluation and final artifacts
(stage 6) remain pending. The separate architecture search is confirming a
shared-weight refinement candidate. See
`reports/STATUS.md` and the machine-readable pilot reports for exact evidence.
