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
6. Evaluate the selected checkpoint on all fixed experimental splits; publish
   per-protein scores, timing, paired comparisons, limitations, reproducible
   commands, and a loadable checkpoint plus Helico contact export.

All accelerator jobs are submitted directly at Iris `--priority batch`, avoiding
implicit child jobs whose priority might default to interactive. Training data
and durable artifacts stay on CoreWeave S3 alongside compute. The eventual
autoresearch extension can use the same config, validation, and run artifact
contracts. Completion requires actual training and measured evaluation results.

## Execution status (2026-09-11 05:32 UTC)

Stages 1–4 are complete. The selected 40M network is queued to resume on 8 H100s at batch
priority after preemption, using the verified decontaminated teacher corpus. Stage 5 is active;
held-out evaluation and final artifacts (stage 6) remain pending. See
`reports/STATUS.md` and the machine-readable pilot reports for exact evidence.
