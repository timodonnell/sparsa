# Protenix v1 transfer pilot

**Observed 2026-10-06:** launched on 16 H100s at batch priority. Native model
checks and the 16-rank collective passed; training saved at step 2 and resumed
through step 4. The cluster then preempted the job during the frozen-branch
startup test. Iris has queued attempt 1 with zero application failures. The
durable driver resumes the remaining branch checks before continuing warmup.
No validation-quality result is available yet.

The approved pilot transfers the native `protenix_base_default_v1.0.0` backbone
(148,895,680 pretrained parameters), using a new contact/time adapter at its
learned recycling projection and a symmetric binary contact head. One native
48-block pass per denoising call; eight reverse steps per rollout. Pair width
128, single width 384. New head also reads final single features so the final
single-update parameters receive gradients.

| Stage | Optimizer steps | Trainable parameters | Peak learning rates |
|---|---:|---|---|
| Shared warmup | 0–1,000 | Adapter and new head (101,250 parameters) | 2e-4 |
| Full fine-tuning | 1,000–6,000 | Entire retained backbone, adapter and head (148,996,930) | Backbone 1e-5; new layers 2e-4 |
| Frozen control | 1,000–6,000 | Adapter and new head (101,250 parameters) | 2e-4 |

The two branches inherit **the same step-1,000 weights, EMA, optimizer, RNG and
data position**. One 16-H100 Iris gang runs the stages sequentially at **batch
priority**, with automatic checkpoint recovery. Each branch sees 640,000 new
teacher presentations, plus 128,000 shared warmup presentations (not unique
protein counts). Global batch 128; crop 384; per-GPU microbatch 2, logical batch
8, four accumulation rounds; ranking normalization groups span two ranks to
retain the existing 16-protein denominator.

Teacher inventory/mix, eight-step binary refresh diffusion, weighted BCE (4),
ranking KL (0.05), EMA (0.999), contact priors and evaluation seed match the current
campaign. The pilot has a fixed 6,000-step WSD horizon: 100-step learning-rate
ramp, constant rate through step 5,000, then decay to 10%. Both branches use the
same schedule. Pairformer dropout is zero, matching the current experiments.

Full evaluation at warmup and each branch endpoint uses **97 fixed validation
proteins × 100 rollouts**: oracle R-precision and occurrence-count consensus
R-precision with the ground-truth R. Held-out evaluation is not used. Comparison
must include shared warmup cost and stage GPU-hours; original pretraining is
additional data/compute, not free from-scratch training.

## Features and transfer checks

Native input embedding uses amino-acid identity and ideal isolated-residue
chemical features. Profile is query one-hot; deletion mean is zero. MSA,
templates, PLMs, coordinate diffusion and confidence heads are absent. Reference
residues use the official CCD RDKit conformers, centered per residue with native
reference-coordinate augmentation disabled; these are not protein coordinates.
Protenix's standard protein `token_bonds` features are zero. Full sequences are
unpadded before native attention; equal-length proteins/rollouts share batches.

Native source is vendored at commit `85767b811c40ed46e73a9b39519cf6bfca8701ba`,
with only import-path rewriting; torch layer normalization and triangle kernels
are used. See `sparsa/vendor/protenix/SOURCE.json` and its Apache license.
Initialization loads all 2,840 retained tensor keys strictly, at unchanged FP32
precision. [Checkpoint hashes](initialization.json) and
[pretraining overlap audit](overlap_audit.json) are recorded.

Local and H100 GPU checks passed for adapter gradients through the frozen trunk,
full-backbone gradients, exact padding invariance and symmetric predictions. The
H100 full-crop, two-protein fine-tuning microbatch took 8.13 seconds and peaked at
13.0 GB; four complete eight-step rollouts of the longest validation protein
(761 residues) took 119.9 seconds and peaked at 20.3 GB. These are preflight
measurements, not steady-state optimizer-step times. See [evidence](preflight.json).
Before long training, the driver also tests actual DDP checkpoint/resume and
branch/resume data digests and optimizer-state separation. Any failed stage stops
the driver.

The public pre-2021-09-30 training index has no exact PDB-ID overlap with the 97
validation proteins. This is a regenerated public index; it does not prove
sequence/homology or distillation-data disjointness.

## Operations

Install optional dependencies with `uv sync --extra protenix`. The driver is
`scripts/run_protenix_pilot.py`; frozen/full configurations are
`configs/protenix_v1_frozen.yaml` and `configs/protenix_v1_finetune.yaml`.

Output root:
`s3://marin-us-east-02a/marin/protein-structure/sparsa/protenix-v1/pilot-s23`

`status.json` records the active stage; `completed/` records completed stage
runtime; `preflight/` records transfer and resume checks. Each of `warmup/`,
`finetune/`, `frozen/` has independent logs/checkpoints/provenance. The warmup
validation is in `warmup-validation/`; endpoint evaluations are in each branch's
`validation-fixed-r-v2/`. Launch and observed progress are in `active-run.json`.
