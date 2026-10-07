# Pairformer campaign status

## October 7: 13:38–13:41 UTC update

All five from-scratch runs are training. Latest logged steps and recent training
speeds are below; durable checkpoints are saved every 100 steps.

| Run | Latest logged step | Hardware | Recent seconds/step |
|---|---:|---|---:|
| N128-32GPU | 9,400 | 32 H100 | 8.51 |
| N128 | 2,800 | 8 dedicated A100 | 55.23 |
| N256 | 1,620 | 8 dedicated A100 | 92.38 |
| C128 | 2,680 | 8 H100 | 33.43 |
| C256 | 950 | 8 H100 | 51.06 |

N128-32GPU has accumulated 1,203,200 teacher crop presentations including its
inherited parent training. Its first 97-protein, 100-rollout validation starts
at step 10,000, approximately 1.5 training hours away if uninterrupted. None of
these five runs has a completed scientific validation checkpoint yet.

Cluster interruptions dominate the cached runs' wall-clock progress: Iris
records 23 preemptions for C128 and 24 for C256. C256 also has three recorded
failures; both current attempts are advancing without errors in their recent
logs. The dedicated A100 services have zero restarts.

The [Protenix transfer pilot](../protenix_transfer/RUNNING.md) has completed
warmup validation and reached full-fine-tuning step 3,070 on 16 H100s. Its frozen
warmup result is weak: oracle R-precision@100 0.093689 and frequency consensus
0.094307. Full-fine-tuning quality has not yet been evaluated.

Current measurements: [latest status](latest-status.json).

## October 6: 17:17 UTC update

All five from-scratch Pairformer runs are actively training. These are latest
**logged** steps; checkpoints are saved every 100 steps.

| Run | Latest logged step | Hardware | Recent seconds/step |
|---|---:|---|---:|
| N128-32GPU | 2,600 | 32 H100 | 8.42 |
| N128 | 1,470 | 8 dedicated A100 | 55.20 |
| N256 | 830 | 8 dedicated A100 | 92.28 |
| C128 | 1,630 | 8 H100 | 33.48 |
| C256 | 920 | 8 H100 | 51.06 |

No run has reached its first scheduled 10k-step scientific evaluation. The
[Protenix transfer pilot](../protenix_transfer/RUNNING.md) is also training: step
460 of its 1,000-step adapter/head warmup on 16 H100s. Its real branch/resume gates
passed after automatic recovery from one cluster preemption. First transfer
validation is at step 1,000; no transfer accuracy result is available yet.

Current measurements: [latest status](latest-status.json).

## October 6: 32-GPU N128 comparison

The **32-H100 N128 clone is training**, as of October 6, 14:00 UTC:
`/bizon/sparsa-pairformer-v1-pf-n128-32gpu-s23-300k-r1`, at Iris **batch priority**.
It passed a real 32-rank NCCL collective, trained/saved at step 1,202, resumed
through 1,204, and entered long training. Latest observed step: **1,220**.
The most recent ten-step segment measured **8.49 s/step**, compared with
**55.23 s/step** for the original 8-A100 run. This is an early measurement;
the hardware change contributes to the difference.

The clone branches N128 at step **1,200** (153,600 historical crop exposures),
preserving model, EMA, AdamW state and the 300k learning-rate schedule. The
archived checkpoint was verified by SHA-256. The original N128 service is
confirmed active. The two cached Iris runs were preempted and awaiting capacity
at the latest cluster check; their automatic resume remains configured.

| | Original N128 | N128-32GPU |
|---|---:|---:|
| Hardware | 8 A100 80GB, dedicated | 32 H100 80GB, Iris batch |
| Nodes | 1 | 4 |
| Microbatch per GPU | 4 | 4 |
| Accumulation | 4 | 1 |
| Global batch | 128 | 128 |
| LR / schedule | 2e-4 / 300k WSD | unchanged |

The new world size uses fresh, explicitly seeded data/noise streams. It is a
stochastic branch, not an exact RNG continuation. Ranking normalization still
uses groups of 16 proteins, including zero-contact crops. CPU checks verify
equivalent loss and gradients when a fixed batch is partitioned across 32 ranks.
The hardware change also affects throughput, so any speedup against the A100
baseline is not a pure GPU-count scaling result.

The completed startup gate verified **154,112 total crop exposures** at step
1,204: 153,600 inherited plus 512 new, with all 3,033 optimizer parameter states
restored. It checks all-rank data exposure before entering long training. The branch retains the fixed 97-protein validation split, 100
rollouts and evaluations every 10k optimizer steps. Checkpoints remain every
100 steps; recovery preserves 32-rank RNG and data cursors.

Evidence: [parent checkpoint](n128_32gpu_parent.json),
[submission](../jobs/pairformer-v1-pf-n128-32gpu-s23-300k-r1.json), and
[active run manifest](active-runs.json). Implementation source: `0f83221`.
**35 relevant CPU tests passed**, including loss partitioning, fork guardrails,
exposure accounting and concurrent/interrupted checkpoint-cache downloads.
The GPU startup gate also passed. See the [resume audit](n128_32gpu_preflight.json)
and [live launch evidence](n128_32gpu_launch.json).

The [diffusion specification](diffusion_spec.md) documents the exact noise,
loss, reverse sampler and scoring. The [Protenix assessment](../protenix_transfer/feasibility.md)
recommends a v1 transfer pilot; no transfer training result is claimed.

## Original launch evidence (October 5)

As of **October 5, 2026, 16:32 UTC**, all four new jobs are running. Each has
completed four optimizer steps, including checkpoint resume. All are now in the
full **97 validation proteins × 100 rollouts** launch check. That check has not
yet completed; the supervised entrypoint starts long training automatically only
after it passes. Startup scores are operational checks, not architecture evidence.

| Arm | Trunk input | Pair width | Parameters | Hardware | State |
|---|---|---:|---:|---|---|
| PF-C128 | Sequence features; cached across rollouts | 128 | 157.2M | 8 H100, Iris batch | Full preflight validation |
| PF-C256 | Sequence features; cached across rollouts | 256 | 244.8M | 8 H100, Iris batch | Full preflight validation |
| PF-N128 | Sequence features + current noisy map/time | 128 | 157.2M | a100-1, 8 A100 80GB | Full preflight validation |
| PF-N256 | Sequence features + current noisy map/time | 256 | 244.8M | a100-2, 8 A100 80GB | Full preflight validation |

All use the same 4-layer, width-384 sequence encoder, 48 independent Pairformer
blocks, four-block diffusion head and eight-step binary diffusion process. The
[plan](PLAN.md) records the complete comparison. The prior G2/G3/G4 series is
retired: only the two new Iris jobs and two new dedicated services are active.

## Verification and runtime

- **97 CPU tests passed.** Gradient connectivity, cached sampling equivalence,
  noise entry, activation checkpointing and validation gate rejection are covered.
- All four production models restored a checkpoint and advanced to step 4 with
  finite losses and gradient norms. Eight-rank teacher counts and data digests
  agree across all four arms, despite using S3 versus local mirrors. Each has
  consumed 512 crops from the same full 5,405-file teacher inventory.
- Both widths completed four full rollouts on the longest validation protein
  (761 residues), in both cached and noisy modes, with finite probabilities.
  Noisy-mode peak inference allocation was 13.36 / 26.27 GiB at rollout batch 4.
- Full distributed validation is in progress. No post-relaunch errors or automatic
  restarts were observed. The cached-128 job has completed multiple proteins.
- Initial source was `11373b7`; current launch source is `90aca06`. The second
  launch preserves model/optimizer/data state and changes only recovery saves
  from every 1,000 to **100 steps**. At measured throughput, the earlier cadence
  would lose too much work to Iris preemptions. Original provenance is retained;
  runtime records the active code and checkpoint cadence.

A100 resumed training measured **64.6 s/step (128 channels)** and **105.2 s/step
(256 channels)**. Initial H100 measurements were about 40 s / 72 s per step
(the latter includes initial-save overhead). These deep models are substantially
slower than G4. A100 step 10k is roughly 7.5 / 12.2 days before validation overhead;
Iris preemptions make calendar estimates less reliable. The configured horizon
is 300k steps, with fixed-validation reviews at 30k and 90k. Regular evaluation
remains every **10k steps**, using corrected oracle best@100 and separate
frequency-consensus comparison to MarinFold. No held-out data is used.

## Inspect and recover results

`active-runs.json` contains current job/service identities and output locations.
Dedicated services are `sparsa-pf-n128-s23` and `sparsa-pf-n256-s23`; their stage
is recorded in `<output>/preflight/stage.json`. A completed launch gate writes
`<output>/preflight/complete.json`; scientific evaluations later appear under
`<output>/validation-fixed-r-v2/`.

Recover future scientific evaluations and make plots for this campaign alone:

```bash
uv run python scripts/fetch_fixed_r_evaluations.py \
  --pod '<active Sparsa pod>' --manifest reports/pairformer_v1/active-runs.json
uv run python scripts/fetch_fixed_r_evaluations.py \
  --host '<SSH destination for a100-1>' --model pf-n128 \
  --run-dir /home/ubuntu/sparsa-runs/pf-n128-s23-300k
uv run python scripts/fetch_fixed_r_evaluations.py \
  --host '<SSH destination for a100-2>' --model pf-n256 \
  --run-dir /home/ubuntu/sparsa-runs/pf-n256-s23-300k
python scripts/plot_fixed_r_progress.py \
  --models PF-C128 PF-C256 PF-N128 PF-N256 --out reports/pairformer_v1/figures
```

Plotting correctly refuses an empty scientific series during launch validation.
Evidence: `status.json`, `iris-status.json`, `launch-verification.json`,
`pf-n*-launch.json`, `sampling-a100-*.json`, `profile-a100-*.json` and the Iris
submission records in `reports/jobs/`. Connection addresses are omitted.
