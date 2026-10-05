# Pairformer campaign status

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
