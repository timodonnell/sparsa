# Fixed-R diffusion evaluation

The earlier diffusion oracle scores divided by the number of emitted contacts
when a rollout supplied fewer than R. They were precision among emitted contacts,
not R-precision. In particular, the reported G2-RSC 30k value of 0.446965 is not
valid evidence of improved R-precision. Earlier consensus scores are unaffected.

The corrected scorer (`metric_version: fixed-r-v2`) uses the frozen 97-protein
validation split. For each protein and separation range, R is the number of
true contacts among resolved candidate pairs. Every readout divides true
positives by **R**. Zero-R cases are undefined and excluded from macro averages;
the counts of scored proteins are recorded.

- **Oracle best-of-100:** rank all eligible pairs by each rollout's final denoiser
  contact probabilities, evaluate top R, and take the best rollout per protein.
  Unselected pairs participate in the ranking. This uses probabilities already
  produced by diffusion; it requires no extra model passes.
- **Frequency consensus:** count each pair at most once per sampled map, rank all
  eligible pairs by occurrence count across 100 maps, and score top R. Zero-vote
  pairs remain candidates. This is compared with MarinFold frequency consensus.
- **Mean-probability ensemble:** average final probabilities across rollouts,
  rank all eligible pairs, and score top R. This is a separate diagnostic.
- **Sampled-map oracle:** rank only emitted contacts by their final probabilities
  and divide by R, giving missing ranks zero credit. This isolates the denominator
  correction from the change to dense probability ranking.

Ties use ascending residue-pair order, independent of labels. R and resolution
masks are evaluation inputs; prediction ranking does not inspect contact labels.
The full probability and frequency rankings can support a chosen contact budget
at inference, where ground-truth R is not known.

## Matched MarinFold reference

The pinned reference is exp321 `full_iid_single`: exp277 m2-p06 full-epoch 1.5B,
checkpoint 266344, T=1, top-p=0.95, 100 rollouts on exactly these 97 eval-val
proteins. Sequences, ground truth and resolved residues were checked against
Sparsa's frozen benchmark. Exp321's internally named dev/test cohorts partition
these validation proteins; no Sparsa eval-test or de novo results were used.

| Metric | All contacts (separation >= 6) | Long range (>= 24) |
|---|---:|---:|
| Frequency consensus R-precision | **0.552634** | **0.539416** |
| Validity-gated oracle best-of-100, fixed R | **0.524309** | **0.519901** |

The old 0.5199 all-range reference line was incorrect: that value is long-range
oracle accuracy. Consensus and oracle must use their respective baselines.
MarinFold ranks an individual rollout in emission order; Sparsa's dense oracle
uses model probabilities. The denominator and candidate universe match, but the
readouts differ. The sampled-map diagnostic makes that distinction explicit.

The baseline values are imported from published per-protein tables; local raw
MarinFold rollout files were unavailable. Source hashes and scoring policies are
in `marinfold-iid100-reference.json`; the paired rows are in
`marinfold-iid100-per-protein.csv`. Reproduce the import with:

```bash
uv run python scripts/import_diffusion_reference.py --marinfold-root /path/to/MarinFold
```

## Rollout of the correction

CPU regression tests and a real one-GPU evaluation smoke passed. Corrected
training evaluations use `validation-fixed-r-v2/step-N.*`, leaving old artifacts
intact. Summary files identify the scoring policy and rollout batch size; rollout
rows include R, selected count and true-positive counts for both dense and
sampled rankings. A summary is written only after the detailed rows succeed.

All four active jobs were replaced at batch priority and their durable evaluation
policies now identify `fixed-r-v2`. They resume at G2-wide 260k, SC 150k, G2-RSC
30k, and G3 25k, re-evaluating those checkpoints before continuing the original
300k-step training schedules. G1 completed training; a separate batch-priority
job re-evaluates its 300k checkpoint. At this update the full evaluations are
pending; the one-protein GPU smoke succeeded. Job identities, submission
commands and status snapshots are recorded here and in `reports/jobs`.

Historical consensus curves remain usable;
historical oracle curves require fresh sampling because saved summaries do not
contain the full probability rankings or enough sparse denominators to repair
all scores exactly.

## Current comparisons

![Fixed-R oracle and frequency-consensus comparisons](figures/r_precision_by_step.png)

The oracle panel excludes every legacy score. The consensus panel retains the
valid historical measurements and uses the consensus reference, not the oracle
reference. Recover completed corrected evaluations and regenerate both panels:

```bash
uv run python scripts/fetch_fixed_r_evaluations.py --pod iris-bizon-sparsa-REPLACE-WITH-LIVE-POD
python scripts/plot_fixed_r_progress.py
```

The plotting environment needs matplotlib, pandas and numpy. Corrected results
are stored under `milestones/`; each recovered evaluation is checked against
the frozen protein identities and its per-protein metric means.
