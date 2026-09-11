# Sparsa: first completed campaign

Sparsa predicts a dense residue-contact matrix from amino-acid sequence alone.
The 39,994,657-parameter model completed training and experimental evaluation,
but remains substantially less accurate than the pinned MarinFold reference.
The architecture-search system is implemented and completed its first bounded
campaign; no candidate met the promotion rule.

## Experimental results

Macro R-precision uses MarinFold's frozen exp245 splits and unmodified exp89
scorer. Targets are native-amino-acid ConFind degree >= 0.001, with sequence
separation >= 6 and the same resolved-residue candidate masks. R is the number
of true contacts in the scoring universe; it is used for evaluation, not for
choosing how many contacts to export at inference.

Primary reference: `marinfold-exp232-decontam-m2-p06-step145199`.

| Split | N | Sparsa R | MarinFold R | Paired difference (95% bootstrap interval) |
|---|---:|---:|---:|---|
| Validation | 97 | 0.247716 | 0.519799 | -0.272083 [-0.311070, -0.232739] |
| Test | 217 | 0.254047 | 0.537655 | -0.283608 [-0.308311, -0.259382] |
| De novo | 19 | 0.505273 | 0.591381 | -0.086108 [-0.137410, -0.032376] |

| Split | Sparsa long-range R | MarinFold long-range R |
|---|---:|---:|
| Validation | 0.207335 | 0.501730 |
| Test | 0.204232 | 0.512810 |
| De novo | 0.437127 | 0.548926 |

The later decontaminated MarinFold step-363000 reference scores 0.551707 on
validation and 0.609832 on de novo. Its saved results do not cover the 217 test
proteins, so it is reported separately. The exp199 comparison is a contaminated
contrast, not the primary decontaminated reference.

[Per-protein results](final/per_protein.csv), [all metrics](final/summary.csv),
[paired comparisons](final/paired_comparisons.csv), and
[later-reference comparisons](final/paired_comparisons_step363000.csv) preserve
full precision. Each comparison checks exact per-protein candidate, true-contact,
and top-k counts before 5,000 paired bootstrap resamples (seed 17).

## Training and selection

The randomly initialized architecture combines a 12-layer, 512-dimensional
bidirectional sequence encoder with a 16-block, 96-dimensional pair network
using dilated convolutions and triangle multiplication. Inputs contain no MSA,
template, or pretrained PLM features. Supervision comes from MarinFold's curated
AF2 and ESMFold2 contact maps, sampled 50/50 by protein.

The complete eligible inventory contains 3,962,774 AFDB and 65,553,178 ESMFold2
Atlas rows across 5,405 shards. The corpus was sampled; training did not consume
all eligible rows. The [teacher audit](teacher_audit.json) records the 229
truncated AFDB exclusions and upstream decontamination provenance.

| Phase | Steps | Crop limit | Global batch | Sampled crops |
|---|---:|---:|---:|---:|
| Large-model pilot | 3,000 | 256 | 32 | 96,000 |
| Main training | 100,000 | 384 | 64 | 6,400,000 |
| Long-sequence finetuning | 3,000 | 1,024 | 32 | 96,000 |

Main training starts from the large pilot; finetuning starts from main step
96000. The selected model is a 50/50 **EMA weight average of finetuning steps
2500 and 1000**, without a distance cap. It runs as one model at inference.
The table lists completed phase budgets; selected weights precede final steps,
and retries can replay crops. All weights originated in this project's random
initialization and supervised training.

Only the 97 validation proteins guided decisions. Final selection compared
53 main-run and 9 finetuning candidates, including capped readouts and top-two
EMA averages. The chosen model was saved before scoring held-out proteins.
Its validation R-precision is 0.247716, versus 0.246151 for the original main-run
best. The improvement is small. Full [selection records](final/candidate_run_selection.json)
and [evaluation provenance](final/evaluation_manifest.json) are preserved.

## Artifacts and use

On this workstation, the recovered artifact directory is:

```text
/data/sparsa-runtime/outputs/sequence-pair-40m-20260911/sparsa-results
```

Durable storage, requiring the existing CoreWeave S3 access:

```text
s3://marin-us-east-02a/marin/protein-structure/sparsa/evaluations/sequence-pair-40m-20260911
```

The directory contains `sparsa.pt`, 333 dense score matrices, model/vocabulary
contracts, both training phases' metadata, timings, comparisons, and checksums.
Weights and matrices are kept outside Git. Checkpoint SHA-256:

```text
fad90bd545849e566e0ba9ca9d9a258977d2e9ef12125ae8f0ea7da4ae64e029
```

```bash
uv sync
uv run sparsa predict \
  --checkpoint /data/sparsa-runtime/outputs/sequence-pair-40m-20260911/sparsa-results/sparsa.pt \
  --sequence ACDEFGHIKLMNPQRSTVWY --out outputs/example --top-l 1
```

This writes `contacts.npz` and a zero-based, positive-only `helico_contacts.txt`.
The contact budget is top L; unlisted pairs remain unknown. The actual CLI and
Helico strict parser passed on [81-residue](final/helico_short.json) and
[761-residue](final/helico_long.json) validation chains. These checks establish
contact interoperability, not downstream structure quality. The Helico checkout
was dirty; the reports pin the actual parser and dependency hashes.

The [artifact audit](final/artifact_audit.json) verified all 362 indexed file
checksums, all 333 finite symmetric score matrices, source/data hashes, paired
scoring universes, and selection consistency. Prediction code was unchanged
between evaluation and these checks.

## Runtime, search, and limitations

On one H100, the 333 protein forward passes plus sigmoid and score copies took
9.773 seconds total (median 24.10 ms/protein). The longest chain was 920 residues.
Timing excludes model loading (0.861 seconds), tokenization, input transfer,
validation selection (413.252 seconds), metric calculation, and file writing.
See [per-protein timings](final/timings.csv). No equivalent measured MarinFold
latency is available here, so no speedup ratio is claimed.

All GPU jobs used Iris batch priority. All 35 jobs are terminal, and the CPU
artifact bridges were stopped after recovery. Recorded running resource time
was **132.4724 H100-hours**, including pilots, retries, profiling, search,
finetuning, and evaluation. This is not a billing or utilization measurement;
six stopped attempts use labelled job-finish estimates and none has an unknown
started duration. See [compute summary](compute_summary.json).

The [architecture search](AUTORESEARCH.md) ran eight trials covering four ideas,
with matched training exposure and two-seed confirmation for promising changes.
Two ideas improved both seeds slightly, but failed the paired-interval gate.
The original architecture remains the search champion. The controller supports
new campaigns, resume, budgets, source isolation, and export; see the
[operating guide](../research/README.md).

This experiment does not establish that an LLM architecture is intrinsically
better: Sparsa has 40M parameters versus MarinFold's 1.5B, and training exposure,
compute, and readout differ. MarinFold's reference uses sampled contact-list
rollouts and vote ranking; Sparsa uses a single dense prediction. Validation
was adaptively reused for search and readout selection, so validation intervals
are descriptive. The held-out results remain the fixed outcome of this campaign.
