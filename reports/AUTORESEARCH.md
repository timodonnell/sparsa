# Architecture search: first campaign

No candidate met the promotion rule; the original architecture was retained.
Two candidates improved both seeds slightly, but their paired intervals crossed
zero. This does not establish that the architectures cannot help with more
training or a different recipe.

Each trial trained from scratch for 3,000 steps / 96,000 crops with crop 256,
using the same decontaminated teacher mixture and 97 validation proteins.
The comparison fixes training exposure, not FLOPs or wall time. These scores
are separate from the much longer production training run.

| Candidate | Change | Seed 17 R | Seed 37 R | Outcome |
|---|---|---:|---:|---|
| c000 | Original 40M model | 0.149292 | 0.151978 | Retained baseline |
| c001 | Sequence-attention scores added to pair features | 0.151790 | 0.154626 | Failed paired-interval gate |
| c002 | Shared row/column pair attention | 0.150449 | — | Below screening threshold |
| c003 | Bilinear sequence-to-pair features | 0.150309 | — | Below screening threshold |
| c004 | Two pair-trunk passes with shared weights | 0.152717 | 0.153062 | Failed paired-interval gate |

For c001, mean R-precision gain was 0.002573 with paired 95% interval
[-0.000611, 0.005652]. For c004, mean gain was 0.002254 with interval
[-0.000445, 0.005181]. These intervals are descriptive because the search
adaptively reuses validation. No test or de novo evaluation guided the campaign.

All eight GPU trials succeeded at batch priority. Recorded running resource time
was **13.1386 H100-hours**; all **32 reserved H100-hours** were assigned, ending
the bounded campaign. Reservations are not refunded for fast trials. The CPU
artifact bridge was stopped. Running resource time excludes queue waits and is
not a billing or GPU-utilization measurement.

- [Decisions, scores, checkpoint references, and exported-source hashes](autoresearch_progress.json)
- [Per-job accounting](autoresearch_job_accounting.json)
- [Verification of all eight stored model-source snapshots](autoresearch_source_verification.json)
- [Protocol, controls, export, and resumption instructions](../research/README.md)

The local `outputs/autoresearch-v1/champion-export` directory contains the
retained baseline's runnable source and checkpoint references. It is a short-run
search artifact; production checkpoint selection is a separate process.
