# Sparsa training campaign

Status at 2026-09-11 15:47 UTC: implementation and pilots complete; production
main training is running and has reached step 90000. Final test/de novo
evaluation has not been run.

## Current production job

- Iris job: `/bizon/sparsa-sequence-pair-40m-20260911-r1`
- Cluster config: `/home/bizon/git/marin-freshiris/lib/iris/config/cw-rno2a.yaml`
- Pod label: `iris.job_id=bizon.sparsa-sequence-pair-40m-20260911-r1`.
  Discover the current pod with this label; pod names change on preemption.
- Priority: **batch**, root job, one node / 8 H100s.
- Output: `s3://marin-us-east-02a/marin/protein-structure/sparsa/runs/sequence-pair-40m-20260911`
- Config: `configs/sequence_pair.yaml`; 39,994,657 parameters, crop 384,
  global batch 64, 100,000 steps, validation every 2,000 steps, recovery checkpoints
  every 500 steps, EMA 0.999.
- Starts from the large pilot's step-3000 online weights and EMA; all weights
  originated in this project's random initialization and supervised training.
- Measured main-run speed: about 0.44 s/step; peak memory about 51.4 GiB/GPU.
- Expected main-run compute: approximately 12 hours / 96 H100-hours, plus pilots,
  replay after preemption, and the separate long-sequence phase; the main job
  timeout is 18 hours. It retries up to five times with automatic checkpoint and
  data-cursor restoration. Checkpoints retain optimizer and RNG state.
- Best original-readout validation: step 88000, R-precision **0.246046**,
  long-range **0.205477**. Step 90000 scores **0.245524** overall and
  **0.205540** long-range. Selection uses overall R-precision. The complete
  validation trajectory and durable latest/best pointers are recorded in
  `main_validation_progress.json`. These remain far below MarinFold's roughly
  0.52–0.55 validation R-precision; no architecture superiority is established.
- At about 05:23 UTC the original attempt was preempted for a higher-priority
  workload. A replacement was also preempted; there was one intervening pod-deletion
  retry. Attempt 3 initially waited in SchedulingGated for eight batch GPUs.
  The CPU-only checkpoint audit confirmed that step 18000 is durable and that
  the best pointer still selects step 16000. See `preemption_checkpoint.json`.
  Subsequent brief allocations were preempted too. Attempt 6 restored the full
  step-18000 checkpoint on eight GPUs and advanced beyond step 18900 at about
  0.44 s/step. See `resume_verification.json`. The unsaved tail is being replayed.
  Keep discovering current pods by label; do not reuse deleted pod names.
- Repeated preemptions lost up to roughly 1,900 steps between 2,000-step saves.
  After attempt 7 saved step 22000, the original job was intentionally cancelled
  and its pod deletion verified. Replacement `-r1` resumed the same run URI and
  full checkpoint with recovery saves every 500 steps. Its startup provenance
  confirms eight GPUs, batch priority, and the step-22000 resume. Validation
  remains every 2,000 steps; recovery-only saves do not change best-model selection.
  The first recovery-only save at step 22500 succeeded: latest has no validation
  result and best remains step 22000. See `checkpoint_handoff.json` for the
  durable-object check, pointers, resume identity, and source hashes.
  Both job handles are included in `main_job_accounting.json`; refresh this
  snapshot at completion to include all running time and preempted attempts.
  Iris omitted the final attempt's finish timestamp on the cancelled original
  job. Accounting now uses its job-finish timestamp as a labelled estimate,
  instead of incorrectly accruing time until every later observation. Six
  accounting regression tests pass, including the actual cancellation timestamps.
  Unknown stopped-attempt durations are reported separately; current main-run
  accounting has one estimated finish and no unmeasured started attempts.
- The replacement job was preempted at about 08:38 UTC, immediately after its
  durable step-38000 checkpoint. Iris started attempt 1 on a new node and its
  startup provenance confirms a full step-38000 resume on eight H100s at batch
  priority. The best pointer also selects step 38000. No manual resubmission
  or coordinator change was needed.
  Attempt 1 was preempted again shortly after restoration. At 08:39 UTC, attempt
  2 was waiting in SchedulingGated for batch capacity; the job and coordinator
  remain live. Attempt 2 was also preempted; attempt 3 subsequently obtained
  eight batch H100s, restored step 38000, and advanced past step 38100. See
  `resume_38000_verification.json`. Continue discovering current pods by label.
- A fourth replacement-job preemption occurred after the durable step-57500
  recovery checkpoint. Attempt 4 restored that checkpoint at 11:19:52 UTC on
  eight batch H100s, then advanced through the validated step-64000 checkpoint.
  See `resume_57500_verification.json`. The coordinator remains live and no
  manual resubmission was needed. Main-run running resource time is about
  69.36 H100-hours at that historical accounting snapshot, including replay.
- Replacement-job preemption 5 occurred after step 64200; attempt 5 received
  eight H100s on another node and restored step 64000 at batch priority. Training
  has advanced past the restore point. See `resume_64000_verification.json`.
  The coordinator remains active; no duplicate phase jobs have been submitted.
- Attempt 5 ended with a SIGABRT at about 15:08 UTC. Iris's saved diagnostic
  is truncated and does not establish the root cause. Attempt 6 restored the
  full step-85500 checkpoint on eight batch H100s and advanced through step 90000.
  See `resume_85500_verification.json`. This was an automatic retry of the same
  job; the completion coordinator remains live.
- Completion time depends on batch capacity. Keep priority at batch and allow
  Iris to schedule any retries.
  Final artifacts will preserve resume provenance; completion elapsed time is
  for the final attempt, not total campaign compute or wall-clock time.

## Evidence and choices

- Exact teacher eligibility audit: AFDB 3,962,774 retained of 3,963,003; ESMFold2
  Atlas 65,553,178 retained of 65,553,178. The 229 AFDB exclusions are truncated
  contact lists. All retained rows emit every above-threshold contact according
  to the source metadata. See `teacher_audit.json`.
- Compact (5,030,641 parameter) pilot: best eval-val R-precision 0.130447.
- Same compact network without triangle multiplication: 0.130356 at its saved
  step-2500 checkpoint. This pilot was interrupted by a synchronous log-upload
  stall after step 2650. It does not establish a meaningful triangle advantage.
- Large (39,994,657 parameter) pilot: 0.149709 at step 3000 / 96,000 crops;
  long-range R-precision 0.110275. Selected as the main architecture.
- Matched updated-loader pilots at 64,000 crops: 4x positive loss weighting
  reaches a best 0.125321; 16x reaches 0.123764. Retain 4x weighting.
- GPU profiling: crop 384, batch 8, no activation checkpointing uses 51.0 GiB
  and gives 19.0 crops/s/GPU on a fixed-shape synthetic throughput workload.
  Checkpointing the same workload gives 13.1 crops/s/GPU. See `gpu_profile.json`.
- The combined local suite passes: 21 tests in 10.55 seconds, including
  finite-worker cursor restoration, masked loss, padding isolation, symmetry,
  optimization, checkpoint averaging/loading, and attempt-time accounting. A real
  4-GPU resume from step 500 successfully trained/evaluated step 501.
- Helico's actual `contacts_from_pairs(..., strict=True)` accepts the export;
  unlisted entries remain UNKNOWN. See `helico_integration.json`.

All pilot details are in `pilot_summary.json`; launcher commands are in `jobs/`.
The original pilot writer made synchronous small-object writes every 25 steps;
two pilots stalled there and were cancelled. The production writer flushes
history at checkpoints and bounds network timeouts/retries. Data-loader state
now saves compact shard/row cursors rather than replaying the entire stream.

## Remaining work

A validation-only checkpoint averaging preview completed as
`/bizon/sparsa-average-preview-20260911` on one batch H100. It compared uniform
EMA weight averages of the top 2, 3, and 5 checkpoints against the best original
readout and every capped checkpoint through step 40000, using only eval-val.
Two local tests verify prediction-preserving checkpoint reload and rejection of
incompatible amino-acid vocabularies. The top-two average with cap 383 improved
R-precision from 0.228426 to 0.228618 (long-range: 0.187723 to 0.188320).
The gain is small. The top-three and top-five averages did not beat the best single
checkpoint. See `average_preview.json`; the experiment took 189.49 seconds.
Final selection now includes the top-two average within each completed training
phase, using only validation. The one-H100 batch job
`/bizon/sparsa-average-integration-20260911` passed: reloaded averaged weights
reproduce both readouts' validation metrics exactly. At the newer step-42000
snapshot, the best single checkpoint scored 0.230973 versus the average's
0.230448, and the selector correctly retained the single checkpoint. See
`average_integration.json`. Both experiments used validation only.

The local `scripts/finish_run.py --name sequence-pair-40m-20260911 --profile-job
/bizon/sparsa-long-profile-20260911 --training-job
/bizon/sparsa-sequence-pair-40m-20260911-r1` coordinator is running; read `handoff.pid`
for its current PID. It waits for main training and the long-crop GPU profile,
then runs the long-sequence finetune, final validation selection/evaluation, and
CPU artifact recovery. All jobs use **batch** priority. Recovered checksums are
verified before completion. **Do not submit a duplicate final evaluation while
the coordinator is active.** Inspect:

- `outputs/sequence-pair-40m-20260911/handoff.json`
- `outputs/sequence-pair-40m-20260911/handoff.log`
- `outputs/sequence-pair-40m-20260911/handoff.pid`

A file lock prevents two coordinators for this run. Restart the same command if
the local process stops; already submitted evaluation jobs are reused. A failed
remote job is reported for inspection. Final result interpretation, checkpoint
CLI verification, and completion of this project still require the agent.

1. Monitor training and recover from any infrastructure failures. Use only
   eval-val for training decisions. Do not score eval-test or eval-denovo until
   the model/readout choice is fixed.
2. The final-evaluation script rechecks the best original readout and evaluates
   distance capping at crop minus one at every durable validation checkpoint,
   plus the top-two EMA weight average with both readouts, using validation only.
   It saves the chosen checkpoint and inference setting before held-out
   evaluation. A real one-H100 batch preview passed for the first six checkpoints:
   capping improved each, reaching 0.203275 R-precision / 0.154691 long-range at
   step 12000 (original: 0.202191 / 0.152688). See `readout_preview.json`.
   Tests verify unchanged within-cap outputs and checkpoint reload reproduction.
   Training computations are unchanged.
3. The coordinator runs a 3,000-step finetune from the main run's best weights: crop
   1024, global batch 32, LR 5e-5, activation checkpointing, new seed 119. This
   supplies labels at separations absent from crop-384 training. Its output is
   `s3://marin-us-east-02a/marin/protein-structure/sparsa/runs/sequence-pair-40m-20260911-long`.
   The one-H100 prerequisite `sparsa-long-profile-20260911` completed successfully.
   At length 1024, batch 2 used 24.09 GiB and took 1.122 seconds per microbatch
   with backward/optimizer on a synthetic workload (no DDP or input I/O). See
   `long_profile.json`. Both remote script entry-point checks passed.
   Unseen distance embeddings start as copies of the trained edge bin at
   separation 383. Functional tests verify equivalence to capped inference.
   Resume skips this initialization and restores the finetune checkpoint intact.
   Finetuning now saves recovery checkpoints every 100 steps, while validation
   remains every 500 steps. At the measured larger-crop speed, this bounds replay
   to roughly four minutes instead of roughly twenty minutes per preemption.
4. Run `scripts/final_evaluate.py` as a one-H100 **batch** job after both phases
   complete. It chooses across both runs using validation only, so the finetune
   is retained only if it improves the primary metric. It uses `best.json`, evaluates all 333 fixed proteins, checks exact
   candidate/true/top-k parity against MarinFold, and writes inference weights,
   scores, timing, and paired comparisons to an evaluation prefix on S3.
   It also compares against the later step-363000 MarinFold reference on the 116
   available validation/de novo proteins; the 217 test proteins are unavailable
   for that reference. Both source and filtered table hashes are pinned separately.
5. Recover artifacts via a CPU-only `scripts/recover_results.py` task and
   `scripts/fetch_results.py`, or read individual files through a live task using
   `scripts/fetch_artifact.py`. Keep large local artifacts under `outputs`
   (a local ignored link to `/data/sparsa-runtime/outputs`).
6. Verify the released checkpoint/CLI, write final results and limitations,
   update README and this status, and mark the active goal complete only then.

## Monitoring and infrastructure notes

The old workstation Iris client is rejected by the controller. The isolated
current launcher is `.tools/iris/bin/iris` (a local link into
`/data/sparsa-runtime/iris`). Direct RNO2A submission works; the stale federation
route to `cw-us-east-02a` stayed queued and its probe was cancelled.

`iris job logs` returns no lines with the current client/server combination.
Use read-only Kubernetes logs instead:

```bash
kubectl --kubeconfig ~/.kube/coreweave-iris --context marin-rn02a_RNO2A -n iris \
  get pods -l iris.job_id=bizon.sparsa-sequence-pair-40m-20260911-r1 -o wide
# Then use the returned current pod name:
kubectl --kubeconfig ~/.kube/coreweave-iris --context marin-rn02a_RNO2A -n iris \
  logs CURRENT_POD -c task --tail=20
```

Only inspect/cancel this project's own jobs. Do not restart the shared cluster.
External workstation S3 credentials are expired; in-task Iris-injected storage
access works. Recovery helpers keep credentials inside the task. No bulk corpus
copy or public model publication is needed.

This campaign compares practical performance with pinned MarinFold checkpoints.
It does not isolate architecture causally: parameter count, training exposure,
compute budget, and readout differ from the 1.5B LLM. The fixed experimental
scoring protocol is directly comparable, and these differences must remain
visible in the final report.

## Separate architecture search

The 32 H100-hour reserved search campaign is running on four batch H100s per
trial. Both matched-budget baseline seeds completed (0.149292 and 0.151978
R-precision after 3000 steps). Candidate c001, sequence-attention features in the
pair trunk, improved both seeds by about 0.0026 on average, but its paired 95%
interval crossed zero, so it was not promoted. Candidate c002 adds shared
row/column attention in the pair trunk and is training. See
`autoresearch_progress.json`. These short-run scores are not comparisons against
the production run's much larger training budget. Only validation guides search.
