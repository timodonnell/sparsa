# Sparsa training campaign

Status at 2026-09-11 05:32 UTC: implementation and pilots complete; production
training is queued for batch capacity after preemption. Final test/de novo
evaluation has not been run.

## Current production job

- Iris job: `/bizon/sparsa-sequence-pair-40m-20260911`
- Cluster config: `/home/bizon/git/marin-freshiris/lib/iris/config/cw-rno2a.yaml`
- Pod label: `iris.job_id=bizon.sparsa-sequence-pair-40m-20260911`.
  Discover the current pod with this label; pod names change on preemption.
- Priority: **batch**, root job, one node / 8 H100s.
- Output: `s3://marin-us-east-02a/marin/protein-structure/sparsa/runs/sequence-pair-40m-20260911`
- Config: `configs/sequence_pair.yaml`; 39,994,657 parameters, crop 384,
  global batch 64, 100,000 steps, validation every 2,000 steps, EMA 0.999.
- Starts from the large pilot's step-3000 online weights and EMA; all weights
  originated in this project's random initialization and supervised training.
- Measured main-run speed: about 0.44 s/step; peak memory about 51.4 GiB/GPU.
- Expected compute: approximately 12 hours / 96 H100-hours plus pilots; the job
  timeout is 18 hours. It retries up to five times with automatic checkpoint and
  data-cursor restoration. Checkpoints retain optimizer and RNG state.
- Production validation: step 2000 R-precision 0.160246; step 4000 0.174896;
  step 6000 0.184494; step 8000 0.192853; step 10000 0.196659;
  step 12000 0.202191; step 14000 0.206153; step 16000 0.210827;
  step 18000 0.209735 (long-range 0.166595). The raw selection remains step 16000.
- At about 05:23 UTC the original attempt was preempted for a higher-priority
  workload. A replacement was also preempted; there was one intervening pod-deletion
  retry. Attempt 3 is SchedulingGated, awaiting eight GPUs at batch priority.
  The CPU-only checkpoint audit confirmed that step 18000 is durable and that
  the best pointer still selects step 16000. See `preemption_checkpoint.json`.
  About 1,900 unsaved steps must be replayed. Resume has not yet been verified
  on a replacement GPU attempt; the coordinator remains live and waiting.
- The original approximate 11 a.m. Eastern completion estimate is now conditional
  on batch capacity. Keep priority at batch and allow Iris to schedule the retry.
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
- Eleven local tests pass, including finite-worker cursor restoration, masked loss,
  padding isolation, symmetry, optimization, and checkpoint loading. A real
  4-GPU resume from step 500 successfully trained/evaluated step 501.
- Helico's actual `contacts_from_pairs(..., strict=True)` accepts the export;
  unlisted entries remain UNKNOWN. See `helico_integration.json`.

All pilot details are in `pilot_summary.json`; launcher commands are in `jobs/`.
The original pilot writer made synchronous small-object writes every 25 steps;
two pilots stalled there and were cancelled. The production writer flushes
history at checkpoints and bounds network timeouts/retries. Data-loader state
now saves compact shard/row cursors rather than replaying the entire stream.

## Remaining work

The local `scripts/finish_run.py --name sequence-pair-40m-20260911` coordinator
is running (initial PID 3284859). It waits for training success, submits the final
evaluation at **batch** priority, then a CPU recovery task, downloads the bundle,
and verifies SHA256 checksums. **Do not submit a duplicate final evaluation while
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
   distance capping at crop minus one at every durable checkpoint, using validation
   only. It saves the chosen checkpoint and inference setting before held-out
   evaluation. A real one-H100 batch preview passed for the first six checkpoints:
   capping improved each, reaching 0.203275 R-precision / 0.154691 long-range at
   step 12000 (original: 0.202191 / 0.152688). See `readout_preview.json`.
   Tests verify unchanged within-cap outputs and checkpoint reload reproduction.
   Training computations are unchanged.
3. Run `scripts/final_evaluate.py` as a one-H100 **batch** job after training
   completes. It uses `best.json`, evaluates all 333 fixed proteins, checks exact
   candidate/true/top-k parity against MarinFold, and writes inference weights,
   scores, timing, and paired comparisons to an evaluation prefix on S3.
   It also compares against the later step-363000 MarinFold reference on the 116
   available validation/de novo proteins; the 217 test proteins are unavailable
   for that reference. Both source and filtered table hashes are pinned separately.
4. Recover artifacts via a CPU-only `scripts/recover_results.py` task and
   `scripts/fetch_results.py`, or read individual files through a live task using
   `scripts/fetch_artifact.py`. Keep large local artifacts under `outputs`
   (a local ignored link to `/data/sparsa-runtime/outputs`).
5. Verify the released checkpoint/CLI, write final results and limitations,
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
  get pods -l iris.job_id=bizon.sparsa-sequence-pair-40m-20260911 -o wide
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
