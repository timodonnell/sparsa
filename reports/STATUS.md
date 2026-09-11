# Campaign status — complete

The 40M sequence-only model completed 100,000 main-training steps and 3,000
long-sequence finetuning steps. Validation selected the uniform EMA weight
average of finetuning steps 2500 and 1000. All 333 experimental proteins have
been evaluated; test R-precision is **0.254047**, compared with **0.537655** for
the pinned decontaminated MarinFold exp232 m2-p06 step-145199 reference.

All 362 recovered artifact checksums and 333 score matrices passed the final
audit. The selected checkpoint passed the actual prediction CLI and Helico's
strict contact parser on 81- and 761-residue validation chains. No Helico
structure-quality evaluation was performed.

The eight-trial architecture search completed without a promotion. Production
training, finetuning, final evaluation, and GPU search trials succeeded. Recovery
and search CPU bridges were stopped after collecting results; all 35 Iris jobs
are terminal. Recorded running resource time is **132.4724 H100-hours**, including
pilots, retries, search, and evaluation. All GPU submissions used batch priority.

- [Final results, limitations, checkpoint, and reproduction](FINAL.md)
- [Architecture-search results](AUTORESEARCH.md)
- [Artifact audit](final/artifact_audit.json)
- [Compute summary](compute_summary.json) and [attempt accounting](all_job_accounting.json)
- [Historical training decisions and recovery notes](TRAINING_HISTORY.md)

Future improvements should use validation for decisions and keep the existing
held-out result fixed as this campaign's baseline. Direct pushes to `main` are
the requested workflow; no PR is required.
