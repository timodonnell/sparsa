# Scaling campaign results

Selection used the fixed 97-protein validation split. The held-out sets were evaluated after selection.

| Split | Proteins | Selected Sparsa | Previous 40M | MarinFold¹ |
|---|---:|---:|---:|---:|
| eval-val | 97 | 0.266908 | 0.247716 | 0.519799 |
| eval-test | 217 | 0.277041 | 0.254047 | 0.537655 |
| eval-denovo | 19 | 0.524895 | 0.505273 | 0.591381 |

¹ Decontaminated exp232 m2-p06, step 145199.

Test R-precision change versus the previous release: +0.022994; paired bootstrap 95% interval [+0.018477, +0.027712].
Validation intervals are descriptive because validation was reused for adaptive selection.

Selected model: 1,815,192,865 parameters from `s3://marin-us-east-02a/marin/protein-structure/sparsa/runs/scale-main-20260911-long`.
Checkpoint: `s3://marin-us-east-02a/marin/protein-structure/sparsa/evaluations/scale-main-20260911/sparsa.pt`.
SHA256: `a403e13551e6e115b067b65c91a820a9a862cd9966ce821c8262b455c58a5f02`.
Campaign running GPU time, including retries: 428.256 H100-hours (see accounting timing qualifications).

Actual CLI exports and the Helico contact parser passed for 81- and 761-residue validation chains. This verifies contact-input compatibility, not downstream structure quality.

Full per-protein scores, paired comparisons, source hashes, model selection, and verification records are in [final/](final/). The original report remains in [../FINAL.md](../FINAL.md).
