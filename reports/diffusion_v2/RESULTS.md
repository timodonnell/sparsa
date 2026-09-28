# Long-scale discrete-diffusion results

Architecture and checkpoint decisions use only the frozen 97-protein `eval-val`
split. Held-out sets have not been read.

## First milestone

Each entry is one fixed 100-rollout bank with rollout seed 20260925. The original
25k control and self-conditioned checkpoints were pruned before evaluation, so
the available comparison uses step 35k for those arms and step 25k for the wide
arm.

| Arm | Step | Protein crops | Oracle R@100 | Long R@100 | Mean rollout R | Consensus R |
|---|---:|---:|---:|---:|---:|---:|
| G1-long | 35,000 | 4.48M | 0.245386 | 0.257664 | 0.139765 | 0.232834 |
| G2-wide | 25,000 | 3.20M | 0.255871 | 0.272289 | 0.145363 | 0.236419 |
| G2-wide-SC | 35,000 | 4.48M | **0.349880** | **0.385760** | **0.219382** | **0.260265** |

G2-wide-SC improves step-matched oracle R@100 over G1-long by 0.104494. The
paired protein-bootstrap 95% interval is [0.090685, 0.118386]. Its long-range
gain is 0.128095 [0.102293, 0.157255]. Both estimates use 100,000 resamples with
seed 20260928.

G2-wide at step 25k exceeds G1-long at step 35k by 0.010485 [0.002478,
0.018468]. This comparison favors G1-long in training exposure, but the
architectures are not checkpoint matched. G2-wide-SC exceeds G2-wide by
0.094009 [0.080304, 0.108062], with the same step mismatch in the opposite
interpretive direction: self-conditioning has more training here.

The stopping condition is not met for either candidate because neither trails
the control. The self-conditioned architecture is the clear early leader. Its
0.349880 remains 0.1700 below MarinFold's 0.5199 i.i.d. oracle R@100 reference.
All 100 sampled maps were unique for every protein in every arm, and the oracle
curves were still rising at rollout 100.

CoreWeave preemptions and three distributed-process aborts stopped G1-long and
G2-wide-SC at step 35k. They were resubmitted from their intact checkpoints with
larger retry budgets. G2-wide reached step 125k and remains in progress; its
125k evaluation was submitted as the next milestone because its 100k checkpoint
had already been pruned.
