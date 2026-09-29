# Long-scale discrete-diffusion results

Architecture and checkpoint decisions use only the frozen 97-protein `eval-val`
split. Held-out sets have not been read.

![Oracle R-precision by training step](figures/r_precision_by_step.png)

The figure includes every discrete-diffusion checkpoint evaluated with rollout
seed 20260925. Error bars are pointwise 95% protein-bootstrap intervals. The
plotted values are available in
[`figures/r_precision_by_step.csv`](figures/r_precision_by_step.csv).

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
larger retry budgets. At this point G2-wide had reached step 125k; its 100k
checkpoint had already been pruned, so 125k became the next milestone.

## G2-wide at step 125k

The 125k checkpoint represents 16.0 million crop presentations, five times the
25k exposure. It was evaluated with the same frozen 97-protein split and fixed
100-rollout seed bank.

| Step | Oracle R@100 | Long R@100 | Mean rollout R | Consensus R |
|---:|---:|---:|---:|---:|
| 25,000 | 0.255871 | 0.272289 | 0.145363 | 0.236419 |
| 125,000 | **0.342294** | **0.365924** | **0.227966** | **0.331418** |

The paired gain from 25k to 125k is 0.086423 oracle R@100, with 95% bootstrap
interval [0.069411, 0.104713]. Long-range oracle improves by 0.093635 [0.072246,
0.115739]. Longer training therefore remains strongly productive for the wide
architecture.

G2-wide at 125k is statistically indistinguishable in oracle R@100 from
G2-wide-SC at 35k: the difference is -0.007586 [-0.029697, 0.014701]. The
self-conditioned checkpoint retains more sampling headroom, while the longer
plain-wide run has a much stronger consensus score (0.331418 versus 0.260265).
A checkpoint-matched comparison remains necessary after G2-wide-SC catches up.

The 125k oracle curve still rises from 0.331542 at 64 rollouts to 0.342294 at
100, and all 100 maps remain unique per protein. The remaining gap to MarinFold's
0.5199 oracle reference is 0.1776.

## G2-wide at step 180k

The first automatic 10k-cadence evaluation ran at the resumed 180k checkpoint,
after 23.04 million crop presentations.

| Step | Oracle R@100 | Long R@100 | Mean rollout R | Consensus R |
|---:|---:|---:|---:|---:|
| 125,000 | 0.342294 | 0.365924 | 0.227966 | 0.331418 |
| 180,000 | **0.357893** | **0.377666** | **0.244460** | **0.349876** |

The paired oracle gain from 125k to 180k is 0.015598, with 95% bootstrap
interval [0.007795, 0.023260]. Mean-rollout R-precision improves by 0.016493
[0.010891, 0.022482], and consensus improves by 0.018458 [0.011072,
0.026412]. The long-range oracle gain is 0.011742, but its interval
[-0.007290, 0.028909] includes zero.

G2-wide at 180k is 0.008012 above G2-wide-SC at 35k in oracle R@100, though
the mismatched-step comparison remains statistically unresolved
[-0.014932, 0.031380]. The gap to MarinFold's 0.5199 oracle reference is now
0.1620. Evaluation will continue every 10,000 steps on the frozen validation
split.

## Near-matched control at step 115k

G1-long at step 115k has seen 14.72 million crop presentations, close to
G2-wide's 16.0 million at step 125k.

| Arm | Step | Protein crops | Oracle R@100 | Long R@100 | Mean rollout R | Consensus R |
|---|---:|---:|---:|---:|---:|---:|
| G1-long | 35,000 | 4.48M | 0.245386 | 0.257664 | 0.139765 | 0.232834 |
| G1-long | 115,000 | 14.72M | 0.289951 | 0.307577 | 0.180763 | 0.281096 |
| G2-wide | 125,000 | 16.00M | **0.342294** | **0.365924** | **0.227966** | **0.331418** |

Longer training improves G1-long from 35k to 115k by 0.044565 oracle R@100,
with paired 95% interval [0.033337, 0.056790]. At near-matched exposure,
G2-wide still exceeds G1-long by 0.052344 [0.040348, 0.065436]. Its long-range
advantage is 0.058347 [0.043396, 0.074316]. The wider sequence and pair state
therefore provides a material gain beyond training duration alone.

G2-wide-SC at only 35k also exceeds G1-long at 115k by 0.059929 [0.043342,
0.076525] oracle R@100. Its lower consensus score indicates that this early
self-conditioned model derives more of its advantage from diverse rollouts.
The decisive architecture comparison remains G2-wide versus G2-wide-SC at
matched steps.
