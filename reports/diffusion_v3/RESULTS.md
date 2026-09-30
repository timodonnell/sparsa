# Rollout-consistent diffusion results

Architecture and checkpoint decisions use only the frozen 97-protein `eval-val`
split. Held-out sets remain untouched.

## Existing-model self-conditioning guidance

The G2-wide-SC 110k checkpoint was sampled with weaker self-conditioning to test
whether restoring rollout diversity could recover the stalled oracle metric.
All evaluations use the same 100-rollout seed bank.

![Self-conditioning guidance sweep](guidance/guidance_sweep.png)

| Guidance | Oracle R@100 | Mean rollout R | Consensus R | Pairwise Jaccard |
|---:|---:|---:|---:|---:|
| 0.00 | 0.331278 | 0.212956 | 0.312819 | 0.099624 |
| 0.25 | 0.358760 | 0.233030 | 0.316737 | 0.111344 |
| 0.50 | 0.381376 | 0.250926 | 0.317766 | 0.126114 |
| 0.75 | 0.402364 | 0.267704 | **0.320094** | 0.145394 |
| 1.00 | **0.406518** | **0.280163** | 0.319433 | 0.169320 |

Reducing guidance produces more diverse maps, but the loss of conditional
accuracy is larger than the oracle benefit from that diversity. Guidance 0.5
trails full conditioning by 0.025143 oracle R@100, with paired 95% interval
[-0.037394, -0.013462]. Guidance 0.75 is statistically indistinguishable from
full conditioning at -0.004154 [-0.013745, 0.005111], while its mean rollout
score is lower by 0.012459 [-0.016345, -0.009117]. Full guidance remains the
primary setting.

This rejects a sampling-only explanation for the plateau. The new G2-RSC and
G3-balanced-RSC runs instead change training to expose the model to its own
one-step reverse states. The 26.5M control isolates that change; the 92.0M model
tests whether the corrected reverse process continues to benefit from greater
sequence and global-pair capacity.
