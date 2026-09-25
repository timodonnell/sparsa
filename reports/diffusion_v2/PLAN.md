# Long-scale discrete-diffusion campaign

This campaign asks whether more data, a wider global pair representation, and
self-conditioning can close the oracle best-of-100 gap to MarinFold. Architecture
and checkpoint decisions use only the fixed 97-protein `eval-val` split. Held-out
sets remain untouched.

MarinFold exp232 trained its 1.5B model for 145,200 updates at global batch 128
and sequence length 8,192: 152.3 billion tokens from a 69.5-million-document
corpus. Diffusion v1 trained on 1.6 million protein crops. Each v2 arm therefore
has a 300,000-step horizon at global batch 128, or 38.4 million protein crops.
The source mixture matches MarinFold's successful token-proportional m2 mixture:
5.9522% AFDB and 94.0478% ESM Atlas. Crops increase from 256 to 384 residues.

| Arm | Parameters | Purpose |
|---|---:|---|
| G1-long | 10,936,513 | compute and data control |
| G2-wide | 26,490,753 | 8x512 sequence trunk; 128-wide pair state and triangle operators |
| G2-wide-SC | 26,491,137 | G2-wide plus clean-map self-conditioning |

G2-wide-SC uses the previous reverse step's predicted clean-map probabilities as
continuous pair features. During training, half of minibatches first obtain a
detached clean-map estimate and condition the loss-bearing forward pass on it.
No gradient flows through the preliminary estimate or through a sampled reverse
trajectory.

All arms use eight CoreWeave H100s through Iris at batch priority, seed 23, an
8-step binary D3PM, global batch 128, crop 384, and a 300k-step WSD schedule with
15k warmup and decay over the final 60k steps. They write resumable checkpoints
every 5,000 steps.

The first calibrated 100-rollout evaluation is at step 25,000. An arm may stop
there only if it trails G1-long by more than 0.01 oracle R-precision and the
paired protein-bootstrap upper bound is at most zero. Surviving arms are checked
again at 100,000 steps. G1-long and the best architectural arm continue to the
300,000-step endpoint. The second rollout seed bank is reserved for close
selection decisions. The primary metric remains MarinFold-compatible oracle
best-of-100 R-precision; long-range oracle, mean rollout, consensus, diversity,
and inference cost are diagnostics.
