# Rollout-consistent diffusion campaign

This campaign targets the oracle-R@100 plateau in G2-wide-SC. Architecture and
checkpoint decisions use only the frozen 97-protein `eval-val` split. Held-out
sets remain untouched.

From step 90k to 110k, G2-wide-SC's mean rollout R-precision rose from 0.2721 to
0.2802 and consensus rose from 0.3101 to 0.3194, while oracle R@100 changed from
0.4143 to 0.4065. Mean pairwise Jaccard rose from 0.1627 to 0.1693. The model is
improving its central prediction while rollouts become more alike, so simply
training the current formulation longer is unlikely to deliver the desired
oracle improvement.

The current training-time self-conditioning pass predicts a clean map from the
same noisy state and timestep used by the loss-bearing pass. At inference, the
conditioning map comes from the preceding reverse timestep and the current noisy
state was sampled using that prediction. Rollout-consistent self-conditioning
(RSC) trains on that inference-like one-step roll-in: predict from `x[t+1]`,
sample `x[t]` from the model reverse posterior, then predict the target clean map
at `t` while conditioning on the detached preceding prediction. Terminal-time
examples retain same-t conditioning. Gradients do not pass through sampling or
the preliminary prediction.

| Arm | Parameters | Sequence trunk | Pair trunk | Purpose |
|---|---:|---|---|---|
| G2-RSC | 26,491,137 | 8 x 512 | 128-wide triangle cell | Isolate RSC against G2-wide-SC |
| G3-balanced-RSC | 91,951,105 | 12 x 768 | 256-wide triangle cell | Test whether RSC benefits from substantially greater sequence and global-pair capacity |

Both arms use the established 8-step binary D3PM, self-conditioning probability
0.5, crop 384, global batch 128, seed 23, AFDB/ESM mixture, 300k WSD horizon,
and automatic 100-rollout evaluation every 10k steps. G3 uses microbatch 4 and a
smaller 2e-4 learning rate. Jobs run on eight H100s at batch priority.

Separately, the existing G2-wide-SC 110k checkpoint is evaluated with
self-conditioning guidance values 0, 0.25, 0.5, 0.75, and 1.0. Values below one
mix conditioned logits toward the model's unconditioned prediction, testing
whether reduced conditioning restores useful rollout diversity without
retraining.

Decision points:

- At 30k, compare each new arm with G2-wide-SC at 35k (0.3499), accounting for
  the 5k exposure difference.
- At 90k, require a clear improvement over G2-wide-SC at 90k (0.4143) to justify
  the full run.
- Use paired protein bootstrap intervals and the fixed rollout seed for primary
  decisions. Confirm close winners with the reserved rollout seed.
- Never use held-out test or de novo structures for iteration.
