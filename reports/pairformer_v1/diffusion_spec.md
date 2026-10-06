# Current N128/N256 discrete diffusion

Audited 2026-10-06 against `sparsa/diffusion.py`, `sparsa/pairformer.py`,
`train_diffusion.py`, `evaluate_diffusion.py`, and the frozen campaign configs.

## State and labels

The state is a symmetric binary residue-contact map. An eligible undirected pair
has valid residues and sequence separation at least 6. We sample only the upper
triangle, mirror it, and fix padding, the diagonal and shorter separations to zero.
Targets are the curated MarinFold AF2/ESMFold2 contact documents. The contact
definition is native-amino-acid ConFind contact degree >= 0.001, not a C-beta
8-Angstrom distance threshold. Experimental validation uses the same definition.

## Forward process

There are **T = 8** steps. The stationary Bernoulli contact probability depends
only on sequence separation:

| Separation | p(contact) |
|---|---:|
| 6–11 | 0.025 |
| 12–23 | 0.016 |
| >=24 | 0.005 |

For a pair in bucket b, with p = p_b,

```
f(t) = cos²(((t/T + 0.008)/(1 + 0.008)) π/2)
alpha_bar[t] = f(t)/f(0), with alpha_bar[0]=1 and alpha_bar[8]=0 exactly
alpha[t] = alpha_bar[t] / alpha_bar[t-1]
Q[t] = alpha[t] I + (1-alpha[t]) [[1-p, p], [1-p, p]]
```

Each step keeps the previous bit with probability alpha[t]; otherwise it redraws
from Bernoulli(p). It can delete true contacts and insert false contacts. This is
biased binary replacement noise, not masking, Gaussian noise, or an unbiased
bit flip. Different eligible edges have independent forward corruption.

The exact marginal is

```
q(x_t=1 | x_0) = alpha_bar[t] x_0 + (1-alpha_bar[t]) p.
```

| t | alpha_bar[t] (retained clean signal) |
|---|---:|
| 0 | 1.000000 |
| 1 | 0.957805 |
| 2 | 0.847012 |
| 3 | 0.684227 |
| 4 | 0.493844 |
| 5 | 0.304395 |
| 6 | 0.144272 |
| 7 | 0.037472 |
| 8 | 0.000000 |

For example, at t=4 a true long-range contact survives with probability
0.493844 + 0.506156*0.005 = 0.496374; a non-contact becomes a false long-range
contact with probability 0.002531. At t=8 the whole map is an independent sparse
prior, with no information about its clean target.

## Training

For each protein crop, draw one timestep uniformly from 1..8 and sample its
noisy map directly from the forward marginal. The network predicts clean-map
logits from sequence, timestep and noisy map. There is **one denoising forward
pass and one backward pass per training case**, not an unrolled eight-step
trajectory and not gradients through discrete reverse samples. Current runs use
no self-conditioning and no inner loops.

The objective is positive-weighted BCE (positive weight 4), plus 0.05 times a
ranking KL. BCE is averaged over eligible pairs per protein, then over proteins.
For a protein with R > 0 contacts, the ranking term is

```
logsumexp(logits over eligible pairs)
- mean(logits at true contacts) - log(R).
```

That is KL(uniform true contacts || softmax over eligible pairs). Zero-contact
proteins contribute BCE only. The ranking term is normalized by the number of
positive-contact proteins in each original 16-protein group. This is an
x0-prediction surrogate objective, not an exact variational diffusion loss.

N128 uses a four-layer, width-384 sequence Transformer; width-128 pair features;
48 independently parameterized Pairformer blocks; and four additional pair-update
blocks. Noisy-state and time embeddings enter before the first Pairformer block.
The 48-block trunk is recomputed at every denoising step. N256 changes the pair
and triangle width to 256. C128/C256 instead inject noise after a sequence-only
48-block conditioning trunk, whose output can be cached across sampling steps.

## Reverse sampling and scoring

Initialize x8 from the sparse stationary prior. For t=8,...,1:

1. Predict clean logits l from (sequence, x_t, t).
2. Convert to p_hat(x0=1) = sigmoid((l - log(4))/temperature), temperature=1.
   Subtracting log(4) compensates for BCE's positive weight; the added ranking
   objective means this is a calibration heuristic, not a calibration guarantee.
3. Mix the exact forward posteriors using that predicted clean probability:
   p_theta(x_(t-1)|x_t) = sum_{x0 in {0,1}} q(x_(t-1)|x_t,x0) p_hat(x0).
4. Draw a binary previous map, upper triangle then mirror.

The forward posterior is computed from
q(x_(t-1)=k|x_t=j,x0=i) proportional to Qbar[t-1][i,k] Q[t][k,j].
At t=1 this becomes a Bernoulli clean-map sample. Randomness comes from the
initial map and all reverse draws; learned global pair updates couple the
predictions across edges.

For each validation protein, generate 100 rollouts. **Oracle R-precision@100**
ranks all eligible pairs by each rollout's final dense clean probabilities,
takes exactly the ground-truth R pairs, and reports the best rollout's TP/R.
**Consensus R-precision** ranks pairs by frequency in the 100 final sampled
binary maps, also takes exactly R, and reports TP/R. Ties use ascending residue
pair order. R is used only by the evaluator, never supplied to generation.
Oracle selection uses truth and is not a deployable sample-selection method.
Only the fixed 97-protein validation split guides choices; regular evaluations
are every 10,000 optimizer steps.
