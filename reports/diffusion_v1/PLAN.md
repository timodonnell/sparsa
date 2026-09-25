# Triangle discrete-diffusion screen

This campaign tests whether a stochastic, globally reasoning contact-map model
creates useful oracle best-of-100 headroom. Architecture and checkpoint choices
use only the fixed 97-protein `eval-val` split. Held-out evaluation remains
untouched.

All arms use a six-layer, 384-wide sequence transformer and a directed 64-wide
pair state. The pair denoiser contains outgoing and incoming triangle
multiplication, starting-node and ending-node triangle attention, and a 4x pair
transition. It contains no convolution. The full model has about 11 million
parameters.

The binary forward process has eight timesteps and refreshes contacts toward
fixed priors for sequence separations 6–11, 12–23, and 24+. Training samples one
timestep per protein and predicts the clean map with weighted BCE plus the
existing ranking loss. Reverse inference uses the exact binary D3PM posterior.

| Arm | Continuous work in one denoising call |
|---|---|
| G1 | One triangle cell |
| G1-L2 | One cell applied twice with shared weights and gradients through both applications |
| G1-U2 | Two untied cells with gradients through both applications |

The 25k-step screen uses seed 17, crop 256, global batch 64, the same logical
teacher stream, and 1.6 million training proteins per arm. Every endpoint gets
100 stochastic rollouts per validation protein. The primary metric is the exact
MarinFold oracle best-of-100 R-precision: score each rollout's ordered selected
contacts at its first R positions, then take the best rollout for that protein.
Consensus, mean-rollout accuracy, long-range oracle accuracy, unique-map count,
pairwise Jaccard similarity, and best-of-N curves are diagnostics.

Finalists are re-evaluated with a second fixed rollout seed bank before any
larger training decision. Training and evaluation run on CoreWeave H100s through
Iris at batch priority. Checkpoints and detailed rollout rows are written to
`s3://marin-us-east-02a/marin/protein-structure/sparsa/diffusion-v1/`.

The one-GPU operational smoke completed on 2026-09-25. It exercised teacher
reads, training, checkpointing, eight-step sampling, and oracle evaluation on a
real validation protein.
