# Triangle discrete-diffusion screen results

This screen trained G1, shared-loop G1-L2, and untied G1-U2 for 25,000 steps
(1.6 million examples) on eight CoreWeave H100s per arm at batch priority. Model
and checkpoint decisions used only the fixed 97-protein `eval-val` split. The
held-out split was not read.

Final metrics below are means over two fixed, independent 100-rollout seed
banks. Each bank used the exact MarinFold oracle definition described in
[PLAN.md](PLAN.md).

| Arm | Parameters | Oracle R@100 | Long R@100 | Mean rollout R | Consensus R | Mean contacts | Pairwise Jaccard | Inference GPU-s* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 | 10,936,513 | 0.2224 | 0.2349 | 0.1124 | 0.1951 | 166.4 | 0.0438 | 578 |
| G1-L2 | 10,936,706 | **0.2233** | **0.2369** | **0.1134** | **0.1995** | 157.0 | 0.0456 | 1,125 |
| G1-U2 | 11,062,338 | 0.2222 | 0.2253 | 0.1097 | 0.1976 | 176.3 | 0.0443 | 1,126 |

\*Sum over per-protein inference times; divide by eight for approximate job wall
compute. G1-L2 and G1-U2 do about twice G1's pair-trunk work per denoising step.

The primary differences are not distinguishable on this split. Averaging each
protein over the two rollout banks, the paired bootstrap difference for G1-L2
minus G1 is +0.0009 oracle R@100 (95% CI -0.0055 to +0.0073). G1-U2 minus G1 is
-0.0002 (95% CI -0.0066 to +0.0062). This is one training seed, so these
intervals measure validation-protein variation rather than training variation.
The extra continuous pair update did not produce a measurable oracle gain at
this scale. G1 is the efficient choice; G1-L2 is numerically best but costs
roughly twice as much at inference.

Longer training mattered much more than pair-cell depth. On the first rollout
bank, oracle R@100 rose from 0.1613 to 0.2249 for G1, from 0.1593 to 0.2252 for
G1-L2, and from 0.1583 to 0.2251 for G1-U2 between steps 5,000 and 25,000.

Sampling creates real oracle headroom. The two-bank mean G1-L2 curve is 0.1135
at one rollout, 0.1724 at 8, 0.1999 at 32, and 0.2233 at 100. All 100 sampled
maps were unique for every protein. G1's effective best across both independent
100-rollout banks is 0.2370, so the curve has not fully saturated at 100.

The result remains far behind MarinFold. Its current i.i.d. oracle best-of-100
reference on `eval-val` is 0.5199, leaving an absolute gap of about 0.297. A
seeded-contact MarinFold portfolio reaches 0.5341, but that is a different
sampling policy. The next experiment should change what the pair trunk learns,
rather than simply repeat the same triangle cell. The strongest candidates are
a wider pair state with more triangle-attention heads, self-conditioning on the
previous clean-map estimate, and a contact-count or top-k sampling policy tuned
on `eval-val`. Any larger run should retain G1 as the compute-matched control.

Raw summaries, per-protein scores, averaged architecture metrics, and paired
comparisons are in [`early/`](early/) and [`final/`](final/). Checkpoints and
full evaluation artifacts are under
`s3://marin-us-east-02a/marin/protein-structure/sparsa/diffusion-v1/`.
