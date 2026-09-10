# Discrete diffusion for protein contact-matrix completion

**Recommendation: start with an absorbing-mask model in the MDLM/MD4 family, compare its ordinary sampler with ReMDM remasking, and train one matched categorical D3PM/DiGress baseline.** Keep the sequence encoder, pair network, training examples, and observation masks the same. This separates the value of revising earlier decisions from differences in architecture or data. The choice is a research recommendation; no located protein-contact benchmark establishes a winning variant.

The closest protein precedents are **ProteinSGM**, which inpaints continuous residue-pair geometry, **LSD**, which generates contact maps through continuous latent diffusion, and **Mac-Diff**, which generates sequence-conditioned geometry maps. The closest discrete methodological precedent is **DiGress**, including its appendix on substructure-conditioned generation. None establishes the complete combination of sequence conditioning, diffusion directly over binary residue-pair entries, arbitrary positive/negative observations, and training on tens to hundreds of millions of structures. This is a bounded literature finding, not a claim that no unpublished or unindexed implementation exists.[^1][^2][^3][^4]

The main practical uncertainty is whether iterative sampling improves *jointly coherent completion* enough to justify its extra inference cost over a strong one-pass conditional predictor. A second uncertainty is how well a model trained on mostly single-state predictor outputs learns responses to informative contacts that contradict its sequence-only preference.

## 1. Problem definition

Let $S$ contain the amino-acid sequence, residue indices, and chain identities. Let $E$ be the set of valid unordered residue pairs, $C_e\in\{0,1\}$ their contact labels, and $O\subseteq E$ the observed pairs. The desired distribution is

$$
p_\theta(C_{E\setminus O}\mid S,O,C_O).
$$

The output should support both **joint samples of complete maps** and **estimated marginal contact probabilities**. These serve different purposes: a matrix of marginal probabilities need not describe one realizable conformation.

| Input situation | Meaning |
|---|---|
| No observed pairs | Predict a contact distribution from sequence alone. |
| Positive contacts only | Supplied pairs equal 1; every other valid pair is unknown. |
| Mixed binary observations | Supplied pairs may equal 0 or 1; every other valid pair is unknown. |
| Nearly complete map | Infer a small number of missing entries while preserving all observations. |
| Multichain system | Include interchain pairs, chain boundaries, and residue mappings explicitly. |

**Unknown, observed noncontact, and invalid/unresolved are different states.** Invalid pairs are excluded from supervision; they are not negative examples. Maintain a validity mask and an observation mask separately from the binary labels. During generation, distinguish immutable observations from the model's revisable guesses.

Choose a contact definition before building the corpus. A useful conventional baseline is Cβ–Cβ distance below 8 Å, using Cα for glycine, with a separately specified exclusion for neighboring residues. This is a proposed operational choice. LSD instead uses Cα–Cα below 8 Å, illustrating why contact definitions must accompany every comparison.[^2] A side-chain packing or rotamer-based contact definition should be treated as a separate target, not obtained by silently substituting a backbone-distance threshold.

“Any subset” can mean arbitrary *locations and numbers* of observations. It does not guarantee accurate extrapolation to every possible mask or consistency of contradictory inputs.

## 2. Previous protein-map attempts

### Closest precedents

| Work | What is generated or predicted? | Diffusion and conditioning | Relevance and remaining gap |
|---|---|---|---|
| **[ProteinSGM — Lee, Kim & Kim, 2023](https://www.nature.com/articles/s43588-023-00440-3)**; preprint 2022 | Pairwise distances and orientations, followed by Rosetta reconstruction | Continuous score-based diffusion; conditional generation through image inpainting | Direct precedent for completing protein geometry maps. Its task is backbone design, rather than sequence-conditioned binary-contact completion. Published data description restricts structures to 40–128 residues.[^1] |
| **[LSD — Yim et al., 2025](https://arxiv.org/abs/2504.09374)** | Contact probabilities decoded from residue latents; coordinates generated in a second stage | Continuous latent diffusion plus contact-conditioned structure generation | Closest contact-map generative representation. It does not diffuse binary edges directly or demonstrate arbitrary observed-edge completion. Uses 282,936 filtered AFDB examples of at most 128 residues.[^2] |
| **[Mac-Diff — Wang et al., 2026](https://www.nature.com/articles/s42256-026-01198-9)** | Residue-pair distance/orientation tensors, converted into conformations | Gaussian/score-based diffusion conditioned on ESM-2 sequence features | Strong sequence-conditioned map precedent. Evaluates conformational ensembles and derived contact probabilities, rather than arbitrary binary-entry completion.[^3] |
| **[NeuralPLexer — Qiao et al., preprint 2022, revised 2023](https://arxiv.org/abs/2209.15171)** | Protein–ligand block contacts/distograms and all-atom complex structures | Its contact module samples block adjacencies autoregressively; coordinate generation uses diffusion | Relevant hierarchical contact-to-structure architecture. Its contact sampler should not be described as discrete contact diffusion.[^5] |
| **[MF-ProtDisMap — Zhang et al.](https://www.sciencedirect.com/science/article/abs/pii/S0141813025081942)** | Real-valued interresidue distances | Combines ESM-2, MSA Transformer, and a diffusion-related “Diff-former” feature module | Related distance prediction. The available publisher preview does not establish a discrete map sampler or arbitrary inpainting capability; therefore it is not counted as a direct implementation.[^6] |
| **[ContactGAN — Subramaniya et al., 2021](https://github.com/kiharalab/ContactGAN)** | A refined contact map from an existing predicted map | GAN-based denoising/refinement | Earlier evidence that learning contact-map regularities is useful. It refines predictor errors rather than learning a diffusion completion distribution.[^7] |

**ProteinSGM establishes the inpainting idea.** Its geometric representation comprises Cβ distances and three angular matrices, with padding handled separately. The continuous noise process and reconstruction stage are central differences from a binary-edge model. The transferable lesson is that observed portions of a protein's pairwise representation can guide generation of the remainder; the paper does not establish the proposed sequence-conditioned task at scale.[^1]

**LSD establishes contact maps as an intermediate generative object.** Its structure-to-contact autoencoder produces one latent per residue, and a decoder predicts pairwise contact probabilities. This reduces the number of diffusion variables relative to a dense edge-token representation. Its data filtering and experiments remain far smaller and shorter than the proposed corpus. A latent approach is attractive for scaling, but exact observed-edge conditioning would require additional work.[^2]

**Mac-Diff strengthens the case for sequence-conditioned pair-space diffusion.** It combines sequence embeddings with an ESM-derived contact prior and locality-aware attention. Because that contact prior is predicted from sequence, it differs from externally supplied hard observations. It also generates richer geometric quantities than a contact bit. Its ensemble results are relevant to evaluating contact distributions, but do not determine which discrete corruption process to use.[^3]

### Useful boundaries around the literature

There is also an educational sequence-guided contact-map diffusion implementation in Michel Nivard's *Integrated protein diffusion language models* chapter. It uses a simplified image-diffusion approach with a protein language model. This is a useful prototyping example, not benchmark evidence for discrete diffusion or large-scale completion.[^8]

Many “discrete diffusion for proteins” papers generate **amino-acid sequences** or **per-residue structure tokens**. For example, MapDiff performs inverse folding: it generates sequence conditioned on a backbone.[^9] Campbell et al.'s discrete-flow protein co-design work jointly models sequence and structure, but is not a residue-pair contact-matrix completion result.[^10] Those papers inform training and sampling choices without resolving the target representation here.

The defensible research contribution would therefore be the combination of **arbitrary contact conditioning, globally coherent map sampling, and demonstrated scaling/generalization**. “Using diffusion for protein maps” by itself already has substantial precedent.

## 3. Choosing the discrete diffusion variant

Three choices are often conflated: the **corruption process**, the **training parameterization**, and the **sampler**. D3PM is a broad framework that includes masking; MDLM and MD4 simplify the masked case; ReMDM changes its reverse sampling process. Discrete flow matching provides another way to construct probability paths and transitions. They are not wholly separate model categories.[^4][^11][^12][^13][^14]

| Candidate | Corruption / state space | Can generated decisions change? | Assessment for this project |
|---|---|---|---|
| **MDLM / MD4** | Binary labels plus a distinct MASK state; labels are progressively hidden | Ordinary absorbing reverse sampling keeps revealed values fixed | Best initial training recipe. Missingness has explicit semantics and conditioning is straightforward. |
| **MDLM + ReMDM** | Same masked model, with a remasking reverse process | Yes, through remasking and resampling | First sampler upgrade to test. Particularly relevant if early choices prevent later global consistency. |
| **Categorical D3PM / DiGress** | Randomly replace 0/1 labels using a binary stationary distribution | Yes, 0↔1 transitions remain possible during denoising | Essential matched competitor. Use a sparse contact prior rather than assuming 50% contact noise is optimal. |
| **Discrete flow matching** | Chosen discrete path and continuous-time jump dynamics | Depends on path and added corrector dynamics | Strong second-stage option for studying sampling efficiency and custom priors. A minimal masked path alone does not provide revision. |
| **SEDD** | Learns probability ratios using score entropy; supports different kernels | Depends on the kernel and sampler | Principled alternative parameterization. With two clean classes, implementation simplicity favors clean-label prediction initially. |
| **EDGE / SparseDiff** | Graph-specific deletion or sparse edge-subset processing | Method-dependent | Most relevant if long proteins make dense pair processing prohibitive. These also change architecture or representation. |
| **Continuous latent diffusion** | Continuous noise on compressed residue latents | Yes in latent space | Scaling fallback inspired by LSD. Hard binary-edge control becomes less direct. |

The first three rows are the recommended controlled comparison. The assessments are proposed transfer judgments, not reported rankings on protein contact completion.

### 3.1 Absorbing masking: recommended starting point

For an unobserved pair $e$, let $\alpha(t)$ decrease from 1 to 0:

$$
q(X_{t,e}\mid C_e)
=\alpha(t)\,\delta_{C_e}
+[1-\alpha(t)]\,\delta_{\mathrm{MASK}}.
$$

Always keep $X_{t,e}=C_e$ for $e\in O$. A denoiser predicts the clean binary label from the sequence, current partial map, observation mask, and time.

MDLM and MD4 derive weighted cross-entropy objectives for masked diffusion. Applied to this conditional problem, the basic form is

$$
\mathcal L=
\mathbb E\!\left[
\frac{w(t)}{|E\setminus O|}
\sum_{e\in E\setminus O}
\mathbf1\{X_{t,e}=\mathrm{MASK}\}
\operatorname{CE}\!\left(C_e,f_{\theta,e}(S,X_t,O,C_O,t)\right)
\right],
\quad
w(t)=\frac{-\alpha'(t)}{1-\alpha(t)}.
$$

This expression assumes uniform sampling of $t$; nonuniform timestep sampling requires importance correction to preserve it. The normalization gives proteins comparable weight. A mean over masked entries alone changes the time weighting unless compensated. Handle endpoint singularities and examples with no missing entries explicitly.[^11][^12]

The operational advantage is simple: clean observations remain visible, and unknown entries have a distinct token. Training usually samples one noise level per example; it does not unroll the full sampling trajectory.[^11][^12]

The ordinary sampler has an important limitation: once an unknown entry has been filled, it cannot subsequently be revised. More denoising steps alone do not remove that limitation. **A model-generated noncontact is just as capable of prematurely blocking a fold as a model-generated contact.** The latter statement is a project-specific inference, motivating evaluation of both error types.[^11][^13]

MD4 also permits state-dependent masking schedules.[^12] Preserving rare positive entries longer is a plausible later experiment. Changing class-dependent corruption rates requires the corresponding loss and reverse-transition changes; it should not be implemented by modifying masking probabilities while retaining the ordinary sampler unchanged.

### 3.2 ReMDM: test revision without retraining the main model

[ReMDM](https://arxiv.org/abs/2503.00307) supplies a probabilistically motivated remasking sampler for pretrained masked models. It evaluates text, discretized images, and molecular strings; its protein-contact benefit remains untested.[^13]

Apply remasking only to **generated entries**. Supplied observations remain fixed. Start with a published time-dependent remasking schedule, then compare confidence-based selection. Use the same total number of denoiser evaluations when comparing samplers.

Raw confidence is potentially misleading in sparse maps: a model can be confidently wrong about a noncontact. Record correction rates separately for false positives and false negatives. Consider residue-row or contact-block remasking only as a later hypothesis, because it changes the transition process beyond the initial independent-entry formulation.

Two newer extensions are worth retaining as follow-up options. **PRISM** learns token-quality scores for remasking instead of relying only on softmax confidence. **SCMDM** feeds previous clean-state predictions into subsequent denoising, including at still-masked positions. Their reported gains are in other domains; neither is evidence of protein-contact superiority.[^15][^16]

### 3.3 Categorical D3PM / DiGress: the main alternative

Use a two-class replacement kernel

$$
Q_{t,e}=(1-\beta_t)I+\beta_t\mathbf1m_e^\top,
\qquad
m_e=(1-\pi_e,\pi_e).
$$

Here $\pi_e$ is a training-estimated contact prior. A global empirical rate is the simplest version; conditioning it on known covariates such as sequence separation, protein length, and same-chain status is a proposed extension. Do not use the target map's actual density or degrees at inference.

D3PM supplies tractable categorical forward and reverse transitions. DiGress applies them to edges, uses marginal edge frequencies to improve the noise distribution, and samples one triangle of an undirected adjacency matrix before symmetrizing.[^4][^14]

The model's noisy 0/1 entries are not necessarily trustworthy, so an explicit observation channel is especially important. Train with observations held clean and noise applied only to unobserved entries. With a clean-label predictor, compute the reverse transition by marginalizing the known bridge $q(X_{t-1,e}\mid X_{t,e},C_e)$ over its predicted $C_e$; do not simply resample its clean logits at every step and call that the same diffusion process.[^14]

For binary labels, uniform replacement is closely related to a bit-flip kernel after accounting for the probability of resampling the same value. **DIFUSCO** provides useful precedent for Bernoulli diffusion on binary graph optimization variables, although its tasks and feasibility constraints differ from protein contacts.[^17]

Masking's advantages on large-vocabulary text need not transfer to a two-class contact alphabet. There are only two clean outcomes here, and direct substitution noise may be a useful way to teach correction of wrong contacts. This is why categorical diffusion deserves a matched experiment rather than dismissal based on language-model rankings.

### 3.4 Discrete flow matching and sparse graph methods

The discrete-flow papers by **Campbell et al.** and **Gat et al.** formulate generation using continuous-time Markov chains and learned conditional predictions. **Shaul et al.** extend the choice of paths and transition dynamics. These are useful references if schedule choice, jump rates, or adjustable revision become the dominant experimental questions.[^10][^18][^19]

A simple masked flow can be very close operationally to absorbing diffusion. Choosing the “flow matching” label does not by itself solve early commitment; specify the path and any probability-preserving corrector dynamics. Likewise, **SEDD** changes the learning parameterization, while the chosen corruption kernel still determines important sampling behavior.[^18][^19][^20]

**EDGE** removes edges toward an empty graph and focuses denoising on selected active nodes. **SparseDiff** processes selected edge subsets to reduce memory requirements. These address a real scaling bottleneck, but adapting them introduces questions about candidate-edge recall and conditioning on known absent edges. Sparse internal processing also does not eliminate the cost of eventually emitting a dense matrix.[^21][^22]

## 4. Conditioning, calibration, and structural validity

### Observations and diffusion masks must be separate

For each training map:

1. Select valid target pairs and sample an observed set $O$.
2. Keep observed values clean; corrupt only $E\setminus O$.
3. Supply sequence, chain/index features, observation status, and the current corrupted map.
4. Predict clean labels and apply the appropriate diffusion objective on eligible unknown pairs.

At inference, initialize only unobserved entries from MASK or the categorical prior. Update these entries while keeping observations fixed. Sample each unordered pair once and mirror it, rather than independently predicting the two triangles and averaging binary samples.

This is a proposed **directly trained conditional model**. It avoids relying exclusively on inference-time conditioning of an unconditional model. DiGress does describe clamping for substructure generation, but reports possible long-range consistency problems in that setting. RePaint is another relevant inference-time inpainting precedent.[^4][^23]

**Hard clamping guarantees agreement with supplied bits, not exact sampling from the desired posterior.** Learned denoiser error, finite-step approximations, and factorized simultaneous updates remain. Small blocks or more evaluations can reduce some sampling errors, but need empirical validation.

### Train the observation regimes that will be used

Use a mixture of no observations, uniformly chosen binary observations, positive-only observations, residue-row masks, missing contiguous blocks, and interdomain or interchain masks. Include almost-complete inputs and sparse long-range positives. Choose mask sizes across a broad range rather than training at one fixed visibility fraction.

There is a subtle statistical issue with positive-only observations. If training selects observations with a rule $r(O\mid C,S)$, the conditional distribution learned from that dataset is proportional to

$$
p_{\mathrm{data}}(C\mid S)\,r(O\mid C,S)\,
\mathbf1\{C_O\text{ agrees with the observations}\}.
$$

Thus the selection mechanism can itself reveal information. For example, sampling a fixed number of positive edges uniformly can bias conditioning toward maps with particular total contact counts. Decide whether deployment observations represent an actual measurement process or simply imposed constraints. Match or explicitly model the former; for the latter, evaluate and control selection-induced bias. Merely adding an observation mask does not remove it. This is a statistical derivation, not a result claimed by the cited protein papers.

### Preserve probability meaning under class imbalance

A majority-negative matrix makes overall accuracy an inadequate metric. Nonetheless, indiscriminately upweighting positives changes the probability estimate. For true contact probability $p$, weighted binary cross-entropy has optimum

$$
p_{\mathrm{weighted}}
=\frac{w_1p}{w_1p+w_0(1-p)}.
$$

For probabilistic generation, begin with a likelihood-based objective. If pairs are subsampled, use known inclusion probabilities to obtain an unbiased estimate of the intended loss. Evaluate calibration separately on representative pair populations. Class-balanced auxiliary objectives are possible, but should not silently replace the generative objective.

Changing which pairs contribute to the loss does not reduce the cost of a network that still materializes all pair features. Computational sparsity requires an architectural change.

### A binary map does not enforce three-dimensional realizability

Symmetry and an appropriate diagonal are easy constraints. Realizability as a protein contact pattern is much harder: arbitrary symmetric graphs need not correspond to a chain embedded in three-dimensional space. Contacts are not transitive, so enforcing “two contacts imply the third” would be incorrect.

Evaluate complete *samples* with an independent coordinate-reconstruction or constrained-refinement procedure on a smaller test set. Measure recovered contacts, chain geometry, steric clashes, and compatibility with the fixed observations. A failed reconstruction is diagnostic rather than a definitive proof that the map is impossible.

If geometry is the limiting factor, add an auxiliary distance-bin prediction task or a coordinate-consistency stage. Generating 8–16 distance bins is a separate representation experiment. Binary observations then constrain a **set of allowed bins**; they do not reveal an exact distance. The conditioning and sampler must respect this distinction.

## 5. Scaling to tens or hundreds of millions of structures

### Data volume is plausible; effective diversity is the question

AFDB's 2024 publication documents coverage of over **214 million sequences**. The ESM release documents an initial **617 million** predicted metagenomic structures and a subsequent addition of **150 million**. These are historical release counts, not claims about the latest usable high-confidence corpus. Hsu et al. already demonstrated training benefits from **12 million AlphaFold2 predictions** for inverse folding.[^24][^25][^26]

The relevant training unit is a sequence/assembly with a well-defined contact target, confidence information, and provenance. Closely related sequences, duplicate predictions, and overlapping fragments can greatly inflate the nominal count. Track sequence clusters, structural clusters where practical, confidence strata, chain coverage, and teacher identity.

One predicted conformation per sequence mainly teaches the teacher's selected structure. It does not supply experimental state populations or calibrated conformational uncertainty. If ensemble completion is a goal, add genuine multiple-state examples or explicitly labeled ensemble data. AlphaFlow's use of PDB and molecular-dynamics ensembles is relevant precedent for learning beyond a single static prediction.[^27]

For multichain systems, monomer corpora do not supervise interfaces. Add actual predicted complexes or experimental biological assemblies; do not fabricate interchain labels by combining unrelated monomer coordinates. Include same-chain, chain/entity identity, and residue-index features.

### The pair count dominates simple cost estimates

The following are calculations for **100 million structures all of the stated length**, storing only the strict upper triangle. They exclude confidence, validity, sequence, indexing, and other overheads. TB denotes decimal terabytes.

| Residues $L$ | Distinct pairs per structure | Pairs per complete corpus pass | Binary labels at 1 bit/pair | Labels at 1 byte/pair |
|---:|---:|---:|---:|---:|
| 256 | 32,640 | $3.264\times10^{12}$ | 0.408 TB | 3.264 TB |
| 512 | 130,816 | $1.30816\times10^{13}$ | 1.635 TB | 13.082 TB |
| 1,024 | 523,776 | $5.23776\times10^{13}$ | 6.547 TB | 52.378 TB |

For variable lengths, cost depends on $\sum L_i(L_i-1)/2$, not just the mean length. A few long systems can dominate. A dense $L\times L\times128$ BF16 pair tensor alone occupies 64 MiB at $L=512$, and 256 MiB at $L=1024$, before saved activations, gradients, and other tensors.

**Do not flatten all $O(L^2)$ pairs into a standard full-attention transformer.** Attention over that many tokens costs $O(L^4)$. Full row/column attention and dense triangle operations typically cost $O(L^3)$, although memory depends on implementation. A residue-level transformer with shared pair readout and local/multiscale pair operations offers a more practical initial compromise.

Recommended implementation choices:

- Start with a sequence encoder plus a compact pair denoiser using local/dilated convolutions, downsampling, and residue-level global communication. Add a small number of triangle operations only if their benefit warrants the measured cost.
- Encode sequence once per sampling run and reuse its features. Compare a modest pretrained protein encoder with a trainable raw-sequence encoder; pretrained embeddings contain evolutionary information learned during pretraining even without an explicit MSA.
- Store sequence, compact coordinates or packed contacts, validity, and provenance in large shards. Use on-the-fly corruption; do not materialize noisy copies.
- Bucket by length and batch by estimated pair cost. Mix whole short proteins with crops; preserve original indices and include examples spanning distant sequence regions. Validate on whole structures.
- Let a later sparse architecture query every valid pair eventually, including pairs outside its initial candidate set. Otherwise a missed candidate becomes an irreversible false negative.

These are engineering proposals. Neither ProteinSGM's short-protein results nor LSD's filtered corpus establishes throughput or accuracy at the proposed scale.

### Teacher quality and evaluation leakage

Retain residue confidence and, where available, pairwise/domain-placement confidence. High local confidence alone does not establish the relative placement of two domains. Select or weight supervision accordingly; uncertain geometry and unresolved residues must not become confident noncontacts.

Build experimental validation/test sets before collecting the full distilled training corpus. Exclude homologous sequences and overlapping domains across splits, and include a harder structural-family or fold holdout. Report the actual sequence-identity and coverage rules rather than calling any single threshold “leakage-proof.”

Student-side filtering cannot establish that the teacher or a pretrained sequence encoder never saw related test data. Separate **student generalization**, **agreement with teacher outputs**, and **accuracy against experimental structures**.

## 6. Terse implementation plans

All plans use sequence conditioning, separate validity/observation masks, symmetric sampling, and both positive-only and mixed-observation tasks.

| Plan | Implementation | Decision it resolves |
|---|---|---|
| **A — Masked baseline** | Train a binary MDLM/MD4-style model on 1–3 million clustered examples, initially at 128–256 residues. Compare a one-pass conditional prediction with 16/32/64/128-step sampling. | Does iterative joint completion add value over ordinary conditional prediction? |
| **B — Remasking** | Apply ReMDM to A's checkpoints. Compare published time-based and confidence-based schedules at equal total evaluations; keep observed entries immutable. | Does revising generated contacts/noncontacts improve consistency and completion? |
| **C — Categorical replacement** | Train the same denoiser with binary D3PM transitions. Compare uniform and training-marginal priors; add length/separation-specific priors only after the global-prior comparison. | Does direct 0↔1 revision outperform masking plus remasking? |
| **D — Discrete flow matching** | Reuse the clean-label interface with a published discrete path and corrector. Sweep schedules and stochasticity against the best A–C model at equal wall time. | Can a different transition construction improve the quality–cost tradeoff? |
| **E — Richer geometry** | Add an auxiliary distance-bin head first; if helpful, test diffusion over 8–16 bins with binary observations represented as allowed-bin constraints. | Is the information loss from binary contacts the bottleneck? |
| **F — Sparse or latent scaling** | Adapt SparseDiff/EDGE-style edge processing, or an LSD-inspired latent model, after measuring dense-model limits. Explicitly test missed-edge recovery and observation adherence. | Can longer systems and larger datasets be handled without losing the completion capability? |

**Recommended order: A, then B and C, then scale the winner.** B is primarily a sampler comparison, so the initial diffusion study needs two main diffusion training recipes, not three separately trained diffusion models. D–F are contingent branches. The dataset sizes, lengths, and step budgets above are proposed pilot settings, not literature-established optima.

## 7. A small experiment that can select the variant

Use a shared validation suite and at least two training seeds for the main training recipes. Preserve matched data ordering and comparable training compute. Give each sampler an equal tuning budget; report both denoiser evaluations and measured runtime.

Train a dedicated one-pass completion baseline with the same architecture, observations, and data budget. Also report the diffusion model's first-pass output and a sequence-only baseline. This distinguishes the benefit of iterative sampling from the benefit of supplying contacts or changing the training objective.

| Axis | Minimum useful evaluation |
|---|---|
| Observed information | No observations; sparse positive-only sets; mixed binary entries; nearly complete maps; structured missing regions. |
| Contact range | Separate short/medium/long-range intrachain pairs and interchain pairs. |
| Generalization | Whole-protein evaluation; longer lengths; sequence-cluster holdout; a harder structural holdout. |
| Predictive accuracy | AUPRC and precision at fixed contact budgets on **unobserved** pairs; report per-protein and length-stratified results. |
| Probability quality | Brier score/calibration for estimated marginals; a correctly computed diffusion likelihood bound when available. |
| Joint sample quality | Sample-level contact overlap, contact-count/degree distributions, motif coherence, and coordinate-reconstruction diagnostics. |
| Constraint use | Exact observed-bit adherence, plus improvement on remaining pairs as informative observations are supplied. |
| Sampling tradeoff | Accuracy, diversity, structural diagnostics, and latency at matched 16/32/64/128-evaluation budgets. |

Do not score supplied contacts as successful predictions. Do not interpret the sum of marginal binary log losses as the joint map likelihood, or assume different variational bounds are equally tight. Report average-sample quality alongside best-of-$K$, with $K$ fixed across methods; otherwise drawing more samples creates an artificial advantage.

Add an especially revealing **state-selection test**: for a protein with two known structures, provide contacts distinctive of one state and assess completion toward that state. This tests whether observations actually control the result. If such data are scarce, at least compare empty conditioning against increasing informative contacts and against unrelated-but-consistent mask patterns.

A practical scale-up gate is a repeatable improvement over the one-pass baseline on held-out completion and sample coherence, with acceptable calibration and runtime. If ReMDM improves that tradeoff, retain masked training and its flexible sampler. If categorical diffusion wins, prioritize its kernel/schedule. If neither helps, investigate the pair architecture, observation distribution, or geometric representation before multiplying the dataset by 100.

## 8. Implementation references and evidence limits

| Component | Source to start from |
|---|---|
| Masked objective and PyTorch baseline | [MDLM implementation](https://github.com/kuleshov-group/mdlm); [MD4 implementation](https://github.com/google-deepmind/md4). |
| Remasking sampler | [ReMDM implementation](https://github.com/kuleshov-group/remdm). |
| Categorical edge diffusion | [DiGress implementation](https://github.com/cvignac/DiGress), especially transitions, symmetry, and conditional-generation discussion. |
| Geometry-map inpainting reference | [ProteinSGM implementation](https://gitlab.com/mjslee0921/proteinsgm). |
| Sequence-conditioned geometry reference | [Mac-Diff implementation and weights](https://github.com/Paulie-ai/Mac-Diff). |
| Contact-latent architecture | [LSD paper](https://arxiv.org/abs/2504.09374); a dedicated working release was not verified. |

Evidence coverage extends through **10 September 2026**. The core classification is based on original papers, author manuscripts, and author-maintained repositories. ProteinSGM's published abstract/data statement and an accessible author preprint support its representation and inpainting description. MF-ProtDisMap is characterized conservatively from its publisher preview; its exact diffusion implementation and issue date were not verified. The newer PRISM and SCMDM papers are follow-up leads with no located contact-completion evaluation.

No located study provides a controlled head-to-head comparison of the shortlisted variants on this exact task. No cited protein-map study establishes the proposed hundred-million-structure training regime. Those two gaps are precisely what the recommended pilot and subsequent scaling study should resolve.

## Sources

[^1]: Jin Sub Lee, Jisun Kim, Philip M. Kim. **[Score-based generative modeling for de novo protein design](https://www.nature.com/articles/s43588-023-00440-3)**. *Nature Computational Science* 3, 382–392, 2023. DOI: 10.1038/s43588-023-00440-3. [Accessible author preprint](https://www.researchgate.net/publication/364482484_ProteinSGM_Score-based_generative_modeling_for_de_novo_protein_design), 2022; preprint DOI: 10.21203/rs.3.rs-1855828/v1.

[^2]: Jason Yim, Marouane Jaakik, Ge Liu, Jacob Gershon, Karsten Kreis, David Baker, Regina Barzilay, Tommi Jaakkola. **[Hierarchical protein backbone generation with latent and structure diffusion](https://arxiv.org/abs/2504.09374)**. arXiv:2504.09374, 2025. Sections 2–3 and implementation statement.

[^3]: Baoli Wang, Chenglin Wang, Jingyang Chen, Danlin Liu, Changzhi Sun, Jie Zhang, Kai Zhang, Honglin Li. **[Conditional diffusion with locality-aware modal alignment for generating diverse protein conformational ensembles](https://www.nature.com/articles/s42256-026-01198-9)**. *Nature Machine Intelligence* 8, 415–434, 2026. Published 25 February 2026. DOI: 10.1038/s42256-026-01198-9.

[^4]: Clément Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, Pascal Frossard. **[DiGress: Discrete Denoising Diffusion for Graph Generation](https://arxiv.org/abs/2209.14734)**. ICLR 2023; arXiv:2209.14734. Sections 3–4 and Appendix E.

[^5]: Zhuoran Qiao, Weili Nie, Arash Vahdat, Thomas F. Miller III, Anima Anandkumar. **[State-specific protein-ligand complex structure prediction with a multi-scale deep generative model](https://arxiv.org/abs/2209.15171)**. arXiv:2209.15171, 2022; v2, 2023. NeuralPLexer contact-prediction module and Figure 1.

[^6]: Yufei Zhang et al. **[MF-ProtDisMap: protein real-valued distance prediction with fusion of sequence and coevolutionary features](https://www.sciencedirect.com/science/article/abs/pii/S0141813025081942)**. *International Journal of Biological Macromolecules*. DOI: 10.1016/j.ijbiomac.2025.147637. Publisher preview; exact issue date not verified.

[^7]: Sai Raghavendra Maddhuri Venkata Subramaniya, Genki Terashi, Aashish Jain, Yuki Kagaya, Daisuke Kihara. **Protein Contact Map Refinement for Improving Structure Prediction Using Generative Adversarial Networks**. *Bioinformatics*, 2021. [Author-maintained ContactGAN repository and citation](https://github.com/kiharalab/ContactGAN).

[^8]: Michel Nivard. **[Integrated protein diffusion language models](https://michelnivard.github.io/biobook/Chapter5_Proteins.html)**. *Sequence Language Models & Deep Learning in Genomics*, educational chapter; associated [contact-map diffusion example](https://gist.github.com/MichelNivard/21734f228ec29d4c91fcb123f2ec4aaf), 2025.

[^9]: Peizhen Bai et al. **[Mask prior-guided denoising diffusion improves inverse protein folding](https://arxiv.org/abs/2412.07815)**. arXiv:2412.07815, 2024. MapDiff.

[^10]: Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, Tommi Jaakkola. **[Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997)**. arXiv:2402.04997, 2024.

[^11]: Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, Volodymyr Kuleshov. **[Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)**. NeurIPS 2024; arXiv:2406.07524. Sections 3–4.

[^12]: Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, Michalis K. Titsias. **[Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.04329)**. NeurIPS 2024; arXiv:2406.04329, v4 revised 16 January 2025. MD4; weighted cross-entropy and state-dependent schedules.

[^13]: Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, Volodymyr Kuleshov. **[Remasking Discrete Diffusion Models with Inference-Time Scaling](https://arxiv.org/abs/2503.00307)**. NeurIPS 2025; arXiv:2503.00307, v4 revised 7 February 2026. ReMDM.

[^14]: Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg. **[Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)**. NeurIPS 2021; arXiv:2107.03006. D3PM.

[^15]: Jaeyeon Kim, Seunggeun Kim, Taekyun Lee, David Z. Pan, Hyeji Kim, Sham Kakade, Sitan Chen. **[Fine-Tuning Masked Diffusion for Provable Self-Correction](https://arxiv.org/abs/2510.01384)**. arXiv:2510.01384, 2025; v4 revised 22 May 2026. PRISM.

[^16]: Michael Cardei, Huu Binh Ta, Ferdinando Fioretto. **[Simple Self-Conditioning Adaptation for Masked Diffusion Models](https://arxiv.org/abs/2604.26985)**. arXiv:2604.26985, 2026. SCMDM.

[^17]: Zhiqing Sun, Yiming Yang. **[DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization](https://arxiv.org/abs/2302.08224)**. NeurIPS 2023; arXiv:2302.08224.

[^18]: Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi, Yaron Lipman. **[Discrete Flow Matching](https://arxiv.org/abs/2407.15595)**. arXiv:2407.15595, 2024.

[^19]: Neta Shaul, Itai Gat, Marton Havasi, Daniel Severo, Anuroop Sriram, Peter Holderrieth, Brian Karrer, Yaron Lipman, Ricky T. Q. Chen. **[Flow Matching with General Discrete Paths: A Kinetic-Optimal Perspective](https://arxiv.org/abs/2412.03487)**. arXiv:2412.03487, 2024.

[^20]: Aaron Lou, Chenlin Meng, Stefano Ermon. **[Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834)**. ICML 2024; arXiv:2310.16834, first posted 2023. SEDD.

[^21]: Xiaohui Chen, Jiaxing He, Xu Han, Li-Ping Liu. **[Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling](https://arxiv.org/abs/2305.04111)**. arXiv:2305.04111, 2023. EDGE.

[^22]: Yiming Qin, Clément Vignac, Pascal Frossard. **[Sparse Training of Discrete Diffusion Models for Graph Generation](https://arxiv.org/abs/2311.02142)**. arXiv:2311.02142, 2023; v2, 2024. SparseDiff.

[^23]: Andreas Lugmayr, Martin Danelljan, Andrés Romero, Fisher Yu, Radu Timofte, Luc Van Gool. **[RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)**. CVPR 2022; arXiv:2201.09865.

[^24]: Mihaly Varadi et al. **[AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences](https://academic.oup.com/nar/article/52/D1/D368/7337620)**. *Nucleic Acids Research* 52(D1), D368–D375, 2024. DOI: 10.1093/nar/gkad1011.

[^25]: Meta / ESM authors. **[Evolutionary Scale Modeling repository: ESM Metagenomic Atlas release history](https://github.com/facebookresearch/esm)**. Release descriptions for November 2022 and March 2023.

[^26]: Chloe Hsu, Robert Verkuil, Jason Liu, Zeming Lin, Brian Hie, Tom Sercu, Adam Lerer, Alexander Rives. **[Learning inverse folding from millions of predicted structures](https://proceedings.mlr.press/v162/hsu22a.html)**. ICML 2022; PMLR 162, 8946–8970.

[^27]: Bowen Jing, Bonnie Berger, Tommi Jaakkola. **[AlphaFold Meets Flow Matching for Generating Protein Ensembles](https://arxiv.org/abs/2402.04845)**. arXiv:2402.04845, 2024. AlphaFlow / ESMFlow.
