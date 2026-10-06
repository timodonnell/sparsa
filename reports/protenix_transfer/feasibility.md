# Protenix transfer feasibility — 2026-10-06

**Recommendation: test Protenix v1 transfer before committing to another large
from-scratch architecture.** The recycling insertion point is suitable. The
hypothesis is better sample efficiency from pretrained structural reasoning;
neither improved oracle@100 nor lower wall-clock time has been demonstrated.

## Evidence inspected

Upstream source: `bytedance/Protenix` commit
`85767b811c40ed46e73a9b39519cf6bfca8701ba`. Inspected the actual official
`protenix_base_default_v1.0.0.pt` checkpoint's ZIP/pickle metadata using HTTP
range requests and a restricted metadata-only unpickler. No model tensors were
loaded or executed. No v2 weights were downloaded.

| | v1 default | v2 |
|---|---:|---:|
| Pairformer blocks | 48 | 48 |
| Pair width | 128 | 256 |
| Single width | 384 | 384 |
| Complete published model | 368,484,735 parameters (checkpoint verified) | ~464.44M (official model table) |
| Published PDB cutoff | 2021-09-30 | 2021-09-30 |
| MSA/template inputs in original model | Yes | Yes |
| ESM/PLM input in these variants | No | No |

The v1 checkpoint contains **147,400,704 Pairformer parameters**. Its coordinate
diffusion module alone has 203,257,585 parameters, and its confidence head has
12,901,966. Removing those heads, MSA/template modules and the distogram head
leaves **148,895,680 parameters** in the input embedding, projections, recycling
layers and trunk, before adding our adapter/contact head. Thus the proposed v1
backbone is close to N128's size, not a 368M-parameter contact model.

Sources: [model configurations](https://github.com/bytedance/Protenix/blob/85767b811c40ed46e73a9b39519cf6bfca8701ba/configs/configs_model_type.py),
[base configuration](https://github.com/bytedance/Protenix/blob/85767b811c40ed46e73a9b39519cf6bfca8701ba/configs/configs_base.py),
[official checkpoint location](https://github.com/bytedance/Protenix/blob/85767b811c40ed46e73a9b39519cf6bfca8701ba/protenix/web_service/dependency_url.py),
[supported models](https://github.com/bytedance/Protenix/blob/85767b811c40ed46e73a9b39519cf6bfca8701ba/docs/supported_models.md).

## Concrete architecture

Use the native pretrained modules and their exact feature conventions, rather
than loading their weights into Sparsa's similar but incompatible Pairformer.
The upstream recycling path is literally

```
z = z_init + linear_no_bias_z_cycle(layernorm_z_cycle(z_previous))
```

Replace the previous latent pair representation with a learned adapter output:

```
u_ij = adapter(noisy_contact_ij, timestep, separation/prior features)
z_input = z_init(sequence) + W_recycle(LN_recycle(u))
s_input = s_init(sequence) + W_single_recycle(LN_single(zeros))
s, z = pretrained_pairformer(s_input, z_input)  # one 48-block pass
contact_logits_ij = symmetric_contact_head(z_ij, z_ji)
```

The adapter projects to 128 channels for v1, 256 for v2. Load the learned recycle
normalization/projection parameters; do not re-zero them as at model
initialization. Use small, nonzero residual gating/initialization and inspect
activation scales and adapter gradients. Include time as a channel-dependent
embedding, since a uniform scalar shift would be removed by LayerNorm. The
recycling representation is a learned latent tensor, not a contact map: learning
the adapter is a substantive transfer problem.

Retain the native input embedder and relative-position features. Construct them
from sequence and ideal residue chemistry; no experimental coordinates, templates,
MSA searches, pretrained PLM embeddings, or homolog-derived profile/deletion
features. Supply a query one-hot profile and zero deletion means. Bypass MSA and
template updates. The native MSA module explicitly returns its input unchanged
when the `msa` feature is absent. A query-only pass through that module could be a
later sequence-only ablation if bypassing it transfers poorly.

Discard coordinate diffusion and confidence prediction. Train a new binary
ConFind contact head. The native distogram describes distance bins and cannot
be converted exactly to MarinFold's ConFind labels with a distance threshold.
Use our existing 8-step binary diffusion, loss, teacher inventory and evaluators
initially, to isolate transfer from changes to the diffusion task.

One trunk pass per sampled training timestep is sufficient. During sampling,
run this trunk once at each of the eight reverse steps. Start without extra
latent recycling across diffusion steps; that would require additional training
choices and could introduce train/sampling mismatch. Upstream default inference
uses 10 recycling passes; copying that default at every diffusion step would
multiply trunk cost by ten. Upstream also disables gradients through earlier
recycles, so calling its complete inference wrapper unchanged would be wrong for
an input-adapter training experiment.

Source: [native recycling path](https://github.com/bytedance/Protenix/blob/85767b811c40ed46e73a9b39519cf6bfca8701ba/protenix/model/protenix.py),
[input features](https://github.com/bytedance/Protenix/blob/85767b811c40ed46e73a9b39519cf6bfca8701ba/protenix/model/modules/embedders.py),
[Pairformer/MSA modules](https://github.com/bytedance/Protenix/blob/85767b811c40ed46e73a9b39519cf6bfca8701ba/protenix/model/modules/pairformer.py).

## What may improve, and what may not

- Pretrained geometric/triangular representations may need far fewer teacher
  examples to learn our contact task. This is the main reason to run the pilot.
- Sequence-only deployment is feasible, but deleting pretrained MSA/template
  inputs causes distribution shift. At the terminal diffusion timestep the
  random contact map supplies no target information, so the model must still
  infer useful structure from sequence alone. Accurate denoising with lightly
  corrupted targets does not establish good unconditional rollouts.
- Freezing trunk weights saves parameter-gradient and optimizer-state costs.
  It does **not** eliminate backpropagation through the trunk: gradients must
  reach the trainable adapter at its input. `no_grad()` or cached trunk outputs
  would prevent that learning. Activation memory and input-gradient computation
  remain. Partial unfreezing and LoRA have the same basic limitation.
- Native optimized triangle kernels could affect throughput. Upstream currently
  pins PyTorch 2.7.1 and cuEquivariance 0.8.0, whereas Sparsa uses PyTorch 2.10.0.
  Use an isolated environment, first verify the native torch path, then benchmark
  compatible fused kernels with gradients. No measured speedup is claimed.
- Pretraining may produce a strong but low-diversity predictor. Score oracle@100,
  consensus R-precision, and the oracle uplift over one rollout separately;
  neither upstream coordinate benchmarks nor training loss answer this question.

## Bounded pilot to decide whether to scale

1. Verify native checkpoint key/shape coverage, sequence-only feature construction,
   padding, output symmetry and nonzero adapter gradients. Audit pretrained-data
   overlap with the frozen validation set before interpreting improvements.
2. Train adapter + new head with the v1 trunk frozen for a short 1,000-step warmup.
   This is a stability/transfer probe, not a claim that freezing is optimal.
3. Branch the same warmup checkpoint into two 5,000-step pilots: adapter/head only,
   and adapter/head plus full-trunk fine-tuning at a lower backbone learning rate.
   Suggested starting rates: adapter/head 2e-4, backbone 1e-5; tune only on eval-val.
4. Use the same global batch 128, crop 384, teacher mix, and binary schedule. Run
   full 97-protein, 100-rollout validation at the pilot boundaries. Compare both
   at matched new teacher exposures and measured total GPU-hours, including the
   warmup. Report original pretraining as additional compute/data, not as free
   from-scratch training.
5. If v1 shows a clear gain, run a native-architecture random-init control to
   separate transfer from architecture/kernel effects, then commit to a longer
   fine-tune. Consider partial unfreezing if full fine-tuning erases useful
   representations. Consider v2 only after v1 demonstrates transfer and its
   weight terms are resolved for the intended work.

Use `protenix_base_default_v1.0.0`, not the newer-data
`protenix_base_20250630_v1.0.0`. The 97 scorable validation structures in our
frozen manifest were released between 2023-01-18 and 2024-08-21, after the
default cutoff and before the newer checkpoint's cutoff. A 2021 cutoff helps
but is not proof of
sequence/homology or distillation-data disjointness; do an overlap audit. All
selection stays on the fixed validation split. Held-out experiments remain
separate from improvement decisions.

## Weight terms

The upstream README's general license section says code and model parameters
are Apache 2.0 and describes v1 as fully open source. Its newer, specific v2
notice says v2 weights are proprietary, not under an open-source license, and
restricts reproduction/distribution/transfer without express prior written
consent. The README therefore must not be summarized as granting Apache rights
to v2 weights. Start with v1; review the specific v2 terms before adopting it or
publishing derived weights. This is a description of the published notices,
not a legal determination.

Source: [upstream README, v2 release notice and license section](https://github.com/bytedance/Protenix/blob/85767b811c40ed46e73a9b39519cf6bfca8701ba/README.md).
