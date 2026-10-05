# Sparsa

Protein residue-contact prediction from amino-acid sequence alone. We train
models from scratch on [MarinFold](https://github.com/Open-Athena/MarinFold)'s
curated AF2 and ESMFold2 contact maps, without MSAs or pretrained protein language
model features as inputs.

The goal is to find architectures that improve contact R-precision and compare
them with MarinFold's autoregressive LLM approach. Predictions can be passed to
[Helico](https://github.com/Open-Athena/helico) for structure generation.

- **Baseline:** a 40M-parameter bidirectional sequence encoder with a symmetric
  pair network using convolutions and triangle multiplication.
- **Architecture search:** Codex proposes model code changes; isolated Iris jobs
  train candidates at **batch priority**. Experiments use matched training
  exposure, two-seed confirmation, a bounded compute allocation, and a persistent
  results ledger.
- **Evaluation:** MarinFold's frozen experimental splits and contact definition. Contacts are native-amino-acid ConFind degree >= 0.001 at sequence
  separation >= 6. Only the 97 validation proteins guide search; the 217 test and
  19 de novo proteins were held out until final model selection.

The first training and evaluation campaign is complete. R-precision:

| Experimental split | Proteins | Sparsa | MarinFold¹ |
|---|---:|---:|---:|
| Validation | 97 | 0.24772 | 0.51980 |
| Test | 217 | 0.25405 | 0.53765 |
| De novo | 19 | 0.50527 | 0.59138 |

¹ Decontaminated exp232 m2-p06, step 145199. Sparsa remains substantially worse;
model size, training compute, and readout differ. Eight architecture-search trials
completed; none met the promotion rule. See [results and checkpoint](reports/FINAL.md)
and [search results](reports/AUTORESEARCH.md). See also the longer
[scaling results](reports/scaling_v2/FINAL.md) and current
[pair-trunk experiments](reports/pair_trunk_v1/RUNNING.md).

Current diffusion evaluations use **fixed-R precision**: rank all eligible pairs
by final denoiser probabilities for oracle best-of-100, and by rollout occurrence
counts for consensus. Compare consensus with MarinFold consensus, and oracle
with its oracle reference. Earlier diffusion oracle plots used an incorrect
denominator for sparse maps; see the [metric correction](reports/diffusion_fixed_r/README.md).

The current experiment uses a **4-layer sequence encoder and 48-block Pairformer**.
Four arms compare pair width 128/256 and sequence-only cached conditioning versus
injecting the noisy contact map before the first Pairformer block. Previous
G2/G3/G4 runs were retired. See the [new campaign](reports/pairformer_v1/PLAN.md).

## Use

```bash
uv sync
uv run pytest -q
uv run sparsa predict --checkpoint /path/to/checkpoint.pt \
  --sequence ACDEFGHIKLMNPQRSTVWY --out outputs/example --top-l 1
```

Prediction writes a dense score matrix and a zero-based, positive-only Helico
contact list. Architecture-search checkpoints require their accompanying source
snapshot. Training uses Iris or dedicated CUDA nodes and the curated teacher data.

[Architecture search and controls](research/README.md) ·
[Training and evaluation guide](docs/usage.md) ·
[Benchmark provenance](data/benchmark/provenance.json) ·
[Discrete-diffusion screen](reports/diffusion_v1/RESULTS.md) ·
[Long diffusion campaign](reports/diffusion_v2/PLAN.md)
