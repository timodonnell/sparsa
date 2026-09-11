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
- **Evaluation:** MarinFold's frozen experimental splits and unmodified scoring
  code. Contacts are native-amino-acid ConFind degree >= 0.001 at sequence
  separation >= 6. Only the 97 validation proteins guide search; the 217 test and
  19 de novo proteins are reserved for final evaluation.

Research is in progress. The initial model has not matched MarinFold's validation
accuracy. Matching the evaluation protocol does not equalize model size or
training compute.

## Use

```bash
uv sync
uv run pytest -q
uv run sparsa predict --checkpoint /path/to/checkpoint.pt \
  --sequence ACDEFGHIKLMNPQRSTVWY --out outputs/example --top-l 1
```

Prediction writes a dense score matrix and a zero-based, positive-only Helico
contact list. Architecture-search checkpoints require their accompanying source
snapshot. Training requires Iris and access to the curated CoreWeave S3 data.

[Architecture search and controls](research/README.md) ·
[Training and evaluation guide](docs/usage.md) ·
[Benchmark provenance](data/benchmark/provenance.json)
