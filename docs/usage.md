# Training, prediction, and evaluation

Protein residue contacts from amino-acid sequence alone. Sparsa is trained from
scratch on MarinFold's curated AFDB and ESM-Atlas teacher contact maps, with no
MSAs, pretrained protein language models, templates, or evolutionary features.
The first training and experimental evaluation campaign is complete. See
[measured results and checkpoint](../reports/FINAL.md).

The model combines a bidirectional rotary-attention sequence encoder with a
symmetric pair network: sequence pair products, relative positions, dilated 2D
convolutions, and gated triangle multiplication. It directly predicts a dense
matrix, avoiding autoregressive contact-list generation.

## Automated architecture research

The [autoresearch controller](../research/README.md) can continually propose model
code changes, train them on Iris at batch priority, and retain improvements in
validation R-precision. It uses isolated source snapshots, matched training
exposure, two-seed confirmation, a compute allocation, and a persistent
leaderboard. Test and de novo sets are excluded from the search. See
[research/program.md](../research/program.md) for the proposal instructions and
[research/protocol.yaml](../research/protocol.yaml) for the initial search budget.

## Setup and use

```bash
uv sync
uv run pytest -q

# Training checkpoints include optimizer/RNG state for resume.
# The released sparsa.pt contains inference weights and their model contract.
uv run sparsa predict --checkpoint /path/to/step-N.pt \
  --sequence ACDEFGHIKLMNPQRSTVWY --out outputs/example --top-l 1

# Unlisted pairs remain UNKNOWN, as expected by Helico.
helico-infer --checkpoint /path/to/helico.pt \
  --sequences A:ACDEFGHIKLMNPQRSTVWY \
  --contacts outputs/example/helico_contacts.txt --output outputs/fold.pdb
```

`contacts.npz` contains `score`, `valid`, and `sequence`. Scores are symmetric;
`valid` excludes the diagonal and sequence separations below six. The plain-text
Helico contact list uses zero-based indices and includes only positive contacts.
`--top-l` controls how many pairs are passed downstream; it never uses a target
structure's true contact count. This is a contact budget, not a calibrated
confidence threshold. Scores apply the analytic correction for weighted BCE,
which does not by itself establish empirical probability calibration.

## Training

```bash
uv run python scripts/submit.py --name my-run --config configs/sequence_pair.yaml \
  --gpus 8 --timeout 43200
```

The launcher submits one root Iris job explicitly at **batch priority**. It uses
single-node PyTorch DDP with a dynamically assigned rendezvous port, bf16,
optional activation checkpointing, AdamW, gradient clipping, and EMA validation. Raw
teacher shards and checkpoints live in CoreWeave S3 alongside the GPU pool.
The local `.tools/iris` installation is an isolated launcher environment; pass
`--iris` and `--cluster-config` on another workstation.

Teacher inputs are the existing exp232 mirrors of exp225's decontaminated
corpora: 3,963,003 AFDB documents and 65,553,178 ESM documents before Sparsa's
filters. Sampling mixes the sources 50/50 by protein. Sparsa rejects truncated
contact lists because their unlisted pairs are not valid negative labels. It
also excludes sequences shorter than 12 residues. Random contiguous crops are
used during training; evaluation uses full sequences. These differences must be
reported in comparisons with the original LLM training recipe.

Every run records configuration, parameter count, teacher shard inventory,
benchmark hashes, training loss, validation metrics, and durable step-numbered
checkpoints. Resume restores compact data cursors and RNG state with unchanged world size,
workers, crop, and batch settings. It reloads only the current teacher shards;
prefetched but unconsumed batches are replayed without skipping training data.

After preemption, the completion file's elapsed time covers the final attempt.
Use `scripts/job_accounting.py` with the isolated Iris Python environment to
collect per-attempt running time across retries; its report excludes queue waits
and distinguishes this resource-time estimate from billing or GPU utilization.

## Evaluation contract

The target is **native-amino-acid ConFind degree >= 0.001 at sequence separation
>= 6**, following the actual MarinFold scoring code. It is not an 8 Å distance
contact definition. Experimental structures supervise evaluation only. Unknown
or unresolved residues do not become negative labels.

The frozen exp245 comparison includes 97 `eval-val`, 217 `eval-test`, and 19
`eval-denovo` proteins. The original context-budget exclusion of `8uxt_A` is
retained for comparability. Only `eval-val` may guide architecture and checkpoint
selection. The test and de novo splits are scored for the selected checkpoint.

```bash
uv run sparsa evaluate --checkpoint /path/to/selected.pt \
  --split eval-val --split eval-test --split eval-denovo --out outputs/evaluation
uv run python scripts/compare.py --predictions outputs/evaluation/per_protein.csv \
  --out outputs/evaluation/paired_comparisons.csv
```

Evaluation imports the **unmodified** MarinFold exp89 metric implementation:
R-precision, precision@L, L/2, L/5, and ROC-AUC over all, short, medium, and long
ranges. Stable sorting, resolved-residue candidate pairs, zero-contact cases, and
macro-averaging match upstream. Comparison checks candidate/positive/top-k counts
for every protein before paired bootstrap intervals. Baseline identities are
pinned, including whether their training data was decontaminated; they are not
silently relabeled as the latest MarinFold release.

The later exp232 step-363000 reference is included separately for the 97
validation and 19 de novo proteins; its saved results do not cover this test set.
After training, `scripts/final_evaluate.py` rechecks the best original readout and
evaluates a cap at the largest trained separation at every validated checkpoint,
using validation only. It also tests a uniform EMA weight average of the two
best original checkpoints, with both original and capped readouts. An averaged
checkpoint still uses one model at inference, and its source checkpoints and
weights are preserved in the export. It saves the selected checkpoint and readout before
scoring held-out sets. `scripts/select_readout.py` can run this validation-only
selection separately without scoring test or de novo proteins.

The campaign includes a short crop-1024 finetune after the main crop-384 run,
so the model can learn contacts beyond the main crop's separation limit. Unseen
distance bins initially copy the trained edge bin, reproducing the validated
capped readout while allowing new distances to learn separately. The
final evaluator compares the completed runs on validation and retains the best
candidate before scoring held-out sets. Both runs' training provenance and
validation trajectories accompany the export.

To verify a recovered checkpoint through the actual prediction CLI and Helico's
installed contact parser, use Helico's Python environment:

```bash
uv run python -m scripts.verify_release --checkpoint /path/to/sparsa.pt \
  --helico-python /path/to/helico/.venv/bin/python --out outputs/release-check
```

This uses one validation sequence on CPU, checks the score matrix and ranked
top-L export, and confirms that Helico accepts the zero-based contacts while
leaving all unlisted pairs unknown. `verification.json` records checkpoint and
source hashes. This is an interoperability check, not a structure prediction or
an accuracy evaluation. The released checkpoint passed both examples below.
Use `--validation-protein 7znz_A` to check the longest validation chain (761
residues); stems outside the validation split are rejected.

See `data/benchmark/provenance.json` for upstream revisions and artifact hashes,
`WORK_PLAN.md` for execution stages, and `PLAN.md` for the longer-term discrete
contact-completion research roadmap. Configuration, validation, and artifact
interfaces are reusable by a future automated experiment loop. The initial
implementation does not claim to implement diffusion or arbitrary-contact
completion.
