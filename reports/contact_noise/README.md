# Ground-truth and noised contact maps

Three fully resolved proteins from the fixed validation split, chosen at roughly
100, 200 and 300 residues: `7y54_A`, `8baq_A`, `8daj_A`. No model predictions or
held-out proteins are used. The maps use the current Protenix pilot's eight-step
schedule, which is also used by the current Pairformer experiments.

Each panel is an independent draw using the production
`BinaryDiffusion.sample_forward` training sampler. It shows a draw from
`q(x_t | x_0)` at the indicated level, rather than consecutive states of one
forward trajectory. Ground truth uses MarinFold's native-amino-acid ConFind
degree >= 0.001 and sequence separation >= 6.

For an eligible pair with ground-truth state `x0`:

```
P(xt = 1 | x0) = alpha_bar[t] * x0 + (1 - alpha_bar[t]) * p(separation)
```

The cosine schedule's original-map weights are 100%, 95.8%, 84.7%, 68.4%, 49.4%,
30.4%, 14.4%, 3.7%, and 0%. The separation prior is 2.5% for separations 6–11,
1.6% for 12–23, and 0.5% for 24+. At `t=8`, the draw is independent of ground
truth. Blue contacts at that level are chance overlaps.

Blue means a displayed contact is also in ground truth; orange means it is
absent from ground truth. White means no displayed contact. Gray marks excluded
pairs (separation < 6, or an unresolved residue). Maps are symmetric, axes are
one-based sequence positions, and counts count each undirected pair once.

- `*_noise.png`: full-size 3 × 3 panels for each protein.
- `contact_noise_overview.png`: all proteins and levels in one figure.
- `contact_noise.pdf`: three-page version for export.
- `*_maps.npz`: exact binary maps `[t, i, j]`, eligible-pair mask and schedule.
- `contact_counts.csv`: ground-truth/shared/added/missing counts per level.
- `provenance.json`: seeds, input/code hashes, schedule and split.

Reproduce from the repository root with matplotlib installed:

```bash
PYTHONPATH=. .venv/bin/python scripts/plot_contact_noise.py
```

Use `--proteins`, `--seed`, `--config` and `--out` to change the inputs.
