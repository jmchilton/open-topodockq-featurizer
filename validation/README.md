# End-to-end scorer validation

Proves the reproduced featurizer drives the (open, MIT) TopoDockQ scorer to the same p-DockQ as the
upstream-shipped features, and that the open scorer pipeline reproduces paper-level test performance.

`run_scorer.py` mirrors upstream `04_inference_from_generated_csv.py` exactly — the training-derived
mask (drop columns constant across the train split → 2,754→2,646), a live `StandardScaler` fit on the
train split, and the `TopoDockQ` MLP + Zenodo `best_model.pth` — but is parameterized on the query
features CSV so our-featurizer output can be compared against the shipped `example_features.csv`.

## Prereqs (not committed — large / licensed)

From Zenodo record 15469415, staged into a local `XDaiNYU/TopoDockQ` clone:
- `trained_model.zip` → `models/best_model.pth`
- `processed_data.zip` → `data/processed_data/{singlePPD_full_bins_features.csv, singlePPD_DockQ.csv}`

## Run (from the TopoDockQ clone dir)

```bash
# 1. our featurizer -> raw 2,754 npy
python -c "import numpy as np; from open_topodockq_featurizer.featurize import featurize_interface; \
           np.save('ours.npy', featurize_interface('<interface>.pdb').reshape(1,-1))"
# 2. upstream reorder -> named CSV
python 03_extract_features_from_npy_to_csv.py --npy_file ours.npy --output_file ours03.csv
# 3. score (mask + scaler + MLP)
python <this>/run_scorer.py --features_csv ours03.csv
```

## Result (2026-07-31)

- Open scorer on shipped features: **TEST PCC 0.8773** (n=9464) vs `pdb2sql_DockQ`.
- Featurizer parity: our 4k38 features → p-DockQ **0.15463053**, bit-identical to the shipped
  `example_features.csv` prediction. 2fns → 0.20463052.
- Confirmed: mask drops exactly 108 training-constant columns → input_dim 2646.

## At-scale featurizer parity (`at_scale_parity.py`)

Featurizes a stratified sample of real interface PDBs from `singlePPD_interface_files.tar.gz`
(Zenodo 15469415, ~4 GB) and diffs our output — via upstream `03`'s reorder, imported unchanged — against
the matching `singlePPD_full_bins_features.csv` rows. Compares by **column name**, so a reorder bug can't
hide as a coincidental match.

```bash
# from the TopoDockQ clone dir (needs ./03_...py):
python <this>/at_scale_parity.py --interface_dir <extracted_pdbs> --features_csv <shipped_subset.csv>
```

**Result (2026-07-31):** 400 structures across **400 distinct PDB entries** →
**max abs diff 1.85e-11, median 1.4e-12, 0 errored, 0 over tol 1e-6.** Featurizer generalizes.

**Boundary:** all 400 (and both fixtures) are clean ATOM-only, both sides well-populated (≥17 peptide
atoms, no HETATM). So empty-side alpha, persistence-floor-empties, and HETATM/blank-element/non-A-B
parsing remain **unexercised by real data** — the singlePPD corpus contains no such cases. Closing them
needs synthetic probes against the upstream `.pyc`, not a larger corpus run.
