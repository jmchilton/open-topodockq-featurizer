# open-topodockq-featurizer

An **open, clean-room reimplementation** of the TopoDockQ interface featurizer — the one piece of
[TopoDockQ](https://github.com/XDaiNYU/TopoDockQ) that ships only as unlicensed `.pyc` bytecode. Built on
[petls-pytorch](https://github.com/Sylverity/petls-pytorch) (Apache-2.0) + GUDHI, with **no `.pyc` in the
path**. Completes the open TopoDockQ vertical (bio-topo-foundry #3, pipeline P2).

## Provenance

The featurizer method was recovered by **black-box observation** of the upstream `.pyc` (CLI-arg sweeps,
instrumented I/O, controlled synthetic inputs) — **never decompiled**, and **not** copied from
`wangru25/TopoDockQ-Feature`. Behavior/method is not copyrightable; the constants below are facts read off
observed output. A prior feasibility spike reproduced all four feature blocks bit-exactly (9/9 channels,
2 PDBs) — see `topodockq-featurizer-spike.md` in the foundry repo.

## What it emits

A raw **2,754-value, channel-major** interface descriptor. 9 element channels (`CC, CN, CO, NC, NN, NO,
OC, ON, OO`), each a bipartite protein×peptide pairing, 306 values per channel:

```
[ 33 betti-death-bins | 6 Rips-H0 barcode stats | 231 spectra (7 stats × 33 filtrations) | 18 alpha-H1 | 18 alpha-H2 ]
```

The 7 per-filtration spectra stats: `[sum, min, max, mean, std(ddof=0), var(ddof=0)]` of the nonzero L0
eigenvalues, then Betti-0. Filtrations `arange(2.0, 10.25, 0.25)` (33 points).

### Pinned constants

`max_edge_length = 7.0` (H0/betti Rips) · intra-chain pad `= 100.0` · `cut = 7.0` (alpha clip) ·
persistence floor `= 0.1` · `2×` diameter scaling on alpha circumradii · float64 required.

## Relationship to the (open, MIT) scorer

The TopoDockQ **scorer is open MIT source** (`03_extract_features_from_npy_to_csv.py`,
`04_inference_from_generated_csv.py`, `src/model.py`, `src/train.py`). This project does **not** reimplement
it. It emits the raw 2,754 npy that upstream `03` reorders into named columns and `04` consumes. Note:

- **The 2,754→2,646 mask is training-derived**, applied inside `04` (drop columns constant across the
  training split). Not our concern to reproduce — but requires Zenodo `singlePPD_full_bins_features.csv`.
- **No scaler is serialized** — `04` fits `StandardScaler` live from the training CSV.

## Status

Featurizer **reproduced bit-exact** and validated **at scale**. `featurize_interface(interface_pdb)`
matches the upstream-shipped features on **400 real interface structures spanning 400 distinct PDB
entries** (Zenodo `singlePPD_interface_files.tar.gz` → our raw 2,754 → upstream `03` reorder → diff vs
`singlePPD_full_bins_features.csv`): **max abs diff 1.85e-11, median 1.4e-12, 0 errors, 0 over
tolerance**. Plus two curated fixtures in-repo (`4k38`, minted `2fns`). See `tests/test_parity.py` and
`validation/at_scale_parity.py`.

Still **unexercised** (not by fixtures nor by the 400 — the whole singlePPD corpus is uniformly clean
ATOM-only, chain A/B, both sides well-populated, ≥17 peptide atoms, no HETATM): channels with an
empty/near-empty side (alpha zeros-guard), the persistence-floor-empties path, and parsing of HETATM /
blank-element / non-A-B-chain inputs. The real data simply contains no such cases — closing these
requires **synthetic probes** against the upstream `.pyc`, not more corpus.

## Validation plan

1. **Local (no Zenodo):** bit-exact vs the committed `4k38` raw 2,754 npy — full assembled vector, correct
   channel order into upstream `03`. (The committed `2fns` npy is 954-wide reduced-config — not a full oracle.)
2. **End-to-end (acceptance gate):** our features → unchanged upstream `03→04` (+ Zenodo record 15469415
   training CSV, DockQ CSV, weights) → predicted p-DockQ matches the scorer on `example_features.csv`.
   **DONE:** 4k38 p-DockQ bit-identical; open scorer reproduces test PCC 0.8773 (n=9464). See `validation/`.
3. **At scale (DONE):** 400 diverse interface PDBs featurized and diffed against the shipped feature rows,
   all bit-exact — closes alpha_topo generalization and column identity/order beyond a couple of PDBs.
   Residual: the atypical-input paths above, absent from the corpus (need synthetic probes).

## License

MIT — see `LICENSE`. (Matches the TopoDockQ tool this completes; OSI/L3-eligible.)
