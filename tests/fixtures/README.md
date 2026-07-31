# Test fixtures

## `oracle_2fns_full_2754.npy`

Full-config (2,754-wide) oracle feature vector for the `2fns_ranked_3_sp_interface` complex.

**Provenance:** produced by **running** the released (unlicensed) upstream featurizer `.pyc` as a
black box — `python -m main --pdb_id 2fns --model_id 3 --bins <34-edge list> --filtration <33-point
list>` in a py3.8.20 env with the upstream pins (gudhi 3.8.0, numpy 1.24.3, scikit-learn 1.3.0). The
`.pyc` was **never decompiled**; this file is its numerical *output* (facts), used here only as a
comparison oracle. The full-config args were validated by first reproducing the committed `4k38`
scorer-repo vector bit-exactly with the same args.

Upstream ships only a *reduced-config* 954-wide `2fns` vector, so this full-width oracle had to be
minted to get a second full-config parity fixture beyond `4k38`.

**Exact full-config `.pyc` CLI** (validated to reproduce the committed `4k38` scorer-repo vector
bit-exact, max diff 1.8e-12):

```
python -m main --pdb_id <ID> --model_id <N> \
  --bins       "[0, 2.0, 2.25, 2.5, ..., 10.0]"   # 34 numbers: 0 + arange(2.0,10.25,0.25) -> 33 betti bins
  --filtration "[2.0, 2.25, 2.5, ..., 10.0]"       # 33 numbers: arange(2.0,10.25,0.25) (no leading 0)
  --file_path <dir with <ID>_ranked_<N>_sp_interface.pdb> --saving_path <out>
```

The leading `0` belongs to `--bins` only (it is the first histogram edge -> bin 0 = `[0, 2.0)`);
adding it to `--filtration` yields a spurious 34th spectra block (2,817 wide).

## `probes/` -- synthetic atypical-input probes

Six hand-built interface PDBs (`probes/gen.py`) that exercise paths the real singlePPD corpus never
reaches, each paired with a `.pyc`-minted oracle `.npy` (same black-box provenance as above). Built
with jittered irrational coordinates so no distance lands exactly on the 0.25 filtration grid (an
exact tie would expose the petls `<= t` vs upstream `< t` edge-inclusion convention -- irrelevant to
real, continuous distances). Consumed by `tests/test_probes.py`. These caught four featurizer bugs
that real-data parity could not: betti first edge (0.0 not 1.75), HETATM must be ignored, empty-side
channels emit a zero block, and single-interval alpha must be computed not zeroed.
