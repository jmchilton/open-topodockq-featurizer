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
