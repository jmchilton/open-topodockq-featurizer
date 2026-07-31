"""At-scale featurizer parity: featurize a sample of real interface PDBs and diff our output
against the upstream-shipped feature rows.

For each interface PDB `<pdb>_ranked_<model>_sp_interface.pdb`:
  featurize_interface -> raw (1,2754) -> upstream `03` reorder (imported unchanged) -> named cols
  -> diff vs the shipped `singlePPD_full_bins_features.csv` row keyed on (pdb_id, af_model_id).

Compares by COLUMN NAME (not position), so a reorder bug can't hide as a coincidental match.
Run from the TopoDockQ clone dir (needs `03_extract_features_from_npy_to_csv.py` on disk).
"""
import argparse
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd

from open_topodockq_featurizer.featurize import featurize_interface

# Import upstream 03 unchanged (module name starts with a digit -> load by path).
_spec = importlib.util.spec_from_file_location("extract03", "./03_extract_features_from_npy_to_csv.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

NAME_RE = re.compile(r"^(?P<pdb>[^_]+)_ranked_(?P<model>\d+)_sp_interface\.pdb$")


def reorder_named(raw):
    """Replicate 03's __main__ reorder exactly, returning {colname: value}. raw is (1,2754)."""
    ex = _m.ExtractFeatures(_m.atom_types, _m.total_atom_combination_features)
    combos = _m.atom_combinations
    filtration = np.round(np.arange(2, 10.25, 0.25), 2)
    out = {}
    for fv in _m.filtration_list:
        for i, ci in enumerate(combos):
            pf = ex.extract_features_by_filtration(raw, fv, filtration, ci).ravel()  # 8 values
            for k in range(8):
                out[f"persistent_{fv}_{str(i * 8 + k + 1).zfill(2)}"] = pf[k]
    col = 1
    for ci in combos:
        sf = ex.extract_static_features(raw, ci).ravel()  # 42 values
        for v in sf:
            out[f"static_{str(col).zfill(2) if col < 100 else str(col)}"] = v
            col += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface_dir", required=True)
    ap.add_argument("--features_csv", required=True, help="subset of shipped rows (small)")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ship = pd.read_csv(args.features_csv)
    ship_idx = ship.set_index(["pdb_id", "af_model_id"])
    feat_cols = [c for c in ship.columns if c.startswith(("persistent_", "static_"))]
    print(f"shipped rows={len(ship)}  feature cols={len(feat_cols)}")

    pdbs = sorted(f for f in os.listdir(args.interface_dir) if f.endswith("_interface.pdb"))
    if args.limit:
        pdbs = pdbs[: args.limit]
    print(f"interface pdbs to test={len(pdbs)}")

    results, worst, missing, errored = [], [], [], []
    for n, fn in enumerate(pdbs):
        mo = NAME_RE.match(fn)
        if not mo:
            missing.append((fn, "unparsable name"))
            continue
        pdb, model = mo["pdb"], int(mo["model"])
        try:
            row = ship_idx.loc[(pdb, model)]
        except KeyError:
            missing.append((fn, "no shipped row"))
            continue
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        try:
            raw = featurize_interface(os.path.join(args.interface_dir, fn)).reshape(1, -1)
            named = reorder_named(raw)
        except Exception as e:  # noqa: BLE001 -- want to surface any structure that breaks us
            errored.append((fn, repr(e)))
            continue
        ours = np.array([named[c] for c in feat_cols], dtype=np.float64)
        theirs = row[feat_cols].to_numpy(dtype=np.float64)
        d = np.abs(ours - theirs)
        mad = float(np.nanmax(d))
        results.append(mad)
        if mad > args.tol:
            j = int(np.nanargmax(d))
            worst.append((fn, mad, feat_cols[j], float(ours[j]), float(theirs[j])))
        if (n + 1) % 25 == 0:
            print(f"  ...{n + 1}/{len(pdbs)}  running max-abs-diff={max(results):.2e}")

    print("\n===== SUMMARY =====")
    print(f"compared        : {len(results)}")
    print(f"errored         : {len(errored)}")
    print(f"missing/unmapped: {len(missing)}")
    if results:
        arr = np.array(results)
        print(f"max abs diff    : {arr.max():.3e}")
        print(f"median abs diff : {np.median(arr):.3e}")
        print(f"# over tol={args.tol:g} : {int((arr > args.tol).sum())}")
    for fn, e in errored[:20]:
        print(f"  ERROR {fn}: {e}")
    for w in sorted(worst, key=lambda x: -x[1])[:20]:
        print(f"  OVER-TOL {w[0]} mad={w[1]:.3e} col={w[2]} ours={w[3]:.6g} theirs={w[4]:.6g}")
    for fn, why in missing[:10]:
        print(f"  SKIP {fn}: {why}")

    sys.exit(1 if (errored or (results and np.array(results).max() > args.tol)) else 0)


if __name__ == "__main__":
    main()
