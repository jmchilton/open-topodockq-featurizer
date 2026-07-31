"""Bit-exact parity on synthetic probes that exercise atypical-input paths absent from real data.

The full singlePPD corpus (and the 4k38/2fns fixtures) is uniformly clean ATOM-only, both sides
well-populated, no HETATM -- so at-scale parity on real structures cannot reach these paths. Each
probe below was hand-built to hit one, and its oracle .npy was minted by running the upstream `.pyc`
black box on the same synthetic PDB (never decompiled; see fixtures/probes/gen.py for construction
and fixtures/README.md for provenance).

These caught four featurizer bugs invisible to real-data parity:
  - psane/pextra : betti first bin edge is 0.0, not 1.75 (real distances never fall below ~2.3A)
  - phet         : the .pyc ignores HETATM records (we must too)
  - pempty/pblank: a channel with an empty side must emit a zero block, not raise
  - pextra       : alpha stats must be computed for a single interval, not zeroed
"""
import os

import numpy as np
import pytest

from open_topodockq_featurizer.featurize import featurize_interface, RAW_WIDTH

PROBES = os.path.join(os.path.dirname(__file__), "fixtures", "probes")
TOL = 1e-9

# what each probe exercises (documented so a failure names the path, not just the file)
PROBES_WHAT = {
    "psane": "baseline: both sides all elements populated",
    "pempty": "empty element-channels (protein has no O; peptide has no N)",
    "ponepep": "single-atom peptide side",
    "phet": "peptide atoms as HETATM (must be ignored -> all-zero vector)",
    "pblank": "blank element column (must be skipped -> all-zero vector)",
    "pextra": "extra non-A/B chain + H/S atoms (must be ignored); single-interval alpha",
}


@pytest.mark.parametrize("name", sorted(PROBES_WHAT))
def test_probe_bit_exact(name):
    pdb = os.path.join(PROBES, f"{name}_ranked_0_sp_interface.pdb")
    oracle = np.load(os.path.join(PROBES, f"feature_{name}_ranked_0_sp_interface.npy"))[0]
    ours = featurize_interface(pdb)
    assert ours.shape == (RAW_WIDTH,)
    maxdiff = np.abs(ours - oracle).max()
    assert maxdiff < TOL, f"{name} ({PROBES_WHAT[name]}): max abs diff {maxdiff}"
