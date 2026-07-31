"""Bit-exact parity of the reproduced 2,754 descriptor vs the committed oracle .npy files.

The 4k38 oracle is the full-config (2,754) reference. (The committed 2fns .npy is a reduced-config
954-wide run and is not a full-width oracle, so it is not asserted here.)
"""
import os

import numpy as np
import pytest

from open_topodockq_featurizer.featurize import featurize_interface, RAW_WIDTH

HOME = os.path.expanduser("~")
IFACE = f"{HOME}/projects/repositories/TopoDockQ-Feature/data/interface_files"
ORACLE_4K38 = f"{HOME}/projects/repositories/TopoDockQ/feature/feature_4k38_ranked_44_sp_interface.npy"


@pytest.mark.skipif(not os.path.exists(ORACLE_4K38), reason="oracle fixture not present")
def test_4k38_full_vector_bit_exact():
    ours = featurize_interface(f"{IFACE}/4k38_ranked_44_sp_interface.pdb")
    oracle = np.load(ORACLE_4K38)[0]
    assert ours.shape == (RAW_WIDTH,)
    assert oracle.shape == (RAW_WIDTH,)
    maxdiff = np.abs(ours - oracle).max()
    # actual residual is ~2e-12 (float noise); a tight bound catches an eigenvalue-threshold
    # swap (which would perturb a Betti-0 count / nz.min by ~1e-8), not just gross regressions.
    assert maxdiff < 1e-9, f"max abs diff {maxdiff}"
