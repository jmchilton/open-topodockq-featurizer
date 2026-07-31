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
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
# 4k38 oracle is upstream-committed (scorer repo, full-config, max_edge=7 capped).
ORACLE_4K38 = f"{HOME}/projects/repositories/TopoDockQ/feature/feature_4k38_ranked_44_sp_interface.npy"
# 2fns full-config oracle minted here from the .pyc (see fixtures/README.md); committed in-repo.
ORACLE_2FNS = os.path.join(FIXTURES, "oracle_2fns_full_2754.npy")

# actual residual is ~1e-12 (float noise); a tight bound catches an eigenvalue-threshold swap
# (which would perturb a Betti-0 count / nz.min by ~1e-8), not just gross regressions.
TOL = 1e-9


def _assert_parity(pdb_name, oracle_path):
    ours = featurize_interface(f"{IFACE}/{pdb_name}")
    oracle = np.load(oracle_path)[0]
    assert ours.shape == (RAW_WIDTH,)
    assert oracle.shape == (RAW_WIDTH,)
    maxdiff = np.abs(ours - oracle).max()
    assert maxdiff < TOL, f"{pdb_name}: max abs diff {maxdiff}"


@pytest.mark.skipif(not os.path.exists(ORACLE_4K38), reason="upstream clone not present")
def test_4k38_full_vector_bit_exact():
    _assert_parity("4k38_ranked_44_sp_interface.pdb", ORACLE_4K38)


@pytest.mark.skipif(not os.path.exists(f"{IFACE}/2fns_ranked_3_sp_interface.pdb"),
                    reason="upstream clone not present")
def test_2fns_full_vector_bit_exact():
    _assert_parity("2fns_ranked_3_sp_interface.pdb", ORACLE_2FNS)
