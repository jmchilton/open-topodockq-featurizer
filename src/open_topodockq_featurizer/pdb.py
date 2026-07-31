"""Load a TopoDockQ interface PDB into per-element point clouds.

The interface PDB (produced upstream by ``extract_interface.py``) holds two protein chains:
chain A = protein, chain B = peptide. The featurizer uses only C/N/O heavy atoms; H and S are
dropped (upstream ``atom_types = ['C', 'N', 'O']``).
"""

from __future__ import annotations

import numpy as np

ELEMENTS = ("C", "N", "O")
PROTEIN_CHAIN = "A"
PEPTIDE_CHAIN = "B"


def load_interface(path, protein_chain=PROTEIN_CHAIN, peptide_chain=PEPTIDE_CHAIN):
    """Return ``{element: {"protein": (n,3), "peptide": (m,3)}}`` of float64 coordinates.

    Parsed from fixed-column PDB records; element read from columns 77-78.
    """
    protein = {e: [] for e in ELEMENTS}
    peptide = {e: [] for e in ELEMENTS}
    with open(path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            element = line[76:78].strip().upper()
            if element not in ELEMENTS:
                continue
            chain = line[21]
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            if chain == protein_chain:
                protein[element].append(xyz)
            elif chain == peptide_chain:
                peptide[element].append(xyz)
    return {
        e: {
            "protein": np.asarray(protein[e], dtype=np.float64).reshape(-1, 3),
            "peptide": np.asarray(peptide[e], dtype=np.float64).reshape(-1, 3),
        }
        for e in ELEMENTS
    }
