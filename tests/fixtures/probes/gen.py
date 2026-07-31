"""Generate synthetic interface PDBs to probe atypical-input paths (P2/P3 empty-side channels,
P5 HETATM/blank-element parsing, single-atom side) against the upstream .pyc black box.
Writes <name>_ranked_0_sp_interface.pdb into this dir.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def atom(serial, name, res, chain, resseq, xyz, element, record="ATOM", blank_elem=False):
    el = "  " if blank_elem else f"{element:>2}"
    # PDB fixed columns; name left-justified from col 14 for 1-2 char names (matches real files' "N  ").
    nm = f" {name:<3}" if len(name) <= 3 else name[:4]
    return (f"{record:<6}{serial:>5} {nm}{'':1}{res:>3} {chain}{resseq:>4}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{0.0:6.2f}          {el}")


_JIT = np.random.RandomState(20260731)


def grid(n, origin, step=1.61803):
    """n points on a line from origin, ~step apart, with deterministic irrational jitter so no
    pairwise distance lands exactly on a 0.25 filtration grid point (real interface distances are
    likewise generic; exact-grid ties are a synthetic artifact that the petls `<= t` vs upstream
    `< t` edge-inclusion convention would otherwise expose). Cross-cloud distances stay in ~2-10A."""
    pts = []
    for i in range(n):
        j = _JIT.uniform(-0.29, 0.29, size=3)
        pts.append((origin[0] + i * step + j[0], origin[1] + (i % 3) * 0.7 + j[1],
                    origin[2] + (i % 2) * 0.5 + j[2]))
    return pts


def write(name, records):
    lines = []
    s = 1
    for (nm, res, chain, xyz, el, rec, blank) in records:
        lines.append(atom(s, nm, res, chain, 1 + s % 20, xyz, el, rec, blank))
        s += 1
    lines.append("TER")
    with open(os.path.join(HERE, f"{name}_ranked_0_sp_interface.pdb"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return name


def cloud(chain, elements_counts, origin, record="ATOM", blank=False):
    """Build records: dict element->count, placed as a cloud at origin."""
    recs = []
    off = 0
    for el, cnt in elements_counts.items():
        for p in grid(cnt, (origin[0], origin[1] + off, origin[2])):
            recs.append((el, "ALA", chain, p, el, record, blank))
        off += 3.0
    return recs


# --- Probe P0: sanity — normal small structure, both sides all elements (should match trivially)
p0 = cloud("A", {"C": 6, "N": 4, "O": 3}, (0, 0, 0)) + cloud("B", {"C": 5, "N": 3, "O": 2}, (4.0, 0, 0))
write("psane", p0)

# --- Probe P2E: protein has NO O; peptide has NO N  -> multiple empty element-channels
p2 = cloud("A", {"C": 6, "N": 5}, (0, 0, 0)) + cloud("B", {"C": 5, "O": 4}, (4.0, 0, 0))
write("pempty", p2)

# --- Probe P1A: single-atom peptide side (peptide = 1 C atom); protein normal
p1 = cloud("A", {"C": 6, "N": 4, "O": 3}, (0, 0, 0)) + [("C", "ALA", "B", (4.0, 0, 0), "C", "ATOM", False)]
write("ponepep", p1)

# --- Probe HET: same as psane but peptide atoms are HETATM records
p_het = cloud("A", {"C": 6, "N": 4, "O": 3}, (0, 0, 0)) + cloud("B", {"C": 5, "N": 3, "O": 2}, (4.0, 0, 0), record="HETATM")
write("phet", p_het)

# --- Probe BLANK: same as psane but ALL element columns blank (element must be inferred, if at all)
p_blank = cloud("A", {"C": 6, "N": 4, "O": 3}, (0, 0, 0), blank=True) + cloud("B", {"C": 5, "N": 3, "O": 2}, (4.0, 0, 0), blank=True)
write("pblank", p_blank)

# --- Probe EXTRA: psane + extra chain C atoms + H/S atoms (should be ignored by both)
p_extra = (cloud("A", {"C": 6, "N": 4, "O": 3}, (0, 0, 0)) + cloud("B", {"C": 5, "N": 3, "O": 2}, (4.0, 0, 0))
           + cloud("C", {"C": 4}, (8.0, 0, 0)) + cloud("A", {"H": 5, "S": 2}, (0, 10, 0)))
write("pextra", p_extra)

print("wrote:", sorted(f for f in os.listdir(HERE) if f.endswith(".pdb")))
# show a couple lines for format sanity
with open(os.path.join(HERE, "psane_ranked_0_sp_interface.pdb")) as fh:
    for ln in list(fh)[:2]:
        print(repr(ln.rstrip("\n")))
