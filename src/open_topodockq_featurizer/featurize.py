"""Reproduce the TopoDockQ interface featurizer: interface PDB -> raw 2,754 descriptor.

Clean-room, on GUDHI + the petls-pytorch v2 fork; no upstream ``.pyc``. Layout and conventions
recovered by black-box observation of the released ``.pyc`` (see repo README / provenance).

Per element channel (9 of them, protein-element x peptide-element in order CC, CN, CO, NC, NN, NO,
OC, ON, OO), 306 values in order::

    [ 33 betti-death-bins | 6 Rips-H0 barcode stats | 231 spectra (7 x 33 filtrations) | 18 alpha-H1 | 18 alpha-H2 ]
"""

from __future__ import annotations

import numpy as np
import gudhi
import torch

# float64 is required: float32 puts harmonic eigenvalues at ~1e-4 and corrupts the L0 nullity
# (Betti-0). `_config` is a private surface of the pinned petls-pytorch v2 fork — the dtype must be
# set before `Complex` is imported, hence the deferred import below (and the E402s). If a fork bump
# moves `_config`, this is the line to update.
from petls_pytorch import _config

_config.set_dtype(torch.float64)
_config.set_sparse_dtype(torch.float64)
from petls_pytorch.core.complex import Complex  # noqa: E402

from .pdb import ELEMENTS, load_interface

# --- pinned constants (recovered by observation) ---
PAD = 100.0                                          # intra-chain distance padding (any value > MAX_EDGE)
MAX_EDGE = 7.0                                       # Rips cap for betti/barcode-A blocks
FILTRATIONS = np.round(np.arange(2.0, 10.25, 0.25), 2)   # 33 spectra filtration points
# 34 edges -> 33 death-bin intervals. First edge is 0.0 (the upstream `--bins` arg is
# `[0, 2.0, 2.25, ..., 10.0]`): bin 0 = [0, 2.0). Real interface distances never fall below ~2.3A
# so this is indistinguishable from a 1.75 first edge on real data, but a synthetic probe with
# sub-2.0 cross distances proved the true edge is 0.0.
BETTI_EDGES = np.round(np.concatenate([[0.0], np.arange(2.0, 10.25, 0.25)]), 2)
ALPHA_CUT = 7.0                                      # diameter clip for alpha_topo
ALPHA_FLOOR = 0.1                                    # persistence floor for alpha_topo
EIG_TOL = 1e-9                                       # zero-eigenvalue threshold

PER_CHANNEL_WIDTH = 306
RAW_WIDTH = 2754


def _bipartite_distance_matrix(protein_pts, peptide_pts):
    """Full (n+m) distance matrix: cross distances real, intra-chain = ``PAD``."""
    n, m = len(protein_pts), len(peptide_pts)
    d = np.full((n + m, n + m), PAD, dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    if n and m:
        cross = np.linalg.norm(protein_pts[:, None, :] - peptide_pts[None, :, :], axis=2)
        d[:n, n:] = cross
        d[n:, :n] = cross.T
    return d


def _h0_finite_deaths(distance_matrix, max_edge):
    rc = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge)
    st = rc.create_simplex_tree(max_dimension=1)
    st.compute_persistence()
    h0 = st.persistence_intervals_in_dimension(0)
    if h0.size == 0:
        return np.empty(0)
    return h0[np.isfinite(h0[:, 1]), 1]


def betti_bins(distance_matrix):
    """33 death-bin counts on the bipartite Rips capped at ``MAX_EDGE``."""
    deaths = _h0_finite_deaths(distance_matrix, MAX_EDGE)
    hist, _ = np.histogram(deaths, bins=BETTI_EDGES)
    return hist.astype(np.float64)


def barcode_a(distance_matrix):
    """6 finite-H0-death stats [mean, std0, max, min, sum, count] on the ``MAX_EDGE`` Rips."""
    deaths = _h0_finite_deaths(distance_matrix, MAX_EDGE)
    if deaths.size == 0:
        return np.zeros(6)
    return np.array([deaths.mean(), deaths.std(ddof=0), deaths.max(),
                     deaths.min(), deaths.sum(), float(deaths.size)])


def spectra_block(distance_matrix):
    """231 values: per filtration [sum, min, max, mean, std0, var0 of nonzero L0 eigenvalues, Betti-0].

    Empty complexes (no edges at that filtration) emit all-zero 7-tuples.
    """
    rc = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=float(FILTRATIONS[-1]))
    st = rc.create_simplex_tree(max_dimension=1)
    cx = Complex(simplex_tree=st)
    out = np.zeros((len(FILTRATIONS), 7))
    for i, t in enumerate(FILTRATIONS):
        ev = np.asarray(cx.spectra(0, float(t), float(t)), dtype=float)
        nz = ev[ev > EIG_TOL]
        if nz.size == 0:
            continue  # leave all seven at zero (matches oracle empty-complex behaviour)
        out[i] = [nz.sum(), nz.min(), nz.max(), nz.mean(), nz.std(ddof=0), nz.var(ddof=0),
                  float((ev <= EIG_TOL).sum())]
    return out.reshape(-1)


def alpha_topo(points, degree):
    """18 alpha-persistence summary stats for homology ``degree`` on a pooled point cloud."""
    if len(points) < 4:
        return np.zeros(18)
    st = gudhi.AlphaComplex(points=points).create_simplex_tree()
    st.compute_persistence()
    iv = st.persistence_intervals_in_dimension(degree)
    iv = iv[np.isfinite(iv[:, 1])] if iv.size else iv
    if iv.size == 0:
        return np.zeros(18)
    b = np.clip(2.0 * np.sqrt(iv[:, 0]), None, ALPHA_CUT)   # 2*sqrt(alpha) = diameter, clipped
    d = np.clip(2.0 * np.sqrt(iv[:, 1]), None, ALPHA_CUT)
    pers = d - b
    keep = pers > ALPHA_FLOOR
    if not keep.any():
        return np.zeros(18)
    B, D, P = b[keep], d[keep], pers[keep]
    j = int(np.argmax(P))
    return np.array([
        P.mean(), P.std(ddof=0), P.max(), P.min(), P.sum(),
        B[j], D[j],
        B.mean(), B.std(ddof=0), B.max(), B.min(), B.sum(),
        D.mean(), D.std(ddof=0), D.max(), D.min(), D.sum(),
        float(len(P)),
    ])


def featurize_channel(protein_pts, peptide_pts):
    """One 306-value channel block for a (protein-element, peptide-element) pairing.

    A channel with an empty side (no atoms of that element on protein or peptide) emits an all-zero
    block: the upstream ``.pyc`` short-circuits the whole channel (betti/barcode/spectra/alpha) to
    zero, verified by a synthetic probe. A single-atom (non-empty) side is *not* zeroed.
    """
    if len(protein_pts) == 0 or len(peptide_pts) == 0:
        return np.zeros(PER_CHANNEL_WIDTH)
    dm = _bipartite_distance_matrix(protein_pts, peptide_pts)
    pooled = np.vstack([protein_pts, peptide_pts]) if len(protein_pts) or len(peptide_pts) \
        else np.empty((0, 3))
    return np.concatenate([
        betti_bins(dm),
        barcode_a(dm),
        spectra_block(dm),
        alpha_topo(pooled, 1),
        alpha_topo(pooled, 2),
    ])


def featurize_interface(path):
    """Interface PDB -> raw (2754,) channel-major descriptor (9 channels x 306)."""
    data = load_interface(path)
    blocks = []
    for a in ELEMENTS:          # protein element
        for b in ELEMENTS:      # peptide element
            blocks.append(featurize_channel(data[a]["protein"], data[b]["peptide"]))
    return np.concatenate(blocks)
