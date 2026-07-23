"""Reusable low-discrepancy surface sampler for the THz forward model.

Extracted from the man_gun floor-sweep `sample_stl.py` so ANY mesh (STL, SMPL-X,
trimesh primitive) can be sampled the same way: area-weighted Hammersley points on
the triangles, front-culled toward the array, with exact per-point face normals.

Frame convention (sim frame): x=width, y=depth (array at y=0 looking +y, body at
y=standoff), z=height. A face is front-facing (visible to the array) when its
outward normal has n_y < 0. Pass verts ALREADY in this frame.

Calibration: N = A_front / (calib * d^2) lands median NN ~= d for the low-discrepancy
fill (calib=2.67 measured on the man_gun STL, 0.30-0.075 lambda).
"""
from __future__ import annotations
import numpy as np


def _vdc(k, base):
    """Van der Corput low-discrepancy sequence, vectorized over integer array k."""
    out = np.zeros(len(k), np.float64); f = 1.0 / base; kk = k.astype(np.int64).copy()
    while kk.any():
        out += (kk % base) * f; kk //= base; f /= base
    return out


def face_normals_areas(verts, faces, given=None):
    """Per-face unit normal + area. Uses `given` normals if provided (e.g. STL's
    stored normals), else computes from winding (consistent-outward meshes)."""
    v = np.asarray(verts, np.float64)
    tri = v[faces]                                       # (F,3,3)
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n = np.asarray(given, np.float64) if given is not None else cr
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)
    return n, area


def sample_mesh(verts, faces, target_nn, *, face_normals=None, front_cull=True,
                calib=2.67, chunk=4_000_000):
    """Low-discrepancy surface sample of a mesh (verts in sim frame).

    Returns (q, n, info): q (M,3) float32 point positions, n (M,3) float32 per-point
    face normals, info dict {N, area, front_faces}. Front-cull keeps faces with n_y<0.
    """
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)
    nrm, area = face_normals_areas(verts, faces, face_normals)
    mask = nrm[:, 1] < 0.0 if front_cull else np.ones(len(faces), bool)
    Vf = verts[faces[mask]]                              # (Ff,3,3)
    Nf = nrm[mask]; af = area[mask]
    A = float(af.sum())
    N = int(round(A / (calib * target_nn ** 2)))
    cumA = np.cumsum(af); cumA /= cumA[-1]
    qs, ns = [], []
    for c0 in range(0, N, chunk):
        k = np.arange(c0, min(c0 + chunk, N))
        u = (k + 0.5) / N                                # stratified facet-CDF coord
        fi = np.clip(np.searchsorted(cumA, u, side="right"), 0, len(af) - 1)
        r1 = _vdc(k, 2); r2 = _vdc(k, 3)                 # within-facet Hammersley
        su = np.sqrt(r1)
        b0 = (1 - su)[:, None]; b1 = (su * (1 - r2))[:, None]; b2 = (su * r2)[:, None]
        tri = Vf[fi]
        p = b0 * tri[:, 0] + b1 * tri[:, 1] + b2 * tri[:, 2]
        qs.append(p.astype(np.float32)); ns.append(Nf[fi].astype(np.float32))
    q = np.vstack(qs); n = np.vstack(ns)
    return q, n, dict(N=N, area=A, front_faces=int(mask.sum()))


def median_nn(q, lam, sub=200_000, seed=0):
    """Median nearest-neighbour spacing (mm and lambda), scipy cKDTree, sub+correct."""
    from scipy.spatial import cKDTree
    q = np.asarray(q, np.float64)
    if len(q) <= sub:
        s = q; corr = 1.0
    else:
        s = q[np.random.default_rng(seed).choice(len(q), sub, replace=False)]
        corr = np.sqrt(sub / len(q))
    d = cKDTree(s).query(s, k=2, workers=-1)[0][:, 1]
    med = np.median(d) * corr
    return med * 1e3, med / lam
