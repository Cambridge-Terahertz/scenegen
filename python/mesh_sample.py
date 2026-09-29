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
    if front_cull:
        _assert_sim_frame(verts, faces, mask)
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

def _assert_sim_frame(verts, faces, mask, *, sample=60_000, frac_limit=0.25):
    """Guard: front_cull assumes SIM-FRAME verts (y = depth from the array).

    Passing SMPL-X verts (y = UP) instead silently keeps DOWNWARD-facing faces rather
    than ARRAY-facing ones -- roughly half the cloud becomes the wrong surface, and the
    error is invisible until someone looks at a depth map. It shipped in four scripts
    before being caught (2026-09-28).

    INVARIANT: a correct front-cull leaves ONE surface per line of sight, so binning the
    kept faces by the two cross-range axes gives a SMALL depth spread per cell. A
    wrong-frame cull mixes front and back, so the spread approaches the object's whole
    extent. Measured on a subdivided SMPL-X body: 1.8 cm correct vs 82.7 cm wrong -- a
    45x separation, so the threshold is not delicate.
    """
    idx = np.flatnonzero(mask)
    if len(idx) < 200:
        return
    if len(idx) > sample:
        idx = idx[np.linspace(0, len(idx) - 1, sample).astype(np.int64)]
    c = verts[faces[idx]].mean(1)
    ext = c.max(0) - c.min(0)
    if ext[1] <= 0:
        return
    cell = max(float(max(ext[0], ext[2])) / 50.0, 1e-6)
    key = (np.floor(c[:, 0] / cell).astype(np.int64) * 1_000_003
           + np.floor(c[:, 2] / cell).astype(np.int64))
    o = np.argsort(key, kind="stable")
    ks, ys = key[o], c[o, 1]
    b = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1, len(ks)]
    sp = [np.ptp(ys[b[i]:b[i + 1]]) for i in range(len(b) - 1) if b[i + 1] - b[i] > 3]
    if len(sp) < 20:
        return
    frac = float(np.median(sp)) / float(ext[1])
    if frac > frac_limit:
        raise ValueError(
            f"sample_mesh(front_cull=True): the kept surface has a median per-cell depth "
            f"spread of {frac*100:.0f}% of the total extent (expected <{frac_limit*100:.0f}%). "
            f"The kept set is NOT a single visible surface, which means `verts` are almost "
            f"certainly not in the SIM frame. front_cull keeps faces with n_y < 0, where y "
            f"must be DEPTH FROM THE ARRAY. If these are SMPL-X verts (y = UP), transform "
            f"first:\n"
            f"    verts_sim = to_sim_frame(np.asarray(verts, np.float64), standoff=...)\n"
            f"    sample_mesh(verts_sim, faces, target_nn, front_cull=True)\n"
            f"(to_sim_frame is a det +1 rotation, so sampling before vs after is "
            f"equivalent -- only the cull moves.) Pass front_cull=False to skip this.")
