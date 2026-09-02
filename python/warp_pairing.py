"""GPU (NVIDIA Warp) reimplementation of the two multipath pairing bottlenecks.

Faithful to multipath.stationary_pairs / multipath.return_visible: SAME point clouds, SAME params,
SAME marched-nearest-neighbour logic -- only the scipy.cKDTree ray-march (single-threaded, ~750 s
for a hip site) is replaced by a Warp HashGrid radius query launched one-thread-per-point on the GPU.

A within-radius query is sufficient (not scipy's global k=1) because a surviving partner must sit
within max_perp of the reflected ray (stationary_pairs) or within r_block of the segment
(return_visible); anything farther is dropped by the same threshold anyway.
"""
import numpy as np
import warp as wp

wp.init()
_DEV = "cuda:0"


@wp.kernel
def _k_stationary(qA: wp.array(dtype=wp.vec3), nA: wp.array(dtype=wp.vec3),
                  qB: wp.array(dtype=wp.vec3), nB: wp.array(dtype=wp.vec3),
                  grid: wp.uint64, tx: wp.vec3,
                  max_perp: float, max_path: float, step: float, face_tol: float,
                  nstep: int, qradius: float,
                  out_ib: wp.array(dtype=wp.int32), out_keep: wp.array(dtype=wp.int32)):
    a = wp.tid()
    pa = qA[a]
    na = nA[a]
    din = pa - tx
    din = din / (wp.length(din) + 1.0e-12)
    drefl = din - 2.0 * wp.dot(din, na) * na
    drefl = drefl / (wp.length(drefl) + 1.0e-12)

    best_perp = float(1.0e30)
    best_ib = int(-1)
    # march the reflected ray; at each step gather skin points near the marched position and keep
    # the one with the smallest perpendicular distance to the ray (matches the scipy min-perp pick)
    for s in range(nstep):
        tt = step + (max_path - step) * float(s) / float(nstep - 1)
        pos = pa + tt * drefl
        query = wp.hash_grid_query(grid, pos, qradius)
        b = int(0)
        while wp.hash_grid_query_next(query, b):    # Warp 1.17: b IS the original point id
            v = qB[b] - pa
            tpar = wp.dot(v, drefl)
            if tpar > 1.0e-4 and tpar <= max_path:
                perp = wp.length(v - tpar * drefl)
                if perp < best_perp:
                    best_perp = perp
                    best_ib = b

    out_ib[a] = best_ib
    keep = int(0)
    if best_ib >= 0 and best_perp <= max_perp:
        v = qB[best_ib] - pa
        r = wp.length(v) + 1.0e-12
        faceA = wp.dot(v, na) > face_tol * r
        faceB = wp.dot(-v, nB[best_ib]) > face_tol * r
        if faceA and faceB:
            keep = 1
    out_keep[a] = keep


def stationary_pairs(qA, nA, qB, nB, tx, *, max_perp=0.0015, max_path=0.05, step=0.001,
                     face_tol=0.02):
    """Warp version of multipath.stationary_pairs. Returns (ia, ib) index arrays."""
    qA = np.ascontiguousarray(qA, np.float32); nA = np.ascontiguousarray(nA, np.float32)
    qB = np.ascontiguousarray(qB, np.float32); nB = np.ascontiguousarray(nB, np.float32)
    t = np.asarray(tx, np.float32).reshape(-1, 3)[0]
    Na = len(qA)
    nstep = max(6, int(np.ceil(max_path / max(0.5 * max_perp, 1e-4))))
    qradius = float(max_perp + 1.5 * step)               # capture any point that can end <= max_perp

    d_qA = wp.array(qA, dtype=wp.vec3, device=_DEV); d_nA = wp.array(nA, dtype=wp.vec3, device=_DEV)
    d_qB = wp.array(qB, dtype=wp.vec3, device=_DEV); d_nB = wp.array(nB, dtype=wp.vec3, device=_DEV)
    grid = wp.HashGrid(128, 128, 128, device=_DEV)
    grid.build(points=d_qB, radius=qradius)
    out_ib = wp.zeros(Na, dtype=wp.int32, device=_DEV)
    out_keep = wp.zeros(Na, dtype=wp.int32, device=_DEV)
    wp.launch(_k_stationary, dim=Na,
              inputs=[d_qA, d_nA, d_qB, d_nB, grid.id, wp.vec3(*t),
                      float(max_perp), float(max_path), float(step), float(face_tol),
                      int(nstep), qradius, out_ib, out_keep], device=_DEV)
    wp.synchronize_device(_DEV)
    ib = out_ib.numpy(); keep = out_keep.numpy().astype(bool)
    ia = np.flatnonzero(keep)
    return ia, ib[keep]


@wp.kernel
def _k_return(pts: wp.array(dtype=wp.vec3), occ: wp.array(dtype=wp.vec3),
              grid: wp.uint64, rxc: wp.vec3,
              r_block: float, self_skip: float, max_t: float, step: float, nstep: int,
              out_clear: wp.array(dtype=wp.int32)):
    i = wp.tid()
    p = pts[i]
    d = rxc - p
    L = wp.length(d)
    dhat = d / (L + 1.0e-12)
    clear = int(1)
    for s in range(nstep):
        tt = self_skip + step * float(s)
        if tt >= max_t:
            break
        if clear == 1 and tt < (L - self_skip):
            pos = p + tt * dhat
            query = wp.hash_grid_query(grid, pos, r_block)
            b = int(0)
            while wp.hash_grid_query_next(query, b):    # Warp 1.17: b IS the original point id
                if wp.length(pos - occ[b]) < r_block:
                    clear = 0
    out_clear[i] = clear


def return_visible(pts, rx_center, occ, *, r_block=0.0012, self_skip=0.003, max_t=0.25,
                   step=0.0008):
    """Warp version of multipath.return_visible. Returns a bool mask (True = clear LOS)."""
    pts = np.ascontiguousarray(pts, np.float32); occ = np.ascontiguousarray(occ, np.float32)
    c = np.asarray(rx_center, np.float32).reshape(3)
    M = len(pts)
    nstep = int(np.ceil((max_t - self_skip) / step)) + 1
    d_pts = wp.array(pts, dtype=wp.vec3, device=_DEV)
    d_occ = wp.array(occ, dtype=wp.vec3, device=_DEV)
    grid = wp.HashGrid(128, 128, 128, device=_DEV)
    grid.build(points=d_occ, radius=float(r_block))
    out_clear = wp.zeros(M, dtype=wp.int32, device=_DEV)
    wp.launch(_k_return, dim=M,
              inputs=[d_pts, d_occ, grid.id, wp.vec3(*c),
                      float(r_block), float(self_skip), float(max_t), float(step), int(nstep),
                      out_clear], device=_DEV)
    wp.synchronize_device(_DEV)
    return out_clear.numpy().astype(bool)


@wp.kernel
def _k_contact(qP: wp.array(dtype=wp.vec3), qQ: wp.array(dtype=wp.vec3),
               grid: wp.uint64, dist: float, out_keep: wp.array(dtype=wp.int32)):
    i = wp.tid()
    p = qP[i]
    keep = int(0)
    query = wp.hash_grid_query(grid, p, dist)
    b = int(0)
    while wp.hash_grid_query_next(query, b):        # Warp 1.17: b IS the original point id
        if wp.length(p - qQ[b]) <= dist:
            keep = 1
    out_keep[i] = keep


def contact_band(qP, qQ, dist):
    """Warp version of multipath.contact_band: indices of qP within `dist` of ANY qQ point.
    A pure within-radius existence test (scipy uses k=1 nearest but only compares to `dist`)."""
    qP = np.ascontiguousarray(qP, np.float32); qQ = np.ascontiguousarray(qQ, np.float32)
    Np = len(qP)
    if Np == 0 or len(qQ) == 0:
        return np.empty(0, np.int64)
    d_qP = wp.array(qP, dtype=wp.vec3, device=_DEV)
    d_qQ = wp.array(qQ, dtype=wp.vec3, device=_DEV)
    grid = wp.HashGrid(128, 128, 128, device=_DEV)
    grid.build(points=d_qQ, radius=float(dist))
    out_keep = wp.zeros(Np, dtype=wp.int32, device=_DEV)
    wp.launch(_k_contact, dim=Np,
              inputs=[d_qP, d_qQ, grid.id, float(dist), out_keep], device=_DEV)
    wp.synchronize_device(_DEV)
    return np.flatnonzero(out_keep.numpy())
