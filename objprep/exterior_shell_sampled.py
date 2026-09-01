"""Area-sampled interior stripping.

The centroid version fails silently on coarse meshes: it rasterises one point per face at a
1.5 mm pixel, so when faces are LARGER than a pixel (gun2_glock19: 12.7k faces, ~3-4 mm each)
every centroid owns its own pixel, nothing occludes anything, and 1.8% gets removed instead of
~40%. Faces here are not uniform across the library -- 12.7k to 63k for the same object class --
so any per-face test has to be resolution-independent.

Fix: scatter samples over each face in proportion to its AREA, so the rasteriser sees a surface
rather than a dot. A face is exterior if ANY of its samples is front-most from ANY view direction
while the face points at that viewer.
"""
from __future__ import annotations
import sys
import numpy as np
import trimesh

sys.path.insert(0, "/home/jonah/Documents/code/model_gen/python")
from bodies import zbuffer_cull


def _fib_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(th) * np.sin(phi),
                            np.sin(th) * np.sin(phi), np.cos(phi)])


def _area_samples(tri, area, pixel, per_pixel=4.0, seed=0):
    """-> (P, fid): points scattered over faces, ~per_pixel samples per pixel of face area."""
    k = np.maximum(1, np.ceil(area / (pixel * pixel) * per_pixel).astype(np.int64))
    fid = np.repeat(np.arange(len(tri)), k)
    rng = np.random.default_rng(seed)
    u = rng.random(len(fid)); v = rng.random(len(fid))
    sw = u + v > 1.0
    u[sw], v[sw] = 1.0 - u[sw], 1.0 - v[sw]           # uniform over the triangle
    A, B, C = tri[fid, 0], tri[fid, 1], tri[fid, 2]
    return A + u[:, None] * (B - A) + v[:, None] * (C - A), fid


def exterior_faces(mesh, n_dirs=64, pixel=0.0015, keep_depth=0.0008, per_pixel=4.0, seed=0):
    """-> bool mask over faces: True where the face is on the exterior shell."""
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces, np.int64)
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(nrm, axis=1) * 0.5
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30

    P, fid = _area_samples(tri, area, pixel, per_pixel, seed)
    vis = np.zeros(len(F), bool)
    for d in _fib_dirs(n_dirs):
        a = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(a, d); u /= np.linalg.norm(u)
        w = np.cross(d, u)
        facing = (nrm @ d) > 0
        sel = facing[fid]                              # only front-facing samples compete
        if not sel.any():
            continue
        idx = np.flatnonzero(sel)
        Q = np.column_stack([P[idx] @ u, -(P[idx] @ d), P[idx] @ w])
        keep = zbuffer_cull(np.ascontiguousarray(Q), pixel=pixel, keep_depth=keep_depth)
        vis[np.unique(fid[idx[np.asarray(keep).ravel()]])] = True
    return vis


def strip_interior(mesh, **kw):
    """-> (exterior-only Trimesh, stats dict)."""
    keep = exterior_faces(mesh, **kw)
    out = mesh.copy()
    out.update_faces(keep)
    out.remove_unreferenced_vertices()
    stats = dict(faces_in=len(mesh.faces), faces_out=len(out.faces),
                 area_in=mesh.area, area_out=out.area,
                 interior_frac=1.0 - keep.mean())
    return out, stats


if __name__ == "__main__":
    m = trimesh.load(sys.argv[1], process=True)
    out, st = strip_interior(m)
    out.fix_normals()
    out.export(sys.argv[2])
    print(st)
